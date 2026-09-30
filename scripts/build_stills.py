#!/usr/bin/env python3
"""Build a Short from composed stills - no video model, no API, no GPU.

WHY THIS ROUTE
--------------
Every free route fails a different constraint: Google Flow is free and good but
a web UI, HF Inference Providers is scriptable but costs more than $0.10/month
in credits, and this sandbox has 2 cores, 1 GB RAM and no GPU so nothing can be
self-hosted here. Composing stills needs no model at all, so it is the only
route that is simultaneously free, card-free, automatic and runnable here.

It is also not a compromise for this content. The pacing research says a Short
wants a visual change every 2-4 seconds and CoComelon cuts every 1-3 seconds, so
a sequence of composed stills with hard cuts sits inside the format's norms. The
real limits: no new imagery is invented, so every shot is a recomposition of
frames already on disk, and there is no generated motion - only camera motion.

WHAT IT PRODUCES
----------------
One shot per beat, hard cuts, 9:16 1080x1920 @ 25 fps, each shot carrying:
  * a different recomposition (scale, position, mirror) of the character
  * a Ken Burns move, so nothing is a frozen frame
  * a subject-locked crop derived from the MEASURED character box, so the
    character can never leave frame
plus the synthesised music bed from gen_audio.py. Deterministic from the seed.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_plates                                    # noqa: E402
import composite                                    # noqa: E402
import gen_audio                                    # noqa: E402

W, H, FPS = 1080, 1920, 25
PLATE_W, PLATE_H = 720, 1280      # portrait: matches what we actually deliver
SHOT_SECS = 2.5        # pacing research: a visual change every 2-4 seconds

# scale is a fraction of plate HEIGHT; cx/cy are the character centre as a
# fraction of the plate, so he can be moved without leaving frame. mirror
# alternates so consecutive shots read as different angles.
#               scale    cx    cy   mirror  crop_top crop_bot
SHOT_TABLE = [
    (0.62,  0.50, 0.60, False, 0.00, 1.00),   # wide   - full body, room to move
    (0.88,  0.50, 0.56, True,  0.04, 0.96),   # medium - waist up, flipped
    (1.00,  0.50, 0.42, False, 0.00, 0.42),   # face   - head and shoulders
    (0.70,  0.44, 0.42, True,  0.12, 1.00),   # low    - from below, flipped
    (0.66,  0.56, 0.64, False, 0.00, 0.78),   # high   - upper body, from above
]


def _shot(idx):
    """Shot spec for index idx, with a second pass over the table for variety.

    There is only one key frame, so every shot is a recomposition of it. A
    second pass re-uses the same five framings slightly tighter and shifted, so
    a 25-second Short gets ten distinguishable shots instead of five repeats.
    """
    scale, cx, cy, mirror, c0, c1 = SHOT_TABLE[idx % len(SHOT_TABLE)]
    lap = idx // len(SHOT_TABLE)
    if lap:
        scale *= max(0.5, 1.0 - 0.08 * lap)
        cy = min(0.74, cy + 0.05 * lap)
        cx = min(0.62, max(0.38, cx - 0.06 * lap))
        mirror = not mirror
    return scale, cx, cy, mirror, c0, c1

# Ken Burns start/end zoom and pan, as fractions of the plate. Deliberately
# small - the user rejected visible camera moves, and the research says one
# clean slow move beats several messy ones.
#                       zoom0 zoom1    panx  pany
MOVE_TABLE = [
    ((1.00, 1.00), (-0.02,  0.02)),
    ((1.08, 1.00), ( 0.00,  0.00)),
    ((1.00, 1.06), ( 0.01, -0.01)),
    ((1.06, 1.00), ( 0.00,  0.03)),
    ((1.00, 1.10), (-0.03,  0.00)),
]


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _window(ph, pw, box, i, n, move):
    """9:16 crop window for frame i, aimed at the character's measured centre.

    Clamped so it never samples outside the plate. Derived from the box the
    compositor measured, not a fixed rect, so the subject stays in frame even
    when the shot's scale or position changes.
    """
    (z0, z1), (px, py) = move
    t = i / max(n - 1, 1)
    zoom = z0 + (z1 - z0) * t

    win_w = int(ph * 9.0 / 16.0)
    win_h = ph
    if win_w > pw:                              # plate narrower than window
        win_w, win_h = pw, int(pw * 16.0 / 9.0)

    ww = min(int(win_w / zoom), pw)
    wh = min(int(win_h / zoom), ph)

    tx = box[0] + box[2] / 2.0 + px * pw * t
    ty = box[1] + box[3] / 2.0 + py * ph * t
    cx = int(np.clip(tx - ww / 2.0, 0, pw - ww))
    cy = int(np.clip(ty - wh / 2.0, 0, ph - wh))
    return cx, cy, ww, wh


def _render_shot(shot_frame, box, idx, seconds, out_path, tmpdir):
    import cv2
    n = max(2, int(round(seconds * FPS)))
    for i in range(n):
        cx, cy, ww, wh = _window(shot_frame.shape[0], shot_frame.shape[1],
                                 box, i, n, MOVE_TABLE[idx % len(MOVE_TABLE)])
        cv2.imwrite(os.path.join(tmpdir, f"_f{i:04d}.png"),
                    shot_frame[cy:cy + wh, cx:cx + ww])
    subprocess.run([_ffmpeg(), "-y", "-framerate", str(FPS),
                    "-i", os.path.join(tmpdir, "_f%04d.png"),
                    "-vf", f"scale={W}:{H}", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-r", str(FPS), out_path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for i in range(n):
        os.remove(os.path.join(tmpdir, f"_f{i:04d}.png"))
    return out_path


def build(package, key_bgr, out_path, seed=7, workdir=".", plate_name=None):
    """Render the whole Short. Returns out_path."""
    os.makedirs(workdir, exist_ok=True)
    # the scene lives on each clip, not on the package - use the first clip's
    # plate so every shot in a Short shares one room, as the model did
    clips = package.get("clips", [])
    if plate_name is None:
        plate_name = (clips[0].get("scene") if clips else None) or "playroom"
    # Portrait plate, natively drawn. A 9:16 crop of a 16:9 plate would use only
    # 37% of the drawn room and throw the rest away; the stills route has no
    # model forcing landscape, so ask for the aspect we actually deliver.
    plate = bg_plates.plate(plate_name, seed, w=PLATE_W, h=PLATE_H)

    # The key frame MUST be the clean cut-out on white. Feeding
    # hoji_scene_key.png - which is already a composite on a plate - makes
    # character_matte's "bright and neutral" test fail, because the plate behind
    # him is now full of colour. The matte then keeps a rectangle of room around
    # him and he renders as a sticker pasted on the wall.
    if key_bgr is None:
        import cv2
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_bgr = cv2.imread(os.path.join(here, "assets", "hoji_key_720.png"))
    key_h, key_w = key_bgr.shape[:2]
    if key_w > key_h:                     # a landscape plate was passed in
        raise SystemExit(
            "build_stills needs the CLEAN portrait key (assets/hoji_key_720.png), "
            f"got {key_w}x{key_h}. A composite key has no white surround, so the "
            "matte cannot separate the character from the background.")

    # beats are ABSOLUTE SFX timestamps (t runs 0.0 -> 24.2), not durations.
    # Summing them - the obvious reading - yields 298s and a 5-minute "Short".
    # The real length is the last event plus its decay.
    beats = package.get("beats", [])
    tail = 1.0
    total = (max(float(b.get("t", 0.0)) for b in beats) + tail) if beats else 25.0

    # one shot per beat-pair, not per beat: 25 SFX events are a beat grid, not a
    # shot list. The pacing research wants a visual change every 2-4 seconds, so
    # aim for SHOT_SECS rather than one shot per clip.
    n_shots = max(5, int(round(total / SHOT_SECS)))
    per_shot = total / n_shots

    kh = key_bgr.shape[0]
    shots = []
    for idx in range(n_shots):
        scale, cx, cy, mirror, c0, c1 = _shot(idx)
        src = key_bgr[int(c0 * kh):int(c1 * kh)]
        if mirror:
            src = src[:, ::-1].copy()
        frame, _matte, box = composite.composite(
            src, plate, target_h_frac=scale, cx_frac=cx, cy_frac=cy)
        p = os.path.join(workdir, f"shot_{idx:02d}.mp4")
        _render_shot(frame, box, idx, per_shot, p, workdir)
        shots.append(p)

    concat = os.path.join(workdir, "concat.txt")
    with open(concat, "w") as fh:
        for p in shots:
            fh.write(f"file '{os.path.abspath(p)}'\n")
    tmp = os.path.join(workdir, "_silent.mp4")
    subprocess.run([_ffmpeg(), "-y", "-f", "concat", "-safe", "0",
                    "-i", concat, "-c", "copy", tmp], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total = per_shot * len(beats)
    audio = os.path.join(workdir, "audio.wav")
    gen_audio._write(audio, gen_audio.music_loop(seconds=total, bpm=112,
                                                 seed=seed))

    subprocess.run([_ffmpeg(), "-y", "-i", tmp, "-i", audio,
                    "-filter_complex",
                    f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                    f"crop={W}:{H},format=yuv420p[v]",
                    "-map", "[v]", "-map", "1:a", "-shortest",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-maxrate", "8M", "-bufsize", "16M",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", out_path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


if __name__ == "__main__":
    import json
    import cv2
    pkg = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    key = cv2.imread(sys.argv[2] if len(sys.argv) > 2
                     else "../assets/hoji_scene_key.png")
    out = build(pkg, key, sys.argv[3] if len(sys.argv) > 3 else "stills.mp4",
                workdir=os.path.abspath("_stills"))
    print("wrote", out)
