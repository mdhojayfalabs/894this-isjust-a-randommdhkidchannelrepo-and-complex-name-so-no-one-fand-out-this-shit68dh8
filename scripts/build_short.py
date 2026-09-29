#!/usr/bin/env python3
"""
MDH Kids Studio - Short builder, v2.

    python scripts/build_short.py --package day_04 --out videos/day_04.mp4

WHAT CHANGED AND WHY
--------------------
v1 fed the video model a tight studio crop of the character on white. The owner
rejected the result for three defects, each confirmed by measurement:

  * "the character zooms in and out inside the frame" - measured: character
    height swung 428-640px (36% of mean) and vertical centre 92px across one
    Short. A/B test proved the cause is not the prompt: `end_image`,
    `camera_motion`, `negative_prompt` and `seed` are all accepted with HTTP 202
    and then silently ignored, and character width still shrank 33-47% per clip.
    The camera is simply not controllable at generation time.
  * "nothing behind him" - with no scene in the key frame the model invented a
    different one per clip (white void, then orange).
  * "so much light behind that you can't make him out" - no lighting described,
    so it defaulted to a blown-out key behind the subject.

v2 fixes all three structurally rather than by prompting:

  1. A scene is BUILT, not requested. scripts/bg_plates.py draws a lit room and
     scripts/composite.py puts the character in it at a known place. The model
     now receives a complete frame with structure to hold onto. This also fixes
     the framing: the character is scaled to ~58% of frame height so there is
     headroom and a room around him, instead of touching all four edges.
  2. The prompt describes MOTION ONLY. Re-describing the scene in an
     image-to-video prompt is a documented cause of drift, so v1's long scene
     paragraph was actively harmful. Camera direction is stated positively
     ("tripod-locked stationary shot"), never as "no zoom / no pan" - models
     read the noun and do the thing.
  3. The subject is LOCKED IN POST. Because the plate is known, the character's
     mask in every generated frame is recovered by differencing against the
     plate - far more reliable than colour thresholding, which matched 99% of
     the frame on this material. That mask drives a smoothed affine that holds
     the subject in one framing, and a horizontal crop that follows him.
  4. The grade pulls the blown-out key back in and restores local contrast.

The generation call itself is unchanged: POST /ltx-video/v1/image-to-video,
poll /v2/requests/status/{id}. The image is passed as a data: URI, so no
external hosting and no credential is needed for the key frame.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import bg_plates          # noqa: E402
import composite as comp  # noqa: E402
import remaster as rm     # noqa: E402
from gen_audio import mix_voice, music_loop, _write  # noqa: E402

GATEWAY = "https://gateway.pixazo.ai"
SUBMIT = "/ltx-video/v1/image-to-video"
STATUS = "/v2/requests/status/{rid}"

# Landscape is what the model emits; vertical is manufactured afterwards.
GW, GH = 1280, 704
# 9:16 crop taken out of that landscape frame.
VW, VH = 396, 704

HOJI_ANCHOR = (
    "Hoji is a toddler boy with DARK BROWN spiky hair, THICK BLACK ROUND "
    "GLASSES, a small BROWN GOATEE, BROWN skin, wearing an ORANGE-RED SOCCER "
    "JERSEY with a RAINBOW-STAR CREST on the chest, NAVY BLUE SHORTS and "
    "COLOURFUL BLUE RED YELLOW SNEAKERS. Keep his face, hair colour, glasses "
    "and outfit EXACTLY as in the reference image. "
)
# Style carries no negatives: "no text on screen" reads as "text on screen".
STYLE = ("3D Pixar animation, ultra bright studio lighting, saturated colours, "
         "smooth 24fps motion. ")
# Closing clause, positive phrasing only. "no zoom" makes it zoom.
CAMERA = ("Tripod-locked stationary shot, motionless camera, subject motion "
          "only, fixed framing throughout. ")

DEFAULT_KEY = str(ROOT / "assets" / "hoji_key_720.png")
DEFAULT_SCENE = "playroom"


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        for c in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.exists(c):
                return c
        raise SystemExit("ffmpeg not found: pip install imageio-ffmpeg")


FF = _ffmpeg()


def sh(args, **kw):
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, args[:4]))}... failed:\n{r.stderr[-800:]}")
    return r


# ---------------------------------------------------------------- key frame
def landscape_key(key_bgr, scene, seed, char_h_frac=0.58, cx_frac=0.50):
    """Build the landscape key frame the model is given.

    Returns (image_bgr, plate_bgr, char_box). The character is placed at
    char_h_frac of the frame's height, aspect preserved, with real headroom -
    the v1 key frame had him touching all four edges, which is why the model
    framed him edge-to-edge and had no room to put a scene.
    """
    plate = bg_plates.plate(scene, seed, GW, GH)
    img, matte, box = comp.composite(key_bgr, plate,
                                     target_h_frac=char_h_frac,
                                     cx_frac=cx_frac, cy_frac=0.50, shade=0.18)
    if box is None:
        raise RuntimeError("character not found in the key frame")
    return img, plate, box


# ---------------------------------------------------------------- pixazo
class Pixazo:
    def __init__(self, key):
        self.key = key

    def _post(self, path, payload):
        d = json.dumps(payload).encode()
        req = urllib.request.Request(GATEWAY + path, data=d, method="POST", headers={
            "Content-Type": "application/json", "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    def submit(self, motion, image_bgr):
        # data: URI - no hosting, no credential in any URL
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("could not encode the key frame")
        uri = "data:image/png;base64," + __import__("base64").b64encode(buf.tobytes()).decode()
        j = self._post(SUBMIT, {"prompt": HOJI_ANCHOR + STYLE + CAMERA + motion,
                                "image": uri})
        rid = j.get("request_id")
        if not rid:
            raise RuntimeError(f"submit failed: {j}")
        return rid

    def wait(self, rid, timeout=900):
        st = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            req = urllib.request.Request(GATEWAY + STATUS.format(rid=rid), headers={
                "Ocp-Apim-Subscription-Key": self.key, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                j = json.loads(r.read())
            st = j.get("status")
            if st == "COMPLETED":
                urls = j.get("output", {}).get("media_url") or []
                if not urls:
                    raise RuntimeError(f"COMPLETED but no media_url: {j}")
                return urls[0]
            if st in ("ERROR", "FAILED"):
                raise RuntimeError(f"generation {st}: {j.get('error')}")
            time.sleep(10)
        raise TimeoutError(f"{rid} still {st} after {timeout}s")

    def fetch(self, url, dest):
        subprocess.run(["curl", "-s", "-A", "Mozilla/5.0", "-o", dest, url],
                       check=True)
        return dest


# ---------------------------------------------------------------- frames
def read_frames(path, w=GW, h=GH):
    tmp = path + ".frames"
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    sh([FF, "-y", "-i", path, "-vf", f"scale={w}:{h}", os.path.join(tmp, "f_%05d.png")])
    names = sorted(os.listdir(tmp))
    return [cv2.imread(os.path.join(tmp, n)) for n in names]


def write_frames(frames, path, fps=25, audio=None):
    tmp = path + ".frames"
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    for i, fr in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, "f_%05d.png" % i), fr)
    args = [FF, "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%05d.png")]
    if audio:
        args += ["-i", audio, "-c:a", "aac", "-b:a", "128k", "-shortest"]
    args += ["-c:v", "libx264", "-preset", "slow", "-crf", "19",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    sh(args)


# ---------------------------------------------------------------- verticalise
def verticalise(frames, plate_l, target_frac=0.60, target_cy=0.50,
                smooth=0.30, max_gain=1.22):
    """Crop 9:16 around the subject and lock his scale and vertical position.

    Two corrections, both driven by the character's mask recovered by
    differencing each generated frame against the known plate:

      horizontal - the 396px-wide window is centred on him every frame, so a pan
                   becomes a pan of the world around a still subject
      vertical   - a smoothed affine holds his height at target_frac and his
                   centre at target_cy, damped to max_gain so real motion
                   survives
    """
    boxes = []
    for f in frames:
        m = comp.diff_matte(f, plate_l, thresh=22, feather=3)
        ys, xs = np.nonzero(m > 100)
        if not len(xs):
            boxes.append(None)
            continue
        boxes.append((int(xs.min()), int(ys.min()),
                      int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min())))

    ok = [b for b in boxes if b]
    if not ok:
        out = []
        for f in frames:
            x0 = (GW - VW) // 2
            out.append(cv2.resize(f[:, x0:x0 + VW], (VW * 2, VH * 2),
                                  interpolation=cv2.INTER_LANCZOS4))
        return out, boxes

    # horizontal centre, smoothed
    cx = np.array([b[0] + b[2] / 2.0 if b else np.nan for b in boxes])
    cx = pd_fill(cx)
    cx = pd_smooth(cx, smooth)

    mats = rm.lock_transforms(boxes, target_frac=target_frac,
                              target_cy=target_cy, smooth=smooth,
                              W=GW, H=GH, max_gain=max_gain)

    out = []
    for f, m, c in zip(frames, mats, cx):
        w = cv2.warpAffine(f, m, (GW, GH), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        x0 = int(np.clip(round(c - VW / 2.0), 0, GW - VW))
        crop = w[:, x0:x0 + VW]
        out.append(cv2.resize(crop, (VW * 2, VH * 2),
                              interpolation=cv2.INTER_LANCZOS4))
    return out, boxes


def pd_fill(a):
    a = a.copy()
    idx = np.arange(len(a))
    good = ~np.isnan(a)
    if good.sum() == 0:
        return np.zeros_like(a)
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def pd_smooth(a, s):
    if len(a) < 3 or s <= 0:
        return a
    f = a.copy()
    for i in range(1, len(f)):
        f[i] = s * f[i - 1] + (1 - s) * f[i]
    b = a.copy()
    for i in range(len(b) - 2, -1, -1):
        b[i] = s * b[i + 1] + (1 - s) * b[i]
    return 0.5 * (f + b)


# ---------------------------------------------------------------- clip
def build_clip(px, motion, scene, seed, key_bgr, workdir, idx,
               char_h_frac=0.58, retry=2):
    """One clip: key frame -> generate -> verticalise -> grade. Returns frames."""
    img, plate, box = landscape_key(key_bgr, scene, seed,
                                    char_h_frac=char_h_frac)
    kf = os.path.join(workdir, f"clip_{idx}_key.png")
    cv2.imwrite(kf, img)
    pl = os.path.join(workdir, f"clip_{idx}_plate.png")
    cv2.imwrite(pl, plate)

    last = None
    for attempt in range(retry + 1):
        try:
            rid = px.submit(motion, img)
            print(f"  clip {idx}: submitted {rid} (attempt {attempt + 1})", flush=True)
            url = px.wait(rid)
            raw = os.path.join(workdir, f"clip_{idx}_raw.mp4")
            px.fetch(url, raw)
            break
        except Exception as e:
            last = e
            print(f"  clip {idx}: attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(15)
    else:
        raise RuntimeError(f"clip {idx} failed after {retry + 1} attempts: {last}")

    frames = read_frames(raw, GW, GH)
    print(f"  clip {idx}: {len(frames)} frames generated", flush=True)
    vert, boxes = verticalise(frames, plate)
    graded = [rm.grade(f) for f in vert]
    return graded, plate


# ---------------------------------------------------------------- package
def load_package(name):
    p = ROOT / "packages" / f"{name}.json"
    if not p.exists():
        p = ROOT / "packages" / f"{name}.yaml"
    if not p.exists():
        raise SystemExit(f"no package {name}")
    return json.loads(p.read_text())


def motion_prompt(clip, i):
    """Motion only. Never re-describe the scene - that is what causes drift."""
    if isinstance(clip, dict):
        return clip.get("motion") or clip.get("prompt") or clip.get("shot", "")
    return clip


def clip_scene(clip, default):
    """The room this clip plays in, from the package."""
    if isinstance(clip, dict):
        return clip.get("scene") or default
    return default


def build(pkg, dest, image_url=None, key=None, workdir=None,
          char_h_frac=0.80, key_frame=None):
    """Library entry point used by generate_day.py.

    `image_url` is accepted and ignored: the key frame is now built locally and
    passed to the gateway as a data: URI, so no hosting and no credential in a
    URL is involved. It stays in the signature because generate_day.py passes it.
    """
    if not key:
        raise SystemExit("PIXAZO_API_KEY is not set")
    px = Pixazo(key)
    kf = key_frame or os.environ.get("HOJI_SCENE_KEY") or str(ROOT / "assets" / "hoji_scene_key.png")
    key_bgr = cv2.imread(kf)
    if key_bgr is None:
        raise SystemExit(f"cannot read key frame {kf}")

    workdir = str(workdir or (ROOT / "build" / f"day_{pkg.get('day', 0):02d}"))
    os.makedirs(workdir, exist_ok=True)
    seed0 = int(pkg.get("seed", 7))

    all_frames = []
    for i, clip in enumerate(pkg["clips"]):
        cache = os.path.join(workdir, f"clip_{i}_final.mp4")
        if os.path.exists(cache):
            all_frames += read_frames(cache, VW * 2, VH * 2)
            print(f"  clip {i}: reused cache", flush=True)
            continue
        scene = clip_scene(clip, DEFAULT_SCENE)
        frames, _ = build_clip(px, motion_prompt(clip, i), scene, seed0 + i,
                               key_bgr, workdir, i, char_h_frac=char_h_frac)
        write_frames(frames, cache, fps=25)
        all_frames += frames
        print(f"  clip {i}: {len(frames)} frames -> {cache}", flush=True)

    print(f"total frames: {len(all_frames)}", flush=True)
    beats = pkg.get("beats", [])
    bpm = int(pkg.get("bpm", 112))
    dur = len(all_frames) / 25.0
    music = music_loop(dur, bpm=bpm, seed=seed0)
    voice = mix_voice(beats, dur, bpm=bpm)
    if voice is not None:
        n = min(len(music), len(voice))
        music = music[:n] + voice[:n]
    awav = os.path.join(workdir, "audio.wav")
    _write(awav, music)

    dest = str(dest)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    write_frames(all_frames, dest, fps=25, audio=awav)
    print(f"wrote {dest}", flush=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", default=os.environ.get("HOJI_KEY", DEFAULT_KEY))
    ap.add_argument("--workdir", default=str(ROOT / "build"))
    ap.add_argument("--scene", default=None)
    ap.add_argument("--char-h", type=float, default=0.80)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    key = os.environ.get("PIXAZO_API_KEY")
    if not key:
        raise SystemExit("PIXAZO_API_KEY is not set")
    px = Pixazo(key)
    key_bgr = cv2.imread(a.key)
    if key_bgr is None:
        raise SystemExit(f"cannot read key frame {a.key}")

    pkg = load_package(a.package)
    workdir = os.path.join(a.workdir, a.package)
    os.makedirs(workdir, exist_ok=True)
    out_mp4 = os.path.join(workdir, "short.mp4")

    seed0 = int(pkg.get("seed", 7))

    all_frames = []
    for i, clip in enumerate(pkg["clips"]):
        cache = os.path.join(workdir, f"clip_{i}_final.mp4")
        if os.path.exists(cache) and not a.force:
            all_frames += read_frames(cache, VW * 2, VH * 2)
            print(f"  clip {i}: reused cache", flush=True)
            continue
        scene = a.scene or clip_scene(clip, DEFAULT_SCENE)
        frames, _ = build_clip(px, motion_prompt(clip, i), scene, seed0 + i,
                               key_bgr, workdir, i, char_h_frac=a.char_h)
        write_frames(frames, cache, fps=25)
        all_frames += frames
        print(f"  clip {i}: {len(frames)} frames -> {cache}", flush=True)

    print(f"total frames: {len(all_frames)}", flush=True)

    # audio
    beats = pkg.get("beats", [])
    bpm = int(pkg.get("bpm", 112))
    dur = len(all_frames) / 25.0
    music = music_loop(dur, bpm=bpm, seed=seed0)
    voice = mix_voice(beats, dur, bpm=bpm)
    if voice is not None:
        music = music[:len(voice)] + voice[:len(music)]
    awav = os.path.join(workdir, "audio.wav")
    _write(awav, music)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_frames(all_frames, a.out, fps=25, audio=awav)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
