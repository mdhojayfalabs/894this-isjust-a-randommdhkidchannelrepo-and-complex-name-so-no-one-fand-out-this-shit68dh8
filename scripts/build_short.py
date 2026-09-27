#!/usr/bin/env python3
"""
MDH Kids Studio - Short builder.

Turns a content package into a finished 9:16 Short with original audio.

    python scripts/build_short.py --package day_04 --out videos/day_04.mp4

What it does, in order:
  1. renders each clip through Pixazo image-to-video (Hoji key frame + anchor)
  2. crops 1280x704 -> 396x704 (exact 9:16) and upscales to 720x1280
  3. concatenates the clips
  4. lays a synthesised music bed plus beat-timed SFX
  5. makes the last frame match the first so the Short loops cleanly

WHY THESE NUMBERS
-----------------
Shorts retention thresholds are ~65% under 30s and ~50% for 30-60s, and the
22-45 second band beats both extremes. The free LTX model emits ~5s clips, so
a Short is 4-9 clips stitched. Loop rate above 100% is a strong secondary
signal, hence step 5.

The model ignores aspect_ratio, so vertical has to be manufactured in post.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from gen_audio import mix_voice, music_loop, _write  # noqa: E402

# ---------------------------------------------------------------- config
GATEWAY = "https://gateway.pixazo.ai"
SUBMIT = "/ltx-video/v1/image-to-video"
STATUS = "/v2/requests/status/{rid}"

# The Hoji character anchor. MUST accompany every image-to-video request:
# text-to-video does not hold the character (verified - it produced a blonde,
# pale-skinned boy with no glasses from a near-identical prompt).
HOJI_ANCHOR = (
    "Hoji is a toddler boy with DARK BROWN spiky hair, THICK BLACK ROUND "
    "GLASSES, a small BROWN GOATEE, BROWN skin, wearing an ORANGE-RED SOCCER "
    "JERSEY with a RAINBOW-STAR CREST on the chest, NAVY BLUE SHORTS and "
    "COLOURFUL BLUE RED YELLOW SNEAKERS. Keep his face, hair colour, glasses "
    "and outfit EXACTLY as in the reference image. "
)
STYLE = ("3D Pixar animation, ultra bright studio lighting, saturated colours, "
         "smooth 24fps motion, no text on screen, no spoken dialogue. ")


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        for cand in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if os.path.exists(cand):
                return cand
        raise SystemExit("ffmpeg not found: pip install imageio-ffmpeg")


FF = _ffmpeg()


def sh(args, **kw):
    r = subprocess.run(args, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, args[:4]))}... failed:\n{r.stderr[-800:]}")
    return r


# ---------------------------------------------------------------- pixazo
class Pixazo:
    def __init__(self, key):
        self.key = key

    def _post(self, path, payload):
        d = json.dumps(payload).encode()
        req = urllib.request.Request(GATEWAY + path, data=d, method="POST", headers={
            "Content-Type": "application/json", "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": self.key, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())

    def submit(self, motion, image_url):
        j = self._post(SUBMIT, {"prompt": HOJI_ANCHOR + STYLE + motion,
                                "image": image_url})
        rid = j.get("request_id")
        if not rid:
            raise RuntimeError(f"submit failed: {j}")
        return rid

    def wait(self, rid, timeout=900):
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
        # The CDN 403s without a browser UA - urlretrieve fails here.
        sh(["curl", "-sSL", "-A", "Mozilla/5.0", "-o", str(dest), url])


# ---------------------------------------------------------------- video
def verticalise(src, dest):
    """1280x704 landscape -> 396x704 exact 9:16 -> 720x1280.
    442 centres the crop horizontally on the character."""
    sh([FF, "-y", "-i", str(src), "-vf",
        "crop=396:704:442:0,scale=720:1280:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-an", str(dest)])


def concat(clips, dest):
    lst = dest.with_suffix(".txt")
    lst.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    sh([FF, "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(dest)])


def add_audio(video, wav, dest):
    sh([FF, "-y", "-i", str(video), "-i", str(wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
        "-shortest", str(dest)])


def make_loop(src, dest, xfade=0.6):
    """Crossfade the tail back into the head so the auto-restart is hidden.

    Shorts loop automatically, and loop rate above 100% is a strong secondary
    signal. A hard cut from the last frame back to the first is visible; a short
    crossfade is not.

    Structure of the result, total length unchanged:
        [ video 0 .. D-xfade ]  +  [ blend(head, tail) over xfade seconds ]
    where head is the opening xfade seconds and tail is the closing xfade
    seconds. So the viewer sees the ending dissolve into the beginning.
    """
    d = probe_duration(src)
    if d <= xfade * 2.5:
        sh([FF, "-y", "-i", str(src), "-c", "copy", str(dest)])
        return
    fc = (
        f"[0:v]trim=0:{d - xfade:.3f},setpts=PTS-STARTPTS[m];"
        f"[0:v]trim=0:{xfade:.3f},setpts=PTS-STARTPTS[h];"
        f"[0:v]trim=start={d - xfade:.3f},setpts=PTS-STARTPTS[t];"
        f"[h][t]blend=all_expr='A*(1-(T/{xfade:.3f}))+B*(T/{xfade:.3f})'[x];"
        f"[m][x]concat=n=2:v=1:a=0[v]"
    )
    sh([FF, "-y", "-i", str(src), "-filter_complex", fc,
        "-map", "[v]", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", str(dest)])


def probe_duration(path):
    r = sh([FF, "-i", str(path), "-f", "null", "-"])
    for line in r.stderr.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


# ---------------------------------------------------------------- main
def build(package, out, image_url, key, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    px = Pixazo(key)

    clips = []
    for i, motion in enumerate(package["clips"]):
        raw = workdir / f"clip_{i}_raw.mp4"
        vert = workdir / f"clip_{i}_v.mp4"
        if not vert.exists():
            print(f"  clip {i}: submitting...")
            rid = px.submit(motion, image_url)
            print(f"  clip {i}: queued {rid[-12:]}")
            url = px.wait(rid)
            print(f"  clip {i}: generated, downloading")
            px.fetch(url, raw)
            print(f"  clip {i}: verticalising")
            verticalise(raw, vert)
        clips.append(vert)
        print(f"  clip {i}: done ({vert.stat().st_size/1e6:.2f} MB)")

    joined = workdir / "joined.mp4"
    concat(clips, joined)

    # Loop seam: crossfade the tail into the head so the auto-restart is hidden.
    looped = workdir / "looped.mp4"
    make_loop(joined, looped, xfade=package.get("loop_xfade", 0.6))
    print(f"  looped: {probe_duration(looped):.2f}s")

    # Duration comes from the rendered clips themselves, not from the package.
    # The free LTX model emits ~5s per clip, but it is not exact, so probe it.
    total = 0.0
    for c in clips:
        r = sh([FF, "-i", str(c), "-f", "null", "-"])
        for line in r.stderr.splitlines():
            if "Duration:" in line:
                try:
                    hms = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = hms.split(":")
                    total += int(h) * 3600 + int(m) * 60 + float(s)
                except Exception:
                    total += 5.0
                break
        else:
            total += 5.0
    if total <= 0:
        total = 5.0 * len(clips)
    print(f"  measured duration: {total:.2f}s from {len(clips)} clips")
    bed = music_loop(seconds=total, bpm=package.get("bpm", 112),
                     seed=package.get("seed", 7))
    events = [(b["t"], b["cat"], b.get("name", "a"), b.get("gain", 0.7))
              for b in package.get("beats", [])]
    wav = workdir / "audio.wav"
    _write(wav, mix_voice(bed, events, total))
    print(f"  audio: {total:.1f}s bed + {len(events)} sfx")

    add_audio(looped, wav, out)
    print(f"  wrote {out} ({out.stat().st_size/1e6:.2f} MB)")


def main():
    global FF
    FF = _ffmpeg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="JSON file describing the Short")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-url", required=True,
                    help="publicly reachable 9:16 character key frame")
    ap.add_argument("--key", default=os.environ.get("PIXAZO_API_KEY"))
    ap.add_argument("--workdir", default="build")
    a = ap.parse_args()
    if not a.key:
        raise SystemExit("PIXAZO_API_KEY not set")

    pkg = json.loads(Path(a.package).read_text())
    build(pkg, Path(a.out), a.image_url, a.key, Path(a.workdir))


if __name__ == "__main__":
    main()
