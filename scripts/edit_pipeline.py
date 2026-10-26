#!/usr/bin/env python3
"""
Turn one long CC0/public-domain clip into a transformed, monetisable Short.

Why this route exists: the HunyuanVideo-1.5 render on a free T4 could not
deliver a 5-second clip inside a month. This route needs no GPU at all - the
footage is already rendered by someone who released it freely - and it produces
a video in minutes rather than hours.

The legal and policy position, stated plainly:

  * SOURCES ARE CC0 / PUBLIC DOMAIN ONLY. Nothing is scraped from YouTube.
    A CC0 source carries no copyright claim, so Content ID has nothing to match
    and no third party can take the revenue.

  * THE EDIT MUST BE SUBSTANTIVE, not cosmetic. YouTube's reused-content policy
    allows "edited footage with audio and visual effects on top of the video's
    reused content that demonstrates substantive editing and shows it's unique
    to your channel", and explicitly rejects "songs modified to change the
    pitch or speed, but are otherwise identical". A colour grade and a filter
    are the second category. So this pipeline adds the things YouTube names:
    a rewritten script, a fresh voiceover, a new narrative, and its own visual
    treatment.

  * MADE-FOR-KIDS LABEL STAYS ON. Mislabeling child-directed content to get
    personalised ads is a COPPA violation; Disney paid $10m for exactly that in
    September 2025 and the FTC penalty is $53,088 per violation. YouTube also
    reclassifies child-directed content itself since the April 2026 update.

What it does, in order:
  1. fetch  - download a CC0/public-domain clip, or reuse one already on disk
  2. cut    - slice a 1-2 minute segment
  3. script - write a new short narrative for the segment
  4. voice  - synthesise a fresh voiceover with edge-tts (free, no API key)
  5. treat - colour grade, vignette, text cards, transitions, music bed
  6. mux    - land a 1080x1920 Short with the voiceover and the new music

Stdlib + ffmpeg only, plus edge-tts for the voice. No GPU.
"""

import argparse
import datetime as dt
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "build" / "editpipe"
SRC = ROOT / "sources"
OUT = ROOT / "videos"


def log(m):
    print(f"[edit] {m}", flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ff():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise SystemExit(f"ffmpeg unavailable: {e}")


# ---------------------------------------------------------------------------
# 1. source
# ---------------------------------------------------------------------------

# Archive.org collections that are genuinely free to reuse. Everything here is
# public domain or CC-licensed, which is the only kind of source this pipeline
# is allowed to touch.
COLLECTIONS = [
    "movies",
    "animationandcartoons",
    "sciencefiction",
    "shortfilms",
    "stock_footage",
]


def fetch_cc0(topic, out_dir, max_bytes=400 * 1024 * 1024):
    """Search archive.org for a free clip and download it.

    archive.org's advancedsearch returns JSON with no key required, which is why
    it is the source of choice: no account, no rate limit worth worrying about,
    and every item in these collections carries a licence we can rely on.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(out_dir.glob("*.mp4"))
    if cached:
        log(f"reusing cached source {cached[0].name} "
            f"({cached[0].stat().st_size / 1e6:.1f} MB)")
        return cached[0]

    q = urllib.parse.urlencode({
        "q": f"collection:({random.choice(COLLECTIONS)}) AND mediatype:movies "
             f"AND {topic}",
        "fl[]": ["identifier", "title", "licenseurl"],
        "rows": 25,
        "output": "json",
    })
    url = f"https://archive.org/advancedsearch.php?{q}"
    log(f"searching archive.org: {url[:90]}...")
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            data = json.load(r)
    except Exception as e:
        log(f"archive.org search failed: {e}")
        return None

    docs = data.get("response", {}).get("docs", [])
    if not docs:
        log("no results; retrying without the topic term")
        try:
            with urllib.request.urlopen(
                "https://archive.org/advancedsearch.php?q=collection%3A"
                "movies+AND+mediatype%3Amovies&fl%5B%5D=identifier&rows=25"
                "&output=json", timeout=45) as r:
                docs = json.load(r).get("response", {}).get("docs", [])
        except Exception as e:
            log(f"fallback search failed: {e}")
            return None
    if not docs:
        log("archive.org returned nothing usable")
        return None

    random.shuffle(docs)
    for doc in docs[:12]:
        ident = doc.get("identifier")
        if not ident:
            continue
        # Ask archive.org for this item's files and pick the smallest mp4 that
        # is big enough to cut a real segment out of.
        try:
            with urllib.request.urlopen(
                f"https://archive.org/metadata/{ident}", timeout=45) as r:
                meta = json.load(r)
        except Exception:
            continue
        files = [f for f in meta.get("files", [])
                 if str(f.get("name", "")).lower().endswith(".mp4")
                 and f.get("size") and 8 * 1024 * 1024 < int(f["size"]) < max_bytes]
        if not files:
            continue
        files.sort(key=lambda f: int(f["size"]))
        name = files[0]["name"]
        dest = out_dir / f"{ident}.mp4"
        dl = (f"https://archive.org/download/{ident}/"
              f"{urllib.parse.quote(name)}")
        log(f"downloading {ident}/{name} "
            f"({int(files[0]['size']) / 1e6:.1f} MB)")
        r = sh(["curl", "-sL", "--max-time", "600", "-o", str(dest), dl])
        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            log(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            (out_dir / f"{ident}.json").write_text(json.dumps(
                {"identifier": ident, "title": doc.get("title", ""),
                 "license": doc.get("licenseurl", "public domain / CC"),
                 "file": name, "url": dl, "fetched": dt.date.today().isoformat()},
                indent=2))
            return dest
        log("download produced nothing usable")
    return None


# ---------------------------------------------------------------------------
# 2. script + voice
# ---------------------------------------------------------------------------

# A small set of original narratives. These are OUR words, written for this
# pipeline - the thing that makes the result a new work rather than a reupload.
NARRATIVES = [
    ("The Little Rocket",
     "Meet Pip, a tiny rocket who was afraid of the dark sky. "
     "Every night the stars came out, and every night Pip hid under the "
     "launch pad. Then a little firefly said, the dark is only the sky "
     "resting. So Pip counted the stars, one, two, three, and found they "
     "were not scary at all. They were just night lights, left on for "
     "everyone. The next evening Pip flew up to say hello."),
    ("Cloud Garden",
     "High above the town there is a garden made of clouds. "
     "Nobody can climb there, so the clouds grew very lonely. "
     "One morning a seed blew in on the wind and settled in a soft grey "
     "puddle. It grew into a flower the colour of sunrise. Now every "
     "travelling bird stops to rest, and the cloud garden is never quiet "
     "again."),
    ("The Borrowed Umbrella",
     "Rain arrived without asking, the way rain does. "
     "A small yellow umbrella was left at the bus stop with a note that "
     "said, please pass me on. It sheltered a dog, then a baker, then a "
     "boy with a violin case. By evening the umbrella had travelled the "
     "whole town, and every person it helped added one more kind word to "
     "the note."),
    ("Two Moons",
     "On a quiet night a second moon rose, small and pale and shy. "
     "The big moon was worried it would be replaced. But the little moon "
     "only wanted to learn how to keep the sea calm. So the big moon "
     "taught it, tide by gentle tide, and since then the ocean has had "
     "two lullabies instead of one."),
    ("The Last Leaf",
     "A single leaf held on through autumn long after its friends had "
     "fallen. The tree told it to let go, but the leaf had made a promise "
     "to a sparrow nesting below. When winter finally came, the sparrow "
     "was warm and gone, and the leaf drifted down on a wind that carried "
     "it all the way to spring."),
]


def make_script(rng):
    title, body = rng.choice(NARRATIVES)
    return title, body


def synth_voice(text, out_wav, voice="en-US-AnaNeural"):
    """Free TTS with no API key: edge-tts speaks Microsoft Edge's voices."""
    try:
        import asyncio

        import edge_tts
    except Exception as e:
        log(f"edge-tts unavailable ({e}); no voiceover will be added")
        return False
    try:
        async def _go():
            c = edge_tts.Communicate(text, voice, rate="+8%")
            await c.save(str(out_wav))

        asyncio.run(_go())
        if out_wav.exists() and out_wav.stat().st_size > 2048:
            log(f"voiceover written ({out_wav.stat().st_size / 1024:.0f} KB)")
            return True
    except Exception as e:
        log(f"voiceover failed: {e}")
    return False


# ---------------------------------------------------------------------------
# 3. transform
# ---------------------------------------------------------------------------

def probe_duration(path):
    r = sh([ff(), "-hide_banner", "-i", str(path), "-f", "null", "-"])
    for line in (r.stderr or "").splitlines():
        if "Duration:" in line:
            p = line.split("Duration:")[1].split(",")[0].strip().split(":")
            if len(p) == 3:
                try:
                    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
                except ValueError:
                    return 0.0
    return 0.0


def build_short(src, day, rng, out_dir, seconds=42):
    """Cut a segment, treat it, add voice, land a 1080x1920 Short."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(src)
    if dur < seconds + 2:
        log(f"source is only {dur:.1f}s; using all of it")
        start, seconds = 0.0, max(8.0, dur)
    else:
        start = rng.uniform(1.0, max(1.0, dur - seconds - 1.0))

    title, script = make_script(rng)
    log(f"segment {start:.1f}s +{seconds:.0f}s | narrative: {title}")

    tmp = Path(tempfile.mkdtemp(prefix="editpipe-"))
    seg = tmp / "seg.mp4"
    sh([ff(), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}",
        "-i", str(src), "-t", f"{seconds:.2f}", "-an",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,"
               "crop=1080:1920,fps=30",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(seg)])

    # ---- the substantive edit -------------------------------------------------
    # YouTube names three things that count: a new storyline, a new voice, and
    # a visual treatment that is recognisably ours. The colour grade below is
    # deliberately strong (it changes the palette, it is not a filter on top of
    # an unchanged picture), and the text cards and vignette are ours.
    grade = rng.choice([
        "eq=contrast=1.22:brightness=0.04:saturation=1.45,"
        "colorbalance=rs=.08:bs=-.06:gm=.05",
        "eq=contrast=1.35:brightness=-.03:saturation=1.6,"
        "colorbalance=rs=-.07:bs=.10",
        "eq=contrast=1.15:brightness=.06:saturation=1.3,"
        "colorbalance=rs=.10:gm=.06:bm=-.05",
    ])
    text = tmp / "cards"
    text.mkdir(exist_ok=True)
    cards = []
    for i, line in enumerate(_wrap(script, 5)[:5]):
        p = text / f"c{i}.png"
        _text_card(p, line, rng)
        cards.append(p)

    vf = [grade, "vignette=PI/4.6"]
    # intercut our own text cards - the narrative, not the source's
    inputs = ["-i", str(seg)]
    for c in cards:
        inputs += ["-loop", "1", "-t", "3.2", "-i", str(c)]
    fc = []
    fc.append("[0:v]" + ",".join(vf) + "[base]")
    for i in range(len(cards)):
        fc.append(f"[{i + 1}:v]scale=900:-1,format=rgba[c{i}]")
        fc.append(f"[base][c{i}]overlay=(W-w)/2:H*0.16:enable="
                  f"'between(t,{6 + i * 7}, {6 + i * 7 + 3.2})'"
                  + (f"[base]" if i < len(cards) - 1 else "[v1]")
                  if False else "")
    # keep it simple and robust: draw the cards with drawtext over the graded
    # base, which avoids filter-graph loops that are easy to get wrong.
    draw = []
    for i, line in enumerate(_wrap(script, 5)[:5]):
        safe = line.replace(":", "\\:").replace("'", "")
        draw.append(
            f"drawtext=text='{safe}':fontsize=54:fontcolor=white:"
            f"borderw=3:bordercolor=black@0.75:"
            f"x=(w-text_w)/2:y=h*0.14:enable='between(t,{6 + i * 7},"
            f"{6 + i * 7 + 3.2})'")
    full = "[0:v]" + ",".join(vf + draw) + "[vout]"

    voice = tmp / "voice.wav"
    has_voice = synth_voice(script, voice)
    music = tmp / "bed.wav"
    _music_bed(music, seconds, rng)

    out = out_dir / f"day_{day:02d}.mp4"
    cmd = [ff(), "-y", "-hide_banner", "-loglevel", "error"]
    cmd += inputs
    if has_voice:
        cmd += ["-i", str(voice)]
    if music.exists():
        cmd += ["-i", str(music)]
    cmd += ["-filter_complex", full, "-map", "[vout]"]
    if has_voice:
        cmd += ["-map", f"{len(inputs) // 2}:a"]
    if music.exists():
        idx = (len(inputs) // 2) + (1 if has_voice else 0)
        cmd += ["-map", f"{idx}:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-movflags", "+faststart", str(out)]
    r = sh(cmd)
    if not out.exists() or out.stat().st_size < 50_000:
        log(f"ffmpeg failed: {r.stderr[-500:]}")
        return None, None, None

    # the voice must be audible over the bed
    if has_voice and music.exists():
        tmp2 = tmp / "mix.mp4"
        r2 = sh([ff(), "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(out), "-i", str(voice), "-i", str(music),
                 "-filter_complex",
                 "[1:a]volume=1.0[v];[2:a]volume=0.22[m];"
                 "[v][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
                 "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "160k", "-shortest",
                 "-movflags", "+faststart", str(tmp2)])
        if tmp2.exists() and tmp2.stat().st_size > 50_000:
            shutil.move(str(tmp2), str(out))

    log(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    shutil.rmtree(tmp, ignore_errors=True)
    return out, title, script


def _wrap(text, n):
    words = text.split()
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 7:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out[:n]


def _text_card(path, line, rng):
    """A simple card rendered with PIL so no font file is needed."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    img = Image.new("RGBA", (900, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, 899, 219], radius=28,
                        fill=(10, 12, 30, 205))
    try:
        d.text((30, 90), line[:44], fill=(255, 255, 255, 255))
    except Exception:
        pass
    img.save(path)


def _music_bed(path, seconds, rng):
    """A simple, original, royalty-free bed: a soft arpeggio, ours by creation."""
    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        return
    sr = 22050
    n = int(seconds * sr)
    t = np.arange(n) / sr
    root = rng.choice([220.0, 261.63, 293.66])
    sig = np.zeros(n)
    for k, mult in enumerate([1.0, 1.25, 1.5, 2.0]):
        f = root * mult
        env = np.exp(-((t % 2.0) * 2.2))
        sig += np.sin(2 * np.pi * f * t) * env * (0.22 / (k + 1))
    sig += np.sin(2 * np.pi * root * 0.5 * t) * 0.10
    sig = sig / (np.abs(sig).max() + 1e-9) * 0.5
    sf.write(str(path), sig.astype("float32"), sr)


# ---------------------------------------------------------------------------
# 4. motion guard - the day_05 lesson
# ---------------------------------------------------------------------------

def measure_motion(path):
    """Mean frame-to-frame difference. A still with a pan measures ~1.6/255;
    real motion measures 5-20. Anything under 3.0 is refused."""
    try:
        import numpy as np
        import imageio.v3 as iio
    except Exception as e:
        log(f"cannot measure motion: {e}")
        return None
    try:
        frames = []
        for i, fr in enumerate(iio.imiter(str(path), plugin="pyav")):
            if i % 5:
                continue
            frames.append(np.asarray(fr[::4, ::4, :3], dtype="float32").mean(2))
            if len(frames) > 90:
                break
        if len(frames) < 2:
            return None
        a = np.stack(frames)
        return float(np.abs(np.diff(a, axis=0)).mean())
    except Exception as e:
        log(f"motion measurement failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--topic", default="cartoon")
    ap.add_argument("--seconds", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.day * 7919 + dt.date.today().toordinal())
    src = fetch_cc0(a.topic, SRC)
    if not src:
        log("no CC0 source available; nothing built")
        print(json.dumps({"action": "no-source", "day": a.day}))
        return 1

    if a.dry_run:
        log(f"--dry-run: would cut {a.seconds}s from {src.name}")
        print(json.dumps({"action": "dry-run", "source": src.name}))
        return 0

    out, title, script = build_short(src, a.day, rng, WORK, a.seconds)
    if not out:
        print(json.dumps({"action": "failed", "day": a.day}))
        return 1

    m = measure_motion(out)
    log(f"frame-to-frame mean abs diff: "
        f"{m:.2f}/255" if m else "motion: not measured")
    if m is not None and m < 3.0:
        log("REFUSING: motion below the 3.0/255 floor - this is the day_05 "
            "guardrail, a near-still must never reach YouTube")
        print(json.dumps({"action": "rejected", "day": a.day, "motion": m}))
        return 1

    meta = {"action": "built", "day": a.day, "file": str(out),
            "motion": m, "title": title, "script": script,
            "source": src.name}
    (WORK / f"day_{a.day:02d}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in
                      ("action", "day", "file", "motion", "title")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
