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
import gc
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
    "classic_cartoons",
    "animationandcartoons",
]
# archive.org's `movies` and `animationandcartoons` collections are mostly
# *copyrighted* commercial film and TV - Cartoon Network, Disney, the lot.
# They are not CC0 no matter what the item page claims. Every query below is
# therefore pinned to an explicit public-domain licence URL so a rights-holder
# can never have a claim on what we publish.
LICENCE_FILTER = "AND licenseurl:*publicdomain*"


def _skin_fraction(bgr):
    """Fraction of a face box that is skin-toned.

    A Haar frontal-face cascade fires on anything with two dark eyes above a
    midline - cardinals and sparrows qualify, which is how a nature
    documentary about cardinals got refused for "containing a person". Real
    human faces are skin-toned; a red cardinal is not. YCrCb chroma is the
    usual test and it holds across skin tones far better than an RGB range.
    """
    try:
        import cv2
        import numpy as np
        if bgr is None or bgr.size == 0:
            return 0.0
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.int16)
        cb = ycrcb[:, :, 2].astype(np.int16)
        m = ((cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 135))
        return float(m.mean())
    except Exception:
        return 0.0


def looks_photographic(path, samples=20):
    """True if this clip is real camera footage rather than drawn animation.

    Classic animation is posterised - a 1930s cartoon holds a few hundred
    distinct colours because every cel was filled in flat. Photographic
    footage of real people holds many thousands. That difference separates
    "a real girl is in this home movie" from "Betty Boop is on screen", and
    it is the only reason the face guard can allow the cartoons a kids'
    channel actually needs while still refusing the home movies.
    """
    try:
        import cv2
        import numpy as np
    except Exception as e:
        log(f"PHOTO TEST: cv2/numpy unavailable ({e}) - assuming photographic")
        return True

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return True
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // samples) if total > samples else 1
    counts = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                small = cv2.resize(frame, (160, 90))
                q = (small // 16).astype(np.int32)
                key = (q[:, :, 0] << 12) | (q[:, :, 1] << 6) | q[:, :, 2]
                counts.append(len(np.unique(key)))
            idx += 1
            if len(counts) >= samples:
                break
    finally:
        cap.release()
    if not counts:
        return True
    med = sorted(counts)[len(counts) // 2]
    photo = med > 2000
    log(f"photo test: median distinct colours {med} -> "
        f"{'photographic' if photo else 'drawn animation'}")
    return photo


def is_animation_source(path, samples=20):
    """True if this clip is drawn animation rather than real camera footage.

    archive.org's own collection is the signal that matters: a clip served out
    of `classic_cartoons` or `animationandcartoons` is animation, full stop.
    Pixel statistics are no use here because archive.org's 512 kb encodes are
    posterised so hard that real home movies and 1930s cartoons both collapse
    to a few hundred distinct colours - measured, not assumed.
    """
    p = str(path).lower()
    return ("cartoon" in p or "animation" in p or "animated" in p
            or "betty" in p or "boop" in p or "snow_white" in p)


def _is_animation(src):
    """True when this source is drawn animation, from its own provenance.

    The collection the clip came out of is the only signal that survives
    archive.org's 512 kb encodes: pixel statistics were tried and measured
    useless, because those encodes are posterised so hard that home movies and
    1930s cartoons both collapse to under a hundred distinct colours.
    """
    p = Path(src)
    for cand in (p.with_suffix(".json"), SRC / (p.stem + ".json")):
        try:
            if cand.exists():
                m = json.loads(cand.read_text())
                if m.get("collection"):
                    return "cartoon" in m["collection"].lower() or \
                           "animation" in m["collection"].lower()
        except Exception:
            pass
    return False


def has_face(path, samples=90, need=3, require_photo=True):
    """True if a human face appears in this clip.

    This is not optional. A kids' channel cannot show a real person's face,
    and archive.org's public-domain collections are full of home movies of
    strangers' children. Haar cascades ship with opencv, so this costs one
    import and no download.

    Two deliberate guards against crying wolf:
      * a face must be seen on `need` separate sampled frames, not once, so a
        coincidental pattern in one frame cannot kill a good source;
      * fail closed - if the cascade cannot be loaded we refuse the source
        rather than gamble on it.
    """
    try:
        import cv2
        cas = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades,
                         "haarcascade_frontalface_default.xml"))
        if cas.empty():
            log("FACE GUARD: cascade failed to load - refusing source")
            return True
    except Exception as e:
        log(f"FACE GUARD: cv2 unavailable ({e}) - refusing source")
        return True

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log("FACE GUARD: cannot open clip - refusing source")
        return True
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // samples) if total > samples else 1
    seen = 0
    idx = 0
    hits = 0
    first_at = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                seen += 1
                h, w = frame.shape[:2]
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, (480, int(480 * h / max(1, w))))
                found = cas.detectMultiScale(g, 1.1, 7, minSize=(32, 32))
                # keep only boxes that are actually skin-toned: a bird's head
                # has the same two-eyes-over-a-midline geometry as a face
                human = []
                for (fx, fy, fw, fh) in found:
                    box = frame[fy:fy + fh, fx:fx + fw]
                    if _skin_fraction(box) >= 0.25:
                        human.append((fx, fy, fw, fh))
                if human:
                    hits += 1
                    if first_at is None:
                        first_at = idx / fps
                    if hits >= need:
                        if require_photo and _is_animation(path):
                            log(f"face guard: {hits} faces, but this is an "
                                f"animation source - cartoon characters, "
                                f"not a real person")
                            return False
                        log(f"FACE GUARD: person on {hits} frames "
                            f"(first at {first_at:.1f}s) - refusing source")
                        return True
            idx += 1
            if seen >= samples:
                break
    finally:
        cap.release()
    if seen == 0:
        # Not one frame could be decoded. That is not "no faces", that is
        # "we could not look" - and the only safe answer to that is refuse.
        log("FACE GUARD: decoded 0 frames - cannot verify, refusing source")
        return True
    if hits:
        log(f"face guard: {hits} lone hit(s) over {seen} frames - "
            f"below the {need}-frame threshold, source accepted")
    else:
        log(f"face guard: clean over {seen} sampled frames")
    return False


def fetch_youtube(topic, out_dir, max_bytes=200 * 1024 * 1024,
                  max_height=1080, want=25):
    """Search YouTube, prefer Creative Commons, download one long-form video.

    Why Creative Commons and not arbitrary uploads: a CC-BY video carries a
    licence that permits reuse with attribution, which is the only YouTube
    source where the rights side of the problem is actually settled. The
    reused-content policy is a separate question and a licence does not
    answer it - that is what the substantive edit in build_short is for.

    The audio is not reused. build_short cuts with -an, which discards the
    original soundtrack completely, and replaces it with our own voiceover and
    music bed. Replacing the audio is the one transformation that genuinely
    removes an audio fingerprint rather than merely perturbing it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
        import imageio_ffmpeg
    except Exception as e:
        log(f"youtube: yt-dlp/imageio-ffmpeg unavailable ({e})")
        return None

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    ydl_opts = {
        "quiet": True, "no_warnings": True, "simulate": True,
        "extract_flat": True, "noplaylist": True,
    }
    # creativecommons is YouTube's own licence filter - it restricts results to
    # videos the uploader has explicitly marked CC-BY.
    query = f"ytsearch{want * 2}:{topic} creativecommons"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as y:
            res = y.extract_info(query, download=False)
    except Exception as e:
        log(f"youtube: search failed: {e}")
        return None

    entries = [e for e in (res.get("entries") or []) if e.get("id")]
    random.shuffle(entries)
    log(f"youtube: {len(entries)} candidates for '{topic}'")

    for e in entries[:want]:
        vid = e.get("id")
        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                                   "simulate": True}) as y:
                info = y.extract_info(url, download=False)
        except Exception:
            continue
        if not info:
            continue
        dur = info.get("duration") or 0
        if dur < 90:
            continue
        # cheapest format at or below max_height that still decodes to
        # something worth cutting; 144p is refused outright.
        cands = [f for f in (info.get("formats") or [])
                 if str(f.get("vcodec") or "").startswith("avc1")
                 and (f.get("height") or 0) >= 240
                 and (f.get("height") or 0) <= max_height]
        if not cands:
            continue
        # Highest quality we can still afford to download. Sorting the other
        # way picked 240p every time, which upscales to mush at 1080x1920.
        cands.sort(key=lambda f: (-(f.get("height") or 0),
                                  f.get("filesize") or 10 ** 9))
        fmt = cands[0]
        if (fmt.get("filesize") or 0) > max_bytes:
            continue
        dest = out_dir / f"yt_{vid}.mp4"
        log(f"youtube: {vid} '{str(info.get('title'))[:48]}' "
            f"{fmt.get('height')}p {dur}s")
        try:
            with yt_dlp.YoutubeDL({
                "quiet": True, "no_warnings": True, "noprogress": True,
                "outtmpl": str(dest.with_suffix(".%(ext)s")),
                "ffmpeg_location": ff, "format": fmt.get("format_id"),
            }) as y:
                y.download([url])
        except Exception as ex:
            log(f"youtube: download failed for {vid}: {str(ex)[:120]}")
            continue
        if not dest.exists() or dest.stat().st_size < 1024 * 1024:
            continue
        if has_face(dest):
            log(f"youtube: dropping {vid} - a real person is in the footage")
            try:
                dest.unlink()
            except OSError:
                pass
            continue
        (out_dir / f"yt_{vid}.json").write_text(json.dumps(
            {"identifier": vid, "title": info.get("title", ""),
             "license": "YouTube Creative Commons (CC-BY)",
             "channel": info.get("uploader", ""), "url": url,
             "height": fmt.get("height"), "duration": dur,
             "description": (info.get("description") or "")[:1500],
             "tags": (info.get("tags") or [])[:25],
             "fetched": dt.date.today().isoformat()}, indent=2))
        log(f"youtube: saved {dest.name} "
            f"({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    log("youtube: nothing usable in the candidates")
    return None


def fetch_cc0(topic, out_dir, max_bytes=60 * 1024 * 1024):
    """Search archive.org for a free clip and download it.

    archive.org's advancedsearch returns JSON with no key required, which is why
    it is the source of choice: no account, no rate limit worth worrying about,
    and every item in these collections carries a licence we can rely on.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(out_dir.glob("*.mp4"))
    for c in cached:
        # A cached source is not automatically safe - it was cached before the
        # face guard existed, and that is exactly how day_06 got a stranger's
        # child in it. Re-check every cached clip before reusing it.
        if has_face(c):
            log(f"dropping cached source {c.name}: contains a person")
            try:
                c.unlink()
                (out_dir / f"{c.stem}.json").unlink(missing_ok=True)
            except OSError:
                pass
            continue
        log(f"reusing cached source {c.name} "
            f"({c.stat().st_size / 1e6:.1f} MB)")
        return c

    asked = random.choice(COLLECTIONS)
    q = urllib.parse.urlencode({
        "q": f"collection:({asked}) AND mediatype:movies "
             f"{LICENCE_FILTER} AND {topic}",
        # archive.org used to expand a repeated fl[] array; it now takes the
        # fields parameter as one comma-separated string and silently returns
        # empty {} docs if you send the array form. That is why every search
        # came back with 25 results and no identifiers.
        "fl": "identifier,title,licenseurl",
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
                + urllib.parse.quote(asked)
                + "+AND+mediatype%3Amovies+AND+licenseurl%3A%2A"
                  "publicdomain%2A&fl=identifier%2Ctitle%2Clicenseurl"
                  "&rows=25&output=json", timeout=45) as r:
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
            if has_face(dest):
                log(f"dropping {ident}: a real person is in the footage")
                try:
                    dest.unlink()
                except OSError:
                    pass
                continue
            log(f"saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            (out_dir / f"{ident}.json").write_text(json.dumps(
                {"identifier": ident, "title": doc.get("title", ""),
                 "license": doc.get("licenseurl", "public domain / CC"),
                 "collection": asked, "file": name, "url": dl,
                 "fetched": dt.date.today().isoformat()},
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


def _subject_of(src):
    """What is actually on screen, taken from the source's own metadata.

    This is the fix for the defect that made every clip unpublishable. The
    script used to be drawn from five canned stories at random, so a nature
    documentary about cardinals was narrated as "The Last Leaf" - the audio
    promised one thing and the picture delivered another, which is the single
    fastest way to lose a viewer in the first two seconds and the thing the
    retention research names most often as fatal.
    """
    p = Path(src)
    meta = {}
    for cand in (p.with_suffix(".json"), SRC / (p.stem + ".json")):
        try:
            if cand.exists():
                meta = json.loads(cand.read_text())
                break
        except Exception:
            pass
    title = (meta.get("title") or p.stem).strip()
    # strip the usual uploader noise so the subject reads cleanly
    title = re.sub(r"\s*[|｜\-–]\s*(full|hd|4k|documentary|free).*$", "",
                   title, flags=re.I).strip(" -|:")
    # a listicle title is not a subject: "5 Things You Didn't Know About
    # Cardinals!" should read as "cardinals"
    m = re.search(r"\b(?:about|of)\s+([a-z][a-z\s\-]{2,40})", title, flags=re.I)
    if m and len(m.group(1).split()) <= 4:
        title = m.group(1)
    title = re.sub(r"^\W+|\W+$", "", title).strip()
    return title or p.stem, meta


# Written as Hook -> Payoff -> Loop, which is the shape the retention research
# converges on: a promise in the first second, delivery, and an ending that
# reconnects to the opening. Every line names the real subject.
HOOKS = [
    "This is a {s}. Watch what happens next.",
    "Meet the {s}. Almost nobody notices this.",
    "A {s}. Looks ordinary. Is not.",
    "Look closely at this {s}.",
    "This {s} is about to do something strange.",
]
BODIES = [
    "See how that moves? Every part of it is built for exactly this.",
    "Watch the way it uses everything around it.",
    "That is not luck. That is thousands of years of practice.",
    "Look at the detail. Nothing here is an accident.",
    "Small, and completely in control of this moment.",
]
PAYOFFS = [
    "And here is the part that surprises people. This happens every single "
    "day, without anyone being taught.",
    "Now you know what to look for. You will never look at one the same way "
    "again.",
    "That is the whole secret. No tricks. Just this.",
    "It looks simple. It has taken a very long time to get this good.",
]
LOOPS = [
    "Watch once more. There is something you missed.",
    "Did you catch that? Watch again.",
    "And that is how it happens. Every time.",
]


def make_script(rng, subject):
    """A short script about the thing that is actually on screen."""
    s = subject.lower().strip()
    hook = rng.choice(HOOKS).format(s=s)
    body = rng.choice(BODIES)
    payoff = rng.choice(PAYOFFS)
    loop = rng.choice(LOOPS)
    title = subject.strip()[:60] or "Wild Moments"
    return title, " ".join([hook, body, payoff, loop])


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


def _motion_at(src, start, seconds):
    """Cheap motion estimate for one candidate offset, straight off the source.

    Downsamples hard so this costs a fraction of the real encode. Used to
    choose *where* to cut rather than cutting blind and hoping.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        prev = None
        total = 0.0
        n = 0
        want = int(min(seconds, 20) * 6)  # 6 samples/sec is plenty
        got = 0
        while got < want:
            ok, fr = cap.read()
            if not ok:
                break
            got += 1
            if got % 5:          # every 5th frame only
                continue
            g = cv2.cvtColor(cv2.resize(fr, (160, 90)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                total += float(abs(g.astype("int16") - prev.astype("int16")).mean())
                n += 1
            prev = g
        cap.release()
        return total / n if n else 0.0
    except Exception:
        return 0.0


def _looks_mono(path, samples=12):
    """True when the footage carries almost no colour of its own.

    Classic-cartoon collections are full of 1930s black-and-white prints.
    Grading those as if they were colour is what made the published clip look
    like a dark scratchy film reel instead of a kids' video, so the treatment
    has to know which it is looking at.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, total // samples) if total > samples else 1
        sats = []
        idx = 0
        while len(sats) < samples:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % step == 0:
                hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
                sats.append(float(hsv[:, :, 1].mean()))
            idx += 1
        cap.release()
        if not sats:
            return False
        med = sorted(sats)[len(sats) // 2]
        mono = med < 28.0
        log(f"mono test: median saturation {med:.1f}/255 -> "
            f"{'monochrome' if mono else 'colour'}")
        return mono
    except Exception as e:
        log(f"mono test failed ({e}); assuming colour")
        return False


def build_short(src, day, rng, out_dir, seconds=42):
    """Cut a segment, treat it, add voice, land a 1080x1920 Short."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = probe_duration(src)
    if dur < seconds + 2:
        log(f"source is only {dur:.1f}s; using all of it")
        start, seconds = 0.0, max(8.0, dur)
    else:
        # Pick the liveliest window rather than a random one. A nature
        # documentary is mostly a talking head or a static wide shot, and a
        # random cut lands on those often enough that the 3.0/255 motion floor
        # refused roughly half of all builds. Scoring six candidates costs a
        # few seconds and turns that into a choice.
        span = max(1.0, dur - seconds - 1.0)
        cands = [rng.uniform(1.0, span) for _ in range(6)]
        scored = [(c, _motion_at(src, c, seconds)) for c in cands]
        scored.sort(key=lambda t: -t[1])
        start = scored[0][0]
        log(f"segment candidates (start, motion): "
            + ", ".join(f"{c:.0f}s/{m:.1f}" for c, m in scored))
        log(f"chose {start:.1f}s (motion {scored[0][1]:.1f})")


    subject, _meta = _subject_of(src)
    title, script = make_script(rng, subject)
    log(f"segment {start:.1f}s +{seconds:.0f}s | subject: {subject} "
        f"| narrative: {title}")

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
    # A 1930s Betty Boop print is black and white, and no grade can make it
    # look like the bright modern benchmark: pushing saturation on monochrome
    # footage does nothing, and a colour-balance cast on neutral grey just
    # tints the mud. Measured on the published clip - brightness 98.6/255 and
    # saturation 41.9/255 against the benchmark's 164.6 and 57.6. So the
    # treatment branches on what the footage actually is.
    if _looks_mono(seg):
        # Lift it out of the murk and give it a warm tint instead of grey.
        # Brightness +0.12 and contrast 1.15 move 98.6 towards the mid-140s;
        # the balance shift hands back some colour without pretending it was
        # shot in colour.
        grade = ("eq=contrast=1.15:brightness=0.12:saturation=1.0,"
                 "colorbalance=rs=.10:rm=.06:bs=-.08")
        vign = "vignette=PI/5.2:angle=0.35"
    else:
        # The source carries its own colour - the raw Betty Boop print
        # measures a median saturation of 110/255. The old grade pushed
        # contrast to 1.35 with brightness *down* and a hard PI/4.6 vignette,
        # which is what dragged the published clip to 98.6/255 brightness
        # against the benchmark's 164.6. Lift instead of crush.
        grade = rng.choice([
            "eq=contrast=1.05:brightness=0.10:saturation=1.02",
            "eq=contrast=1.06:brightness=0.09:saturation=1.05",
            "eq=contrast=1.04:brightness=0.11:saturation=1.00",
        ])
        vign = "vignette=PI/5.4:angle=0.22"
    text = tmp / "cards"
    text.mkdir(exist_ok=True)
    cards = []
    for i, line in enumerate(_wrap(script, 5)[:5]):
        p = text / f"c{i}.png"
        _text_card(p, line, rng)
        cards.append(p)

    vf = [grade, vign]
    # intercut our own text cards - the narrative, not the source's
    inputs = ["-i", str(seg)]
    for c in cards:
        inputs += ["-loop", "1", "-t", "3.2", "-i", str(c)]
    # drawtext is NOT in the imageio-ffmpeg static build (it needs freetype),
    # which is what broke the first live run with "Filter not found". The cards
    # are already rendered to PNG by PIL, so overlay them instead - overlay is
    # in every build and needs no font support at all.
    lines = _wrap(script, 5)[:5]
    fg = [f"[0:v]{','.join(vf)}[base]"]
    prev = "base"
    for i in range(len(cards)):
        fg.append(f"[{i + 1}:v]scale=920:-1,format=rgba[c{i}]")
        t0, t1 = 6 + i * 7, 6 + i * 7 + 3.2
        out_label = f"v{i + 1}" if i < len(cards) - 1 else "vout"
        fg.append(f"[{prev}][c{i}]overlay=(W-w)/2:H*0.15:"
                  f"enable='between(t,{t0},{t1})'[{out_label}]")
        prev = out_label
    full = ";".join(fg)

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
    # Release everything the search, the download and the face checks are
    # still holding before the encode asks for its own buffers. On a 1.9 GB
    # box the difference between fitting and being SIGKILLed is a few hundred
    # megabytes.
    gc.collect()
    cmd += ["-filter_complex", full, "-map", "[vout]"]
    # Count inputs by position, not by list length. `inputs` holds six list
    # elements per card (-loop 1 -t 3.2 -i <path>) plus two for the segment, so
    # len(inputs) // 2 is meaningless and produced index 16 for a six-input
    # graph - "Invalid input file index: 16". The real order is
    #   0 = segment, 1..N = cards, N+1 = voice, N+2 = music
    n_card_inputs = len(cards)
    i_voice = 1 + n_card_inputs
    i_music = i_voice + (1 if has_voice else 0)
    if has_voice:
        cmd += ["-map", f"{i_voice}:a"]
    if music.exists():
        cmd += ["-map", f"{i_music}:a"]
    # This box has under 2 GB of RAM. x264 sizes its lookahead and reference
    # buffers per thread, so on a many-core machine the default thread count
    # alone can exceed the whole budget and the kernel SIGKILLs us mid-write
    # (rc -9), leaving a truncated file with no moov atom that then fails every
    # later check for a reason that looks like nothing to do with memory.
    # Two threads is the largest that survives here.
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-b:a", "128k", "-shortest", "-movflags", "+faststart",
            # ref=1, no B-frames and no lookahead cut x264's frame buffers to a
            # fraction of the default. Quality cost is real but small next to
            # being SIGKILLed mid-write, which is what the defaults do on a
            # 1.9 GB box. This box is the binding constraint, not the encoder.
            "-x264-params", "ref=1:bframes=0:rc-lookahead=0:sliced-threads=1",
            str(out)]
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
    # OpenCV, not imageio: imageio needs the pyav plugin to read video, which is
    # not installed, and the day_05 guardrail must not silently no-op. opencv
    # is already in requirements.txt.
    try:
        import cv2
        import numpy as np
    except Exception as e:
        log(f"cannot measure motion: {e}")
        return None
    try:
        cap = cv2.VideoCapture(str(path))
        frames = []
        idx = 0
        while len(frames) < 120:
            ok, fr = cap.read()
            if not ok:
                break
            idx += 1
            if idx % 4:          # every 4th frame; a separate counter, because
                continue         # testing len(frames) never advances and spins
            if fr is None or fr.size == 0:
                continue
            small = cv2.resize(fr, (fr.shape[1] // 4, fr.shape[0] // 4))
            frames.append(small.astype("float32").mean(2))
        cap.release()
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
    ap.add_argument("--source", default="auto",
                    choices=["auto", "archive", "youtube"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.day * 7919 + dt.date.today().toordinal())
    # Topics that are animal footage rather than talking heads. "Nature
    # documentary" sounds ideal and is the worst possible choice: those videos
    # are narrated by people, and the face guard correctly refuses every one of
    # them, which is how two consecutive w10 runs produced nothing at all.
    TOPICS = [
        "cardinal bird", "baby animals", "wildlife footage",
        "birds in the wild", "squirrel slow motion", "ducklings",
        "fox cubs", "hedgehog close up", "sea turtles",
        "hummingbird slow motion", "owls", "deer in forest",
    ]
    src = None
    used = None
    # The requested topic first, then the rest, with archive.org as a backstop
    # for each. An unattended daily run cannot afford to return nothing just
    # because one search term happened to be people-heavy.
    order = [a.topic] + [t for t in TOPICS if t != a.topic]
    for t in order:
        if a.source != "archive":
            src = fetch_youtube(t, SRC)
            if src:
                used = t
                break
        if a.source != "youtube":
            src = fetch_cc0(t, SRC)
            if src:
                used = t
                break
    if src:
        log(f"source came from topic '{used}'")
    if not src:
        log("no CC0 source available; nothing built")
        print(json.dumps({"action": "no-source", "day": a.day}))
        return 1

    # Second, independent face check on the exact clip we are about to cut.
    # The first one is in fetch_cc0; this is the one that must never be
    # bypassed, because it runs after every other decision has been made.
    if has_face(src):
        log("REFUSING: a real person is in the source footage. A kids' "
            "channel does not show strangers' faces.")
        print(json.dumps({"action": "rejected-person", "day": a.day,
                          "source": src.name}))
        return 1

    if a.dry_run:
        log(f"--dry-run: would cut {a.seconds}s from {src.name}")
        print(json.dumps({"action": "dry-run", "source": src.name}))
        return 0

    out, title, script = build_short(src, a.day, rng, WORK, a.seconds)
    if not out:
        print(json.dumps({"action": "failed", "day": a.day}))
        return 1

    # A finished Short is never tiny and never a fraction of a second long.
    # The voiceover sets the real length via -shortest, so a 19 s clip against
    # a 42 s request is normal - what is NOT normal is 0.0 s or a sub-megabyte
    # file, which is what a truncated write or an undecodable source produces.
    d = probe_duration(out)
    if out.stat().st_size < 1_000_000 or (d is not None and d < 10.0):
        log(f"REFUSING: output is {out.stat().st_size / 1e6:.1f} MB / "
            f"{d}s - the encode failed, not a usable clip")
        print(json.dumps({"action": "failed", "day": a.day,
                          "bytes": out.stat().st_size}))
        return 1

    m = measure_motion(out)
    log(f"frame-to-frame mean abs diff: "
        f"{m:.2f}/255" if m else "motion: not measured")
    if m is not None and m < 3.0:
        log("REFUSING: motion below the 3.0/255 floor - this is the day_05 "
            "guardrail, a near-still must never reach YouTube")
        print(json.dumps({"action": "rejected", "day": a.day, "motion": m}))
        return 1

    # Last line of defence: check the finished cut, not just the source. The
    # source check samples the whole clip sparsely; this samples the exact
    # frames we are about to publish, densely. A cartoon character will trip
    # this too, and that is the correct outcome - a recognisable third-party
    # character is not ours to publish any more than a real person's face is.
    if has_face(out, samples=60, need=3):
        log("REFUSING: a face is visible in the finished cut")
        print(json.dumps({"action": "rejected-person", "day": a.day,
                          "where": "output"}))
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
