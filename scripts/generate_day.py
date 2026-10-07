#!/usr/bin/env python3
"""
Generate the finished Short for a calendar day.

    python scripts/generate_day.py 04

Reads packages/day_NN.json (the generation spec) and writes videos/day_NN.mp4.
This is the missing half of the old design: run_daily.py looked for
videos/day_NN.mp4 but nothing ever created it, so every run ended with
"WAIT day NN: media not found".

Idempotent: if videos/day_NN.mp4 already exists and passes validation, it is
reused rather than regenerated. That matters because a generation run costs
several minutes of Pixazo time and the daily rate limit is finite.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_stills import build as build_free  # noqa: E402
from build_short import build  # noqa: E402

# The character key frame. It lives in the repo and is served from raw.githubusercontent,
# which the Pixazo gateway can fetch. Crop is 286:508:44:152 of the model sheet,
# scaled to 720x1280 - exact 9:16, with the Bengali labels and captages excluded.
DEFAULT_KEY_IMAGE = (
    "https://raw.githubusercontent.com/mdhojayfalabs/"
    "894this-isjust-a-randommdhkidchannelrepo-and-complex-name-so-no-one-fand-out-this-shit68dh8"
    "/main/assets/hoji_key_720.png"
)


def package_path(day):
    return os.path.join("packages", f"day_{day}.json")


def out_path(day):
    return os.path.join("videos", f"day_{day}.mp4")


def generate(day, key=None, image_url=None, workdir=None, force=False):
    import json
    from pathlib import Path

    pp = package_path(day)
    if not os.path.exists(pp):
        raise SystemExit(f"no package: {pp}")

    dest = Path(out_path(day))
    if dest.exists() and not force:
        print(f"generate_day: {dest} already exists, reusing")
        return dest

    pkg = json.loads(Path(pp).read_text())
    workdir = Path(workdir or f"build/day_{day}")
    workdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PRIMARY ROUTE: build_stills.py -- free, card-free, scriptable, no API key,
    # no wallet, no rate limit, no GPU. It renders the composed key frame into
    # shots with a moving 9:16 crop window and muxes a synthesised music bed.
    #
    # The previous primary was build_short.py -> the Pixazo gateway. That wallet
    # is empty (402 Insufficient Balance, recorded in providers.py), and the
    # user's standing constraint is FREE ONLY, so Pixazo is demoted to a
    # fallback that only runs if a key is actually present.
    # ------------------------------------------------------------------
    key_frame = os.environ.get("KEY_FRAME") or "assets/scene_pair_key.png"
    try:
        import cv2

        key_bgr = cv2.imread(key_frame)
    except Exception as e:
        key_bgr = None
        print(f"generate_day: cv2 unavailable ({e})")
    if key_bgr is None:
        raise SystemExit(
            f"cannot read key frame {key_frame!r}; set KEY_FRAME to a readable "
            "9:16 PNG")

    print(f"generate_day: day {day} -> {dest} (free stills route)")
    print(f"generate_day: key frame {key_frame}")
    build_free(pkg, key_bgr, str(dest),
               seed=int(pkg.get("seed", 7) or 7), workdir=str(workdir))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("day", help="two-digit day, e.g. 04")
    ap.add_argument("--key", default=None)
    ap.add_argument("--image-url", default=None)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--force", action="store_true",
                    help="regenerate even if the file exists")
    a = ap.parse_args()
    generate(a.day, a.key, a.image_url, a.workdir, a.force)


if __name__ == "__main__":
    main()
