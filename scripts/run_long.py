#!/usr/bin/env python3
"""Long-form job. MODE=weekly: ~10-min compilation (latest 6 unused clips).
MODE=episode: 2-5 min episode every 3 days (latest 4 unused, min 3).
Consumed clips tracked in state/long_used.txt (shared - no double builds).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_long import build_compilation
from ping import send
from uploader import upload_video

def _x(s):
    """Substitute {A} {B} {CH} {CHT} tokens using the NAMES_JSON secret."""
    import json
    try:
        n = json.loads(os.environ.get("NAMES_JSON", "{}"))
    except Exception:
        n = {}
    if not isinstance(s, str):
        return s
    for k, v in n.items():
        s = s.replace("{" + k + "}", str(v))
    return s
def main():
    mode = os.environ.get("MODE", "weekly")
    prefix = "ep" if mode == "episode" else "week"
    if mode == "episode":
        n = int(os.environ.get("EP_CLIPS", "4"))
        min_n = max(3, n - 1)
    else:
        n = int(os.environ.get("COMPILE_CLIPS", "6"))
        min_n = n
    num = len(glob.glob(os.path.join("compilations", prefix + "_*.mp4"))) + 1
    out = build_compilation(n, min_n=min_n, mode=prefix)
    if out is None:
        send(f"SKIP {prefix} #{num}: not enough unused clips yet.")
        return 0
    if mode == "episode":
        title = _x("{A} & {B} | New 3D Cartoon for Kids! (Episode #" + str(num) + ")")
        desc = _x("A brand new 3D cartoon adventure with {A} and {B}! \U0001f366\U0001f680\U0001f8eb\n"
                  "Subscribe for daily fun shorts and new cartoons for kids!\n\n"
                  "#{CHT} #{A} #{B} #KidsCartoon #3DAnimation")
    else:
        title = _x("{A} & {B} BEST MOMENTS! \U0001f389 (Compilation #" + str(num) + ") | 3D Animated Cartoon for Kids")
        desc = _x("Watch the most magical moments of {A} and {B}! \U0001f366\U0001f680\U0001f8eb\n"
                  "Subscribe for daily fun animated shorts and long cartoons for kids!\n\n"
                  "#{CHT} #{A} #{B} #KidsCartoon #3DAnimation #KidsCompilation")
    tags = [_x("{CH}"), _x("{A} and {B}"), "kids cartoon", "3d animation kids",
            "kids video", "kids shorts"]
    if "{" in title:
        send("FAIL: build skipped - NAMES_JSON secret missing/incomplete.")
        return 1
    try:
        res = upload_video(out, title, desc, tags, privacy="public", made_for_kids=True)
    except Exception as e:
        send(f"FAIL: {prefix} #{num} upload error: {e}")
        return 1
    send(f"OK {prefix} #{num} published\nhttps://youtu.be/{res['id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
