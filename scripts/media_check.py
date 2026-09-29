#!/usr/bin/env python3
"""Validate a clip before it gets uploaded. Exit 0 = OK, 1 = rejected.

Uses ffmpeg rather than ffprobe, because the only ffmpeg available in this
pipeline comes from imageio-ffmpeg, which ships `ffmpeg` and not `ffprobe`.
The old version silently degraded to "warn-only pass" whenever ffprobe was
absent, which meant a broken clip could be uploaded unchecked.
"""
import json
import os
import re
import shutil
import subprocess
import sys


def _ffmpeg():
    """Locate an ffmpeg binary: PATH first, then imageio-ffmpeg."""
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe(path):
    """Return (duration, width, height, has_audio) or None if unprobeable."""
    ff = _ffmpeg()
    if not ff:
        return None
    out = subprocess.run([ff, "-i", path, "-f", "null", "-"],
                         capture_output=True, text=True)
    err = out.stderr or ""
    dur = w = h = 0.0
    has_audio = False
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for m in re.finditer(r"Stream #\d+:\d+.*?:\s*(\w+).*?,?\s*(\d+)x(\d+)", err):
        if m.group(1).lower() in ("video", "h264", "hevc", "vp9", "av1", "mpeg4"):
            w, h = float(m.group(2)), float(m.group(3))
    if re.search(r"Stream #\d+:\d+.*?:\s*Audio:", err):
        has_audio = True
    if not dur and not w:
        return None
    return {"duration": dur, "width": w, "height": h, "audio": has_audio}


def validate(path, min_dur=8.0, max_dur=61.0):
    if not path or not os.path.exists(path):
        return False, "file not found"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < 0.2:
        return False, f"file too small ({size_mb:.1f} MB) - probably corrupt"
    info = probe(path)
    if info is None:
        # Fail closed rather than passing: an unprobeable file is not something
        # to upload blind.
        return False, "could not probe the file (ffmpeg unavailable or output unreadable)"
    dur = info["duration"]
    if dur < min_dur or dur > max_dur:
        return False, f"duration {dur:.1f}s outside {min_dur}-{max_dur}s (Shorts max 60s)"
    w, h = info["width"], info["height"]
    if not w or not h:
        return False, "no video stream detected"
    if h <= w:
        return False, f"not vertical ({int(w)}x{int(h)}) - needs 9:16 portrait"
    if not info["audio"]:
        return False, "no audio stream - video must keep audio"
    if w < 480:
        return False, f"resolution too low ({int(w)}x{int(h)})"
    return True, f"ok {int(w)}x{int(h)} {dur:.1f}s"


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else ""
    ok, why = validate(p)
    print(json.dumps({"ok": ok, "reason": why}))
    sys.exit(0 if ok else 1)
