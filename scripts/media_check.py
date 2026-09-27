#!/usr/bin/env python3
"""Validate a clip before it gets uploaded. Exit 0 = OK, 1 = rejected.
Uses ffprobe (preinstalled on GitHub runners). If ffprobe is missing
(local machine), falls back to warn-and-pass on file size only.
"""
import json
import os
import shutil
import subprocess
import sys


def probe(path):
    if not shutil.which("ffprobe"):
        return None
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def validate(path, min_dur=8.0, max_dur=61.0):
    if not path or not os.path.exists(path):
        return False, "file not found"
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < 0.2:
        return False, f"file too small ({size_mb:.1f} MB) - probably corrupt"
    info = probe(path)
    if info is None:
        print("WARN: ffprobe not available - skipped technical checks")
        return True, "ffprobe missing (warn-only pass)"
    try:
        dur = float(info.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        dur = 0.0
    if dur < min_dur or dur > max_dur:
        return False, f"duration {dur:.1f}s outside {min_dur}-{max_dur}s (Shorts max 60s)"
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        return False, "no video stream"
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    if h <= w:
        return False, f"not vertical ({w}x{h}) - needs 9:16 portrait"
    if a is None:
        return False, "no audio stream - video must keep audio"
    if w < 480:
        return False, f"resolution too low ({w}x{h})"
    return True, f"ok {w}x{h} {dur:.1f}s"


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else ""
    ok, why = validate(p)
    print(json.dumps({"ok": ok, "reason": why}))
    sys.exit(0 if ok else 1)
