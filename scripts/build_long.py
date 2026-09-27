#!/usr/bin/env python3
"""Stitch the N most recent UNUSED short clips into one long-form video.
Output: compilations/week_NN.mp4 or compilations/ep_NN.mp4 (9:16, 720x1280).
Consumed clips are tracked in state/long_used.txt so the weekly compilation
and the every-3-days episode never build from the same clips.
Returns the output path, or None if not enough unused clips exist yet.
"""
import glob
import os
import re
import subprocess


def day_num(name):
    m = re.search(r"day_(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def build_compilation(n=6, min_n=None, mode="week"):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    min_n = min_n if min_n is not None else n
    used = set()
    up = os.path.join("state", "long_used.txt")
    if os.path.exists(up):
        with open(up) as f:
            used = {ln.strip() for ln in f if ln.strip()}
    files = sorted(glob.glob("videos/day_*.mp4"), key=day_num)
    files = [f for f in files if f not in used][-n:]
    if len(files) < min_n:
        print(f"skipped ({mode}): need {min_n}+ unused clips, found {len(files)}")
        return None
    os.makedirs("build", exist_ok=True)
    norm = []
    for i, f in enumerate(files):
        out = f"build/norm_{i}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", f,
             "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,"
                    "pad=720:1280:(ow-iw)/2:(oh-ih)/2",
             "-r", "24", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
             "-ar", "44100", "-ac", "2", out],
            check=True, capture_output=True)
        norm.append(out)
    with open("build/list.txt", "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in norm)
    os.makedirs("compilations", exist_ok=True)
    n_comp = len(glob.glob(f"compilations/{mode}_*.mp4")) + 1
    out = f"compilations/{mode}_{n_comp:02d}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "build/list.txt",
         "-c", "copy", out],
        check=True, capture_output=True)
    with open(up, "a") as f:
        for p in files:
            f.write(p + "\n")
    print(f"{mode} built: {out}")
    return out
