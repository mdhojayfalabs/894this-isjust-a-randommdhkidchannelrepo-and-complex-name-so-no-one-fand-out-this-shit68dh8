#!/usr/bin/env python3
"""
Fetch a finished Kaggle render into videos/day_NN.mp4 so w1 can publish it.

Why this exists: the render takes ~9 hours (chunked attention on a T4), but a
GitHub Actions job is capped at 6 hours. Render and publish therefore cannot
live in the same job. w7 runs often, does at most one small piece of work, and
exits:

    kernel not running and no clip yet  -> push a new kernel version, exit
    kernel running, no clip yet         -> do nothing, exit (come back later)
    kernel finished, clip available     -> download clip.mp4, write
                                           videos/day_NN.mp4, commit, exit

w1 then finds videos/day_NN.mp4 through find_media() and never has to render
anything itself.

Env:
    KAGGLE_USERNAME / KAGGLE_KEY   Kaggle API credentials
"""

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "mdhkids/hy-clip"


def log(msg):
    print(f"[fetch] {msg}", flush=True)


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def kaggle(*args):
    return sh([sys.executable, "-m", "kaggle", *args])


def read_day():
    p = ROOT / "state" / "next_day.txt"
    return int(p.read_text().strip())


def already_have(day):
    for c in (ROOT / "videos" / f"day_{day:02d}.mp4",
              ROOT / "videos" / f"day_{day}.mp4"):
        if c.exists() and c.stat().st_size > 100_000:
            return c
    return None


def kernel_status():
    r = kaggle("kernels", "status", KERNEL)
    txt = (r.stdout or "") + (r.stderr or "")
    for state in ("COMPLETE", "ERROR", "RUNNING", "QUEUED", "CANCEL"):
        if f"KernelWorkerStatus.{state}" in txt:
            return state, txt.strip().splitlines()[-1]
    return "UNKNOWN", txt.strip()[-200:]


def trigger():
    """Re-push the kernel so Kaggle starts a fresh run of the current code."""
    kdir = ROOT / "kaggle" / "hyclip"
    if not (kdir / "kernel-metadata.json").exists():
        log(f"no kernel dir at {kdir}; cannot trigger")
        return False
    r = kaggle("kernels", "push", "-p", str(kdir))
    if r.returncode != 0:
        log(f"push failed: {(r.stderr or r.stdout)[-300:]}")
        return False
    log(f"triggered: {(r.stdout or '').strip().splitlines()[0]}")
    return True


def download(day):
    with tempfile.TemporaryDirectory() as td:
        r = kaggle("kernels", "output", KERNEL, "-p", td)
        if r.returncode != 0:
            log(f"output download failed: {(r.stderr or '')[-300:]}")
            return None
        src = None
        for cand in Path(td).rglob("*.mp4"):
            if cand.stat().st_size > 100_000:
                src = cand
                break
        if src is None:
            log("kernel finished but produced no mp4")
            return None
        dest = ROOT / "videos" / f"day_{day:02d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        log(f"wrote {dest.relative_to(ROOT)} "
            f"({dest.stat().st_size / 1e6:.1f} MB)")
        return dest


def main():
    day = read_day()
    log(f"next_day = {day:02d}")

    have = already_have(day)
    if have:
        log(f"{have.relative_to(ROOT)} already present; nothing to fetch")
        print(json.dumps({"action": "have", "day": day,
                          "file": str(have.relative_to(ROOT))}))
        return 0

    state, detail = kernel_status()
    log(f"kernel {KERNEL}: {state} ({detail})")

    if state == "RUNNING" or state == "QUEUED":
        # ~9 h render; w7 will be back.
        print(json.dumps({"action": "wait", "day": day, "kernel": state}))
        return 0

    if state in ("COMPLETE", "ERROR"):
        dest = download(day)
        if dest is None:
            log("no clip; re-triggering the kernel")
            if trigger():
                print(json.dumps({"action": "retrigger", "day": day}))
                return 0
            print(json.dumps({"action": "failed", "day": day}))
            return 1
        print(json.dumps({"action": "fetched", "day": day,
                          "file": str(dest.relative_to(ROOT))}))
        return 0

    # UNKNOWN - most likely never run in this session. Start it.
    log("kernel state unknown; triggering a fresh run")
    if trigger():
        print(json.dumps({"action": "triggered", "day": day}))
        return 0
    print(json.dumps({"action": "failed", "day": day}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
