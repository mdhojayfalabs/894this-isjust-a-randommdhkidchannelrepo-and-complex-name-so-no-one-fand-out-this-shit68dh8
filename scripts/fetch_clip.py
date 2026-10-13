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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "mdhkids/hy-clip"

# Kaggle credentials. Written to a config file rather than left in the
# environment: the env-var path works locally but is rejected on the runner,
# and a config file is the only path proven on both.
_CFG_DIR = None


def log(msg):
    print(f"[fetch] {msg}", flush=True)


def cfg_dir():
    """Materialise kaggle.json once and return the dir holding it."""
    global _CFG_DIR
    if _CFG_DIR:
        return _CFG_DIR
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not user or not key:
        log("KAGGLE_USERNAME / KAGGLE_KEY not set; no credentials to write")
        _CFG_DIR = ""
        return _CFG_DIR
    d = tempfile.mkdtemp(prefix="kaggle-")
    f = Path(d, "kaggle.json")
    f.write_text(json.dumps({"username": user, "key": key}))
    os.chmod(f, 0o600)
    _CFG_DIR = d
    return d


def sh(cmd, **kw):
    env = dict(os.environ)
    d = cfg_dir()
    if d:
        env["KAGGLE_CONFIG_DIR"] = d
    return subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)


def kaggle(*args):
    return sh([sys.executable, "-m", "kaggle", *args])


def diagnose():
    """Safe auth probe: lengths and booleans only, never credential material."""
    user = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    log(f"creds present: username={bool(user)} (len {len(user)}), "
        f"key={bool(key)} (len {len(key)})")
    if not (user and key):
        return False
    r = kaggle("kernels", "list", "--mine", "-p", "1")
    out = ((r.stdout or "") + (r.stderr or ""))
    ok = r.returncode == 0
    log(f"kaggle kernels list --mine: rc={r.returncode} "
        f"auth={'OK' if ok else 'REJECTED'}")
    if not ok:
        log(f"    {out.strip()[:400]}")
    return ok


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
    # Never truncate: a truncated error is an undiagnosable error.
    return "UNKNOWN", txt.strip()[-1500:]


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
    if not diagnose():
        log("not authenticated to Kaggle; will retry next pass")
        print(json.dumps({"action": "wait", "day": day, "reason": "no-auth"}))
        return 0

    have = already_have(day)
    if have:
        log(f"{have.relative_to(ROOT)} already present; nothing to fetch")
        print(json.dumps({"action": "have", "day": day,
                          "file": str(have.relative_to(ROOT))}))
        return 0

    # Output first: this is the only step that actually unblocks publishing,
    # and it works for a private kernel that we own.
    dest = download(day)
    if dest is not None:
        print(json.dumps({"action": "fetched", "day": day,
                          "file": str(dest.relative_to(ROOT))}))
        return 0

    state, detail = kernel_status()
    log(f"kernel {KERNEL}: {state}")
    if state not in ("UNKNOWN",):
        log(f"    {detail[:400]}")

    if state in ("RUNNING", "QUEUED"):
        # ~9 h render; w7 will be back.
        log("render in progress; nothing to do this pass")
        print(json.dumps({"action": "wait", "day": day, "kernel": state}))
        return 0

    # COMPLETE but no clip, ERROR, or UNKNOWN: restart the render if we can.
    kdir = ROOT / "kaggle" / "hyclip"
    if not (kdir / "kernel-metadata.json").exists():
        # Do not hard-fail: w7 runs every 20 min and a red run on every pass
        # is noise, not signal. The clip download above already succeeded or
        # will be retried next pass.
        log(f"no kernel dir at {kdir.relative_to(ROOT)}; cannot re-trigger")
        log("will retry next pass")
        print(json.dumps({"action": "wait", "day": day,
                          "kernel": state, "reason": "no-kernel-dir"}))
        return 0

    if trigger():
        log("render re-triggered")
        print(json.dumps({"action": "triggered", "day": day}))
        return 0

    log("could not re-trigger; will retry next pass")
    print(json.dumps({"action": "wait", "day": day, "kernel": state}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
