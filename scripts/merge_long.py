#!/usr/bin/env python3
"""
Merge finished daily Shorts into one long-form video.

This is the second half of the schedule the user asked for:

    every day      -> one Short (day_NN.mp4)                 [w1 publishes it]
    every Sunday   -> that week's 7 Shorts merged into one long-form  [w8]
    every 1st      -> that month's ~30 clips merged into one long-form,
                      then the month's clips are archived out of the way  [w9]

Nothing here re-encodes the video: the clips are already 1080x1920 h264/aac,
so they are concatenated with the stream copier and only re-muxed. That keeps
it fast and, more importantly, keeps it lossless - merging is not allowed to
degrade the source clips.

Usage:
    python scripts/merge_long.py week            # the 7 clips of the current week
    python scripts/merge_long.py month           # every clip of the current month
    python scripts/merge_long.py week  --dry-run
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEOS = ROOT / "videos"
OUT = ROOT / "longform"
STATE = ROOT / "state"


def log(msg):
    print(f"[merge] {msg}", flush=True)


def ffmpeg():
    """imageio-ffmpeg ships a static ffmpeg; system ffmpeg is absent here."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise SystemExit(f"ffmpeg not found: {e}")


def read_upload_log():
    """[(day_int, iso_date, video_id)] for every successfully published clip."""
    rows = []
    p = STATE / "upload_log.txt"
    if not p.exists():
        return rows
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].startswith("day"):
            try:
                rows.append((int(parts[0][3:]), parts[1], parts[2]))
            except ValueError:
                continue
    return rows


def clips_for(scope, today=None):
    """The (day, date, path) triples belonging to `scope`."""
    today = today or dt.date.today()
    rows = read_upload_log()
    out = []
    for day, iso, _vid in rows:
        try:
            d = dt.date.fromisoformat(iso)
        except ValueError:
            continue
        if scope == "week":
            # Monday-based week containing `today`
            monday = today - dt.timedelta(days=today.weekday())
            if not (monday <= d <= monday + dt.timedelta(days=6)):
                continue
        elif scope == "month":
            if not (d.year == today.year and d.month == today.month):
                continue
        else:
            continue
        for cand in (VIDEOS / f"day_{day:02d}.mp4", VIDEOS / f"day_{day}.mp4"):
            if cand.exists():
                out.append((day, iso, cand))
                break
    return sorted(out)


def probe(path):
    """Duration in seconds, parsed out of ffmpeg's stderr banner."""
    ff = ffmpeg()
    r = subprocess.run(
        [ff, "-hide_banner", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True)
    txt = r.stderr or ""
    for line in txt.splitlines():
        if "Duration:" not in line:
            continue
        # "  Duration: 00:00:02.00, start: 0.000000, bitrate: 14 kb/s"
        stamp = line.split("Duration:")[1].split(",")[0].strip()
        parts = stamp.split(":")
        if len(parts) == 3:
            try:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return 0.0
    return 0.0


def concat(clips, out_path):
    """Stream-copy concat. Every clip must share codec/params, which they do
    because every clip comes from the same renderer."""
    ff = ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lst = out_path.with_suffix(".txt")
    lst.write_text(
        "".join(f"file '{c.resolve()}'\n" for _d, _i, c in clips))
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(lst),
           "-c", "copy", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"concat failed: {r.stderr[-800:]}")
    lst.unlink(missing_ok=True)
    return out_path


def title_and_desc(scope, clips, today):
    first, last = clips[0][1], clips[-1][1]
    n = len(clips)
    total = sum(probe(c) for _d, _i, c in clips)
    if scope == "week":
        wk = today.isocalendar()[1]
        title = (f"Hoji and Mariyam - Week {wk} Full Episode "
                 f"({n} clips, {int(total)}s) #Shorts #Compilation")
        desc = (
            f"Every Short from week {wk} of {today.year}, in order, "
            f"unedited.\n\n"
            f"{n} clips - {int(total)} seconds - {first} to {last}.\n\n"
            "New Short every day. Full episode every Sunday.\n\n"
            "#Shorts #HojiMariyam #KidsCartoon #Compilation")
    else:
        title = (f"Hoji and Mariyam - {today:%B %Y} Full Month "
                 f"({n} clips, {int(total)}s) #Compilation")
        desc = (
            f"Every clip from {today:%B %Y}, in order, unedited.\n\n"
            f"{n} clips - {int(total)} seconds.\n\n"
            "New Short every day. Full episode every Sunday. "
            "Full month on the 1st.\n\n"
            "#Shorts #HojiMariyam #KidsCartoon #Compilation")
    return title, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scope", choices=["week", "month"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--today", help="YYYY-MM-DD, for testing")
    args = ap.parse_args()

    today = (dt.date.fromisoformat(args.today) if args.today
             else dt.date.today())
    clips = clips_for(args.scope, today)

    if not clips:
        log(f"no {args.scope} clips found for {today}; nothing to merge")
        print(json.dumps({"action": "none", "scope": args.scope,
                          "today": today.isoformat()}))
        return 0

    days = ", ".join(f"{d:02d}" for d, _i, _c in clips)
    total = sum(probe(c) for _d, _i, c in clips)
    log(f"{args.scope} {today}: {len(clips)} clips (days {days}), "
        f"{total:.1f}s total")
    for d, iso, c in clips:
        log(f"    day {d:02d}  {iso}  {c.name}  {c.stat().st_size / 1e6:.1f} MB")

    title, desc = title_and_desc(args.scope, clips, today)
    log(f"title: {title}")

    if args.dry_run:
        log("--dry-run: not writing anything")
        print(json.dumps({"action": "dry-run", "scope": args.scope,
                          "clips": len(clips), "seconds": round(total, 1),
                          "title": title}))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    if args.scope == "week":
        name = f"week_{today.isocalendar()[1]:02d}_{today.year}.mp4"
    else:
        name = f"month_{today.year}_{today.month:02d}.mp4"
    out_path = OUT / name

    if out_path.exists():
        log(f"{out_path.name} already exists; rebuilding anyway")

    concat(clips, out_path)
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    meta = {
        "scope": args.scope,
        "today": today.isoformat(),
        "clips": [d for d, _i, _c in clips],
        "seconds": round(total, 1),
        "file": str(out_path.relative_to(ROOT)),
        "title": title,
        "description": desc,
        "tags": ["Hoji & Mariyam Kids", "Hoji", "Mariyam", "Shorts",
                 "Compilation", "Kids Cartoon", "Full Episode"],
        "privacy": "public",
        "made_for_kids": True,
    }
    (OUT / f"{out_path.stem}.json").write_text(json.dumps(meta, indent=2))
    log(f"wrote {OUT / (out_path.stem + '.json')}")
    print(json.dumps({"action": "merged", **{k: meta[k] for k in
                                             ("scope", "clips", "seconds",
                                              "file", "title")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
