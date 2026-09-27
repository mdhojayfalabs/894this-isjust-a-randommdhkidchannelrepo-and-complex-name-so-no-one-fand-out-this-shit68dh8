#!/usr/bin/env python3
"""Weekly repo-bloat fix: delete local copies of clips that were successfully
uploaded to YouTube more than N days ago (default 14). The video already lives
on YouTube, so the repo copy is only a short-term backup.
Log format (state/upload_log.txt):  <day> <YYYY-MM-DD> <video_id>
"""
import datetime as dt
import glob
import os

KEEP_DAYS = int(os.environ.get("CLEANUP_KEEP_DAYS", "14"))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    log = "state/upload_log.txt"
    if not os.path.exists(log):
        print("no upload log - nothing to clean")
        return 0
    cutoff = dt.date.today() - dt.timedelta(days=KEEP_DAYS)
    removed = []
    with open(log) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            day, date_s = parts[0], parts[1]
            try:
                uploaded_on = dt.date.fromisoformat(date_s)
            except ValueError:
                continue
            if uploaded_on <= cutoff:
                for p in glob.glob(f"videos/day_{day}*.mp4"):
                    os.remove(p)
                    removed.append(os.path.basename(p))
    if removed:
        print("removed: " + ", ".join(removed))
    else:
        print("nothing to clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
