#!/usr/bin/env python3
"""Daily analytics job: pull stats for the last 50 videos and append to
reports/analytics.csv (the workflow commits it). Free Data API call (~1 unit).
"""
import csv
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "reports", "analytics.csv")


def token_health():
    """One cheap API ping per run to catch a dead/expired push token early."""
    tok = os.environ.get("PUSH_TOKEN", "")
    if not tok:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.github.com/user",
            headers={"Authorization": "token " + tok,
                     "Accept": "application/vnd.github+json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        code = getattr(e, "code", None)
        if code in (401, 403):
            from ping import send
            send(f"ALERT: push token invalid or expired (HTTP {code}). "
                 "Owner: one fresh token needed - reply here when done.")


def main():
    token_health()
    missing = [k for k in ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET")
               if not os.environ.get(k)]
    if missing:
        print("YouTube secrets missing - analytics skipped")
        return 0
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    r = yt.videos().list(part="snippet,statistics", chart="mostRecent",
                         maxResults=50).execute()
    rows = []
    for v in r.get("items", []):
        s, st = v["snippet"], v["statistics"]
        rows.append([
            dt.date.today().isoformat(),
            v["id"],
            int(st.get("viewCount", 0) or 0),
            int(st.get("likeCount", 0) or 0),
        ])
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    is_new = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    with open(CSV, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["date", "video_id", "views", "likes"])
        w.writerows(rows)
    print(f"logged stats for {len(rows)} videos")


if __name__ == "__main__":
    sys.exit(main())
