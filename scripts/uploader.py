#!/usr/bin/env python3
"""Shared YouTube Data API v3 uploader (free quota: 6,000 units/day ~ 3 uploads/day).
Requires env: YT_REFRESH_TOKEN, YT_CLIENT_ID, YT_CLIENT_SECRET.
"""
import os


def _missing():
    return [k for k in ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET")
            if not os.environ.get(k)]


def upload_video(path, title, description, tags, privacy="public", made_for_kids=True):
    """Upload one video. Returns the API response dict (contains id + snippet)."""
    missing = _missing()
    if missing:
        raise RuntimeError(
            "Missing YouTube secrets in GitHub repo Secrets: " + ", ".join(missing))
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippets": {
            "title": (title or "untitled")[:100],
            "description": description or "",
            "tags": tags or [],
            "categoryId": "20",  # Film & Animation
        },
        "status": {
            "privacyStatus": privacy,          # "public" | "unlisted"
            "selfDeclaredMadeForKids": bool(made_for_kids),
            "embeddable": True,
        },
    }
    media = MediaFileUpload(path, mimetype="video/mp4", chunksize=-1)
    return yt.videos().insert(part="snippet,status", body=body, media_body=media).execute()
