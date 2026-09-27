#!/usr/bin/env python3
"""Shared YouTube Data API v3 uploader.

Quota: default 10,000 units/day; videos.insert costs 1,600 -> ~6 uploads/day.
Requires env: YT_REFRESH_TOKEN, YT_CLIENT_ID, YT_CLIENT_SECRET.
"""
import os


def _missing():
    return [k for k in ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET")
            if not os.environ.get(k)]


def upload_video(path, title, description, tags, privacy="public",
                 made_for_kids=True, thumbnail=None):
    """Upload one video. Returns the API response dict (contains id + snippet).

    made_for_kids MUST be True for this channel: the content is unambiguously
    child-directed, and mislabelling it exposes the creator to FTC penalties of
    up to $53,088 per violation. Setting it True disables comments, end screens,
    notifications and personalised ads - that is the legal cost, not a bug.
    """
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
        # NOTE: "snippet" is singular. This was previously "snippets", which the
        # API silently ignores - the upload succeeded but with no title,
        # description, tags or category.
        "snippet": {
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
    resp = yt.videos().insert(
        part="snippet,status", body=body, media_body=media).execute()

    # Custom thumbnail costs 50 quota units and is worth it for CTR.
    if thumbnail and os.path.exists(thumbnail):
        try:
            yt.thumbnails().set(
                videoId=resp["id"],
                media_body=MediaFileUpload(thumbnail, mimetype="image/png"),
            ).execute()
        except Exception as e:  # a missing thumbnail must not fail the upload
            print(f"  thumbnail skipped: {e}")
    return resp
