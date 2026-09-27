#!/usr/bin/env python3
"""
Set the channel watermark (the small branded image in the corner of every video).

    python scripts/set_watermark.py --image assets/watermark.png

Costs 50 quota units, so run it once, not per upload.

HONEST LIMITS - read before expecting anything from this:
  * The watermark renders on **desktop only**. On mobile it is visible but not
    clickable and does not act as a subscribe button.
  * Per the API reference, position.cornerPosition is effectively fixed at
    topRight; you cannot choose a different corner.
  * Because the channel is Made for Kids, viewers have no notification bell and
    no subscribe prompts in the normal places, which is exactly why the watermark
    is worth setting - it is one of the few branding surfaces left.
  * It is **branding, not a funnel**. Treat it as such.

Requirements: square, at least 150x150 px, under 1 MB, PNG or JPEG,
transparent background, and ideally one or two colours - a busy watermark pulls
attention away from the video.
"""
import argparse
import os
import sys


def set_watermark(image_path, channel_id=None, target_channel=None):
    missing = [k for k in ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError("Missing YouTube secrets: " + ", ".join(missing))

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

    if not channel_id:
        r = yt.channels().list(part="id", mine=True).execute()
        items = r.get("items") or []
        if not items:
            raise RuntimeError("no channel returned for mine=true")
        channel_id = items[0]["id"]

    body = {
        "timing": {"type": "offsetFromStart", "offsetMs": 0, "durationMs": 0},
        "position": {"type": "corner", "cornerPosition": "topRight"},
    }
    if target_channel:
        body["targetChannelId"] = target_channel

    media = MediaFileUpload(image_path, mimetype="image/png", resumable=False)
    yt.watermarks().set(channelId=channel_id, body=body, media_body=media).execute()
    print(f"watermark set on channel {channel_id}")
    return channel_id


def unset_watermark(channel_id=None):
    missing = [k for k in ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError("Missing YouTube secrets: " + ", ".join(missing))
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)
    if not channel_id:
        r = yt.channels().list(part="id", mine=True).execute()
        channel_id = (r.get("items") or [{}])[0].get("id")
    yt.watermarks().unset(channelId=channel_id).execute()
    print(f"watermark removed from {channel_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="PNG/JPEG, square, >=150x150, <1MB")
    ap.add_argument("--unset", action="store_true")
    a = ap.parse_args()
    if a.unset:
        unset_watermark()
    else:
        if not a.image:
            sys.exit("--image required (or --unset)")
        set_watermark(a.image)
