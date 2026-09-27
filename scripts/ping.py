#!/usr/bin/env python3
"""Telegram notification (optional). If TG_BOT_TOKEN / TG_CHAT_ID are not set,
it just prints - the pipeline never fails because of notifications.
"""
import os

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


def send(text):
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat = os.environ.get("TG_CHAT_ID", "")
    if not token or not chat or requests is None:
        print(f"[notify:skipped] {text}")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML"},
            timeout=15,
        )
        r.raise_for_status()
        print(f"[notify:ok] {text[:80]}...")
        return True
    except Exception as e:  # never let notifications kill the pipeline
        print(f"[notify:failed] {e}")
        return False
