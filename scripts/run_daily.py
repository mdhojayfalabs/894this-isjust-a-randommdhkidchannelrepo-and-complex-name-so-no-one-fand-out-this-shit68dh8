#!/usr/bin/env python3
"""Publisher job (probe-driven). state/gate.json decides what to do:
  slot  = next calendar day's clip (videos/day_NN)  -> advance day counter
  spare = oldest queued extra (queue/spare_*.mp4 + sidecar .json) -> delete after
  none  = do nothing
Missing media: wait for the next probe, at most one reminder per day.
"""
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from media_check import validate
from ping import send
from uploader import upload_video

def _x(s):
    """Substitute {A} {B} {CH} {CHT} tokens. Source: NAMES env (pipe-joined
    values in fixed order A|B|CH|CHT, assembled in the workflow from NP1-NP3
    secrets) or NAMES_JSON if present."""
    import json
    n = {}
    raw = os.environ.get("NAMES_JSON", "")
    if raw:
        try:
            n = json.loads(raw)
        except Exception:
            n = {}
    if not n:
        vals = os.environ.get("NAMES", "").split("|")
        n = {k: v for k, v in zip(["A", "B", "CH", "CHT"], vals) if v}
    if not isinstance(s, str):
        return s
    for k, v in n.items():
        s = s.replace("{" + k + "}", str(v))
    return s
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alert(msg):
    """At most one reminder per day (any kind) to avoid ping spam."""
    la = os.path.join("state", "last_alert.txt")
    today = dt.date.today().isoformat()
    if os.path.exists(la) and open(la).read().strip() == today:
        return
    send(msg)
    with open(la, "w") as f:
        f.write(today)


def read_state():
    with open(os.path.join("state", "next_day.txt")) as f:
        return f.read().strip()


def write_state(day):
    with open(os.path.join("state", "next_day.txt"), "w") as f:
        f.write(f"{int(day):02d}\n")


def find_media(day):
    files = sorted(glob.glob(os.path.join("videos", f"day_{day}*.mp4")))
    return files[0] if files else None


def load_calendar(day):
    p = os.path.join("calendar", f"day_{day}.yaml")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return yaml.safe_load(f)


def read_count():
    today = dt.date.today().isoformat()
    try:
        p, n = open(os.path.join("state", "last_publish.txt")).read().split()
        return int(n) if p == today else 0
    except Exception:
        return 0


def bump_count():
    today = dt.date.today().isoformat()
    with open(os.path.join("state", "last_publish.txt"), "w") as f:
        f.write(f"{today} {read_count() + 1}\n")


def log_pub(label, vid):
    with open(os.path.join("state", "upload_log.txt"), "a") as f:
        f.write(f"{label} {dt.date.today().isoformat()} {vid}\n")


def check_placeholders(title, desc, tags):
    return "{" in title or "{" in desc or any("{" in t for t in tags)


def publish_slot():
    day = read_state()
    cal = load_calendar(day)
    if cal is None:
        alert(f"WARNING day {day}: calendar entry missing. Nothing published.")
        return 1
    media = find_media(day)
    if media is None:
        alert(f"WAIT day {day}: media not found (videos/day_{day}*.mp4). Will retry.")
        return 0
    ok, why = validate(media)
    if not ok:
        alert(f"FAIL day {day}: clip rejected ({why}). Nothing uploaded.")
        return 1
    title = _x(cal.get("title", ""))
    desc = _x(cal.get("description", ""))
    tags = [_x(t) for t in cal.get("tags", [])]
    if check_placeholders(title, desc, tags):
        alert(f"FAIL day {day}: NAMES mapping incomplete (placeholders left). Fix NAMES_JSON.")
        return 1
    try:
        res = upload_video(path=media, title=title, description=desc, tags=tags,
                           privacy="public", made_for_kids=True)
    except Exception as e:
        alert(f"FAIL day {day}: upload error: {e}")
        return 1
    write_state(f"{int(day) + 1:02d}")
    bump_count()
    log_pub(f"day{int(day):02d}", res["id"])
    send(f"OK day {day} published: {title}\nhttps://youtu.be/{res['id']}")
    return 0


def publish_spare():
    spares = sorted(glob.glob(os.path.join("queue", "spare_*.mp4")))
    if not spares:
        return 0
    spare = spares[0]
    side = spare[:-4] + ".json"
    meta = {}
    if os.path.exists(side):
        try:
            meta = json.load(open(side))
        except Exception:
            meta = {}
    title = _x(meta.get("title", ""))
    desc = _x(meta.get("description", ""))
    tags = [_x(t) for t in meta.get("tags", [])]
    if not title or check_placeholders(title, desc, tags):
        alert("WARN spare without valid metadata - removed.")
        os.remove(spare)
        if os.path.exists(side):
            os.remove(side)
        return 0
    ok, why = validate(spare)
    if not ok:
        alert(f"FAIL spare rejected ({why}). Removed.")
        os.remove(spare)
        if os.path.exists(side):
            os.remove(side)
        return 1
    try:
        res = upload_video(path=spare, title=title, description=desc, tags=tags,
                           privacy="public", made_for_kids=True)
    except Exception as e:
        alert(f"FAIL spare upload error: {e}")
        return 1
    os.remove(spare)
    if os.path.exists(side):
        os.remove(side)
    bump_count()
    log_pub("spare", res["id"])
    send(f"OK spare published: {title}\nhttps://youtu.be/{res['id']}")
    return 0


def main():
    os.chdir(repo_root())
    action = "slot"
    try:
        action = json.load(open(os.path.join("state", "gate.json"))).get("action", "slot")
    except Exception:
        pass
    if action == "spare":
        return publish_spare()
    return publish_slot()


if __name__ == "__main__":
    sys.exit(main())
