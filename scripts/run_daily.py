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
    """Substitute {A} {B} {CH} {CHT} tokens.

    Source precedence:
      1. NAMES_JSON env (explicit JSON object)
      2. NAMES env, if it parses as JSON
      3. NAMES env as pipe-joined values A|B|CH|CHT (legacy chunked format)

    The repo secret NAMES is now stored as JSON, so path 2 is the live path.
    Path 3 is kept only for backwards compatibility with older runs.
    """
    n = {}
    for raw in (os.environ.get("NAMES_JSON", ""), os.environ.get("NAMES", "")):
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                n = parsed
                break
        except Exception:
            pass
    if not n:
        vals = os.environ.get("NAMES", "").split("|")
        n = {k: v for k, v in zip(["A", "B", "CH", "CHT"], vals) if v}
    # Defensive: YAML parses `- {CH}` as the flow mapping {'CH': None}, not the
    # string "{CH}". Anything non-str is stringified so a malformed calendar
    # cannot smuggle a dict into the upload body.
    if not isinstance(s, str):
        s = str(s)
    for k, v in n.items():
        s = s.replace("{" + k + "}", str(v))
    return s
def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def alert(msg):
    """Send at most one reminder per day to avoid ping spam.

    The suppression is Telegram-side ONLY. The message is always written to
    stdout, because the previous behaviour hid real failures: a failed run wrote
    last_alert.txt, so every later run returned non-zero with no output at all
    and the cause was invisible in the workflow log.
    """
    print(f"[alert] {msg}", flush=True)
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
    """Locate the finished Short for this day, generating it if absent.

    The old design assumed videos/day_NN.mp4 was already on disk, so every run
    died with "WAIT day NN: media not found". Generation is now part of the
    publish path: build the clip, then upload it.
    """
    files = sorted(glob.glob(os.path.join("videos", f"day_{day}*.mp4")))
    if files:
        return files[0]
    # not there yet - try to make it
    try:
        from generate_day import generate
        p = generate(day)
        return str(p) if p else None
    except SystemExit as e:
        print(f"generate_day: {e}")
        return None
    except Exception as e:
        print(f"generate_day failed: {type(e).__name__}: {e}")
        return None


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
        # Distinguish the real causes so the Telegram alert is actionable.
        if not os.path.exists(os.path.join("packages", f"day_{day}.json")):
            alert(f"WAIT day {day}: no packages/day_{day}.json exists, so there "
                  f"is nothing to generate.")
        elif not os.environ.get("PIXAZO_API_KEY"):
            alert(f"FAIL day {day}: PIXAZO_API_KEY missing from the workflow "
                  f"env, so generation cannot run.")
        else:
            alert(f"WAIT day {day}: generation failed - see the workflow log. "
                  f"Will retry on the next probe.")
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
