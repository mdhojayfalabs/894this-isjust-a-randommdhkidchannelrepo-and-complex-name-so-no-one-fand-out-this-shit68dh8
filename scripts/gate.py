#!/usr/bin/env python3
"""Daily publish gate.

The publish time is ONE random number of seconds inside the 24-hour UTC day,
drawn fresh each day and stored in state/plan.json. The workflow probes every
20 minutes; the first probe at or after that second is the one that publishes.

This replaces the old two-slot scheme inside an 11:00-15:59 UTC window. That
scheme was wrong on two counts: it confined publishing to a 5-hour slice of the
day, and it used fixed offsets (0-2h, then +30-90 min) rather than a random
second anywhere in 0-24h. It also never fired, which is why 12 consecutive
Actions runs wrote gate.json = {"action": "none"} and uploaded nothing.

Actions in state/gate.json, consumed by run_daily.py:
  slot  - publish the next calendar day's clip
  skip  - today's attempts are exhausted; advance the day counter unpublished
  none  - do nothing

Attempts are capped at MAX_ATTEMPTS so a persistently failing generation cannot
spin forever; a fail is a normal transient (Pixazo congestion), and a retry 20
minutes later is the right response, but six is the limit.

Stdlib only - this runs on every probe before any dependency install.
"""
import datetime as dt
import json
import os
import random

MAX_ATTEMPTS = 6
DAY = 86400


def _read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_plan(today):
    """Today's random target second, plus the attempt counter."""
    path = os.path.join("state", "plan.json")
    plan = _read_json(path)
    if not plan or plan.get("date") != today:
        # A brand-new day: one random second anywhere in 0-24h.
        plan = {
            "date": today,
            "target": random.randint(0, DAY - 1),
            "attempts": 0,
        }
        os.makedirs("state", exist_ok=True)
        with open(path, "w") as f:
            json.dump(plan, f)
    plan.setdefault("attempts", 0)
    return plan


def published_today(today):
    try:
        d, n = open(os.path.join("state", "last_publish.txt")).read().split()
        return d == today and int(n) > 0
    except Exception:
        return False


def write_gate(action, now):
    """Write gate.json only when the action changes.

    The old version rewrote it on every probe with a fresh timestamp, which
    forced the workflow's commit step to push on all 72 daily probes. Writing
    only on change keeps the history readable.
    """
    path = os.path.join("state", "gate.json")
    payload = {"action": action}
    try:
        if _read_json(path, {}) == payload:
            return
    except Exception:
        pass
    os.makedirs("state", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    now = dt.datetime.now(dt.timezone.utc)
    today = now.date().isoformat()
    secs = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()

    plan = load_plan(today)
    target = int(plan["target"])
    attempts = int(plan.get("attempts", 0))

    if published_today(today):
        action = "none"
        reason = "already published today"
    elif secs < target:
        action = "none"
        reason = f"waiting {int(target - secs)}s for today's random slot"
    elif attempts >= MAX_ATTEMPTS:
        action = "skip"
        reason = f"{attempts} failed attempts today - skipping to the next day"
    else:
        action = "slot"
        reason = f"random second {target} reached ({int(secs)}s into the day)"

    # Count the attempt as soon as the gate opens, so a failing run cannot
    # re-open the same slot indefinitely.
    if action == "slot":
        plan["attempts"] = attempts + 1
        with open(os.path.join("state", "plan.json"), "w") as f:
            json.dump(plan, f)

    write_gate(action, now)
    print(f"gate: {int(secs)}s into day | target={target} | attempts={attempts} "
          f"| published={published_today(today)} | action={action} ({reason})")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"go={1 if action in ('slot', 'skip') else 0}\n")


if __name__ == "__main__":
    main()
