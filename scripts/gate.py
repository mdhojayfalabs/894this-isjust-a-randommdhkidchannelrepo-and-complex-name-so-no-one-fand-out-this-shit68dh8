#!/usr/bin/env python3
"""Probe gate: decides if it is time to publish today (slot or spare).
Window: 11:00-15:59 UTC. Random per-day offsets (human-like timing):
first publish = 0-2h after window start, second short = +30-90 min.
Stdlib only - runs on every probe without dependency setup.
Writes state/gate.json (consumed by run_daily.py) and step output go=0/1.
"""
import datetime as dt
import json
import os
import random


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    now = dt.datetime.now(dt.timezone.utc)
    secs = (now - now.replace(hour=11, minute=0, second=0, microsecond=0)).total_seconds()
    today = now.date().isoformat()

    plan_p = os.path.join("state", "plan.json")
    plan = None
    if os.path.exists(plan_p):
        try:
            with open(plan_p) as f:
                plan = json.load(f)
        except Exception:
            plan = None
    if not plan or plan.get("date") != today:
        off1 = random.randint(0, 7200)                      # 0-2h (owner spec)
        off2 = min(off1 + random.randint(1800, 5400), 17900)  # +30-90 min
        plan = {"date": today, "offset1": off1, "offset2": off2}
        os.makedirs("state", exist_ok=True)
        with open(plan_p, "w") as f:
            json.dump(plan, f)

    cnt = 0
    lp = os.path.join("state", "last_publish.txt")
    if os.path.exists(lp):
        try:
            p, n = open(lp).read().split()
            cnt = int(n) if p == today else 0
        except Exception:
            cnt = 0

    action = "none"
    if 0 <= secs <= 17940:
        if cnt == 0 and secs >= plan["offset1"] and secs <= 14400:
            action = "slot"          # slot retries only until 15:00 UTC
        elif cnt == 1 and secs >= plan["offset2"]:
            action = "spare"         # spare may still go until window end

    with open(os.path.join("state", "gate.json"), "w") as f:
        json.dump({"action": action, "ts": now.isoformat()}, f)
    print(f"gate: {secs:.0f}s into window plan={plan} published={cnt} action={action}")

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"go={1 if action != 'none' else 0}\n")


if __name__ == "__main__":
    main()
