#!/usr/bin/env python3
"""
Build packages/day_NN.json for every day in content/CONTENT_PACKAGES.md.

    python scripts/make_packages.py

WHY: the content plan stores shots as prose inside two prompt blocks per day.
The builder needs one JSON object per day with a list of clip motions and a
list of timed SFX beats. This bridges the two.

Clip count: the content plan has 6 shots per day (2 blocks x 3 shots). The
research target is a 22-45 second Short, and the free LTX model emits ~5s per
clip, so 6 shots become 5 clips:
    clip0 = shot1 + shot2   (the hook - face and action together, early)
    clip1 = shot3
    clip2 = shot4
    clip3 = shot5
    clip4 = shot6           (the loop-back beat)
The first clip carries two shots because the opening has to establish the
character AND the action inside the two seconds that decide swipe-away.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "content" / "CONTENT_PACKAGES.md"
OUT = ROOT / "packages"

# SFX vocabulary that gen_audio.py can actually synthesise.
SFX = ["pop", "sparkle", "whoosh", "tada", "boing", "coin", "click"]

# A beat pattern that fits a 112 BPM bed (0.535s per beat). Repeated every
# ~5s, i.e. once per clip, so each cut lands on an accent.
BEAT_TEMPLATE = [
    (0.00, "whoosh", 0.45),
    (0.60, "pop", 0.55),
    (1.60, "boing", 0.45),
    (3.20, "sparkle", 0.55),
]


def parse_days(text):
    parts = re.split(r"\n## Day (\d+) — ", text)
    days = {}
    for i in range(1, len(parts), 2):
        num = int(parts[i])
        body = parts[i + 1]
        title = body.split("\n")[0].strip()
        title = re.sub(r"\s*—.*$", "", title).strip()
        days[num] = (title, body)
    return days


def parse_shots(body):
    """Return (setting, [shot strings]) across both clip blocks."""
    setting = ""
    shots = []
    for m in re.finditer(
            r"### Block \d — Clip ([AB]) prompt \(copy exactly\)\s*\n```\n(.*?)```",
            body, re.S):
        text = m.group(2)
        s = re.search(r"Setting:\s*(.+)", text)
        if s and not setting:
            setting = s.group(1).strip().rstrip(".")
        for sh in re.findall(r"Shot \d \([^)]*\):\s*(.+)", text):
            shots.append(sh.strip().rstrip("."))
    return setting, shots


def build_clips(setting, shots):
    """6 shots -> 5 clips. First clip takes two shots (hook density)."""
    if len(shots) < 5:
        return None
    groups = [
        [shots[0], shots[1]],
        [shots[2]],
        [shots[3]],
        [shots[4]],
        [shots[5]] if len(shots) > 5 else [shots[-1]],
    ]
    clips = []
    for g in groups:
        motion = " ".join(g)
        if setting:
            motion = f"Setting: {setting}. {motion}"
        clips.append(motion)
    return clips


def build_beats(n_clips, clip_len=5.0):
    beats = []
    for i in range(n_clips):
        base = i * clip_len
        for off, cat, gain in BEAT_TEMPLATE:
            beats.append({"t": round(base + off, 2), "cat": cat, "gain": gain})
        # a reveal accent near the end of each clip
        beats.append({"t": round(base + clip_len - 0.8, 2), "cat": "tada", "gain": 0.7})
    beats.sort(key=lambda b: b["t"])
    return beats


def main():
    if not SRC.exists():
        raise SystemExit(f"source not found: {SRC}")
    days = parse_days(SRC.read_text())
    OUT.mkdir(exist_ok=True)
    written = 0
    for d in sorted(days):
        title, body = days[d]
        setting, shots = parse_shots(body)
        clips = build_clips(setting, shots)
        if not clips:
            print(f"  day {d}: skipped (only {len(shots)} shots parsed)")
            continue
        pkg = {
            "day": d,
            "title": title,
            "bpm": 112,
            "seed": 7 + d,
            "loop_xfade": 0.6,
            "clips": clips,
            "beats": build_beats(len(clips)),
        }
        p = OUT / f"day_{d:02d}.json"
        p.write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n")
        total = len(clips) * 5.0
        print(f"  day {d:02d}: {len(clips)} clips, ~{total:.0f}s  -> {p.name}")
        written += 1
    print(f"\n{written} packages written to {OUT}")


if __name__ == "__main__":
    main()
