#!/usr/bin/env python3
"""
Procedural audio for MDH Kids Studio.

WHY SYNTHESISED AND NOT SAMPLED
--------------------------------
Three reasons, in order of importance:

1. YouTube added an **original-sound bonus for channels under 50K subscribers**
   (March 2026). Channels using their own music instead of trending audio get a
   measurable distribution lift. Synthesised audio is, by definition, original.
2. Licensing. Every free SFX library is either "check the licence per file"
   (Freesound), "attribution required" (ZapSplat, FreeSFX), or has no API at all.
   The Pixabay API covers images and video but **not audio**. Synthesising means
   zero licence risk, zero attribution, zero manual downloads.
3. Zero-manual-work. Nothing to fetch, nothing to attribute, nothing to expire.

Output is deterministic: the same seed always produces byte-identical audio.
"""
import hashlib
import math
import struct
import wave
from pathlib import Path

import numpy as np

SR = 44100  # sample rate


# ---------------------------------------------------------------- primitives
def _env(n, attack, decay):
    """Attack/decay envelope, equal-power fades at both ends."""
    a = max(1, int(attack * SR))
    d = max(1, int(decay * SR))
    e = np.ones(n, dtype=np.float64)
    if a > 0:
        e[:a] = np.linspace(0.0, 1.0, a) ** 0.5
    if d > 0:
        e[-d:] = np.linspace(1.0, 0.0, d) ** 0.5
    return e


def _sweep(n, f0, f1, curve=0.72):
    """Exponential pitch sweep. The exponent makes the change happen early,
    which is what makes a sound feel snappy rather than droopy."""
    t = np.linspace(0.0, 1.0, n)
    phase = np.cumsum(f0 * (f1 / f0) ** (t ** curve)) / SR
    return phase


def _write(path, samples):
    samples = np.clip(samples, -1.0, 1.0)
    # peak normalise to -1 dBFS
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.891
    samples -= samples.mean()  # DC removal
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def _variant_seed(category, name, variant=0):
    h = hashlib.sha256(f"{category}:{name}:{variant}".encode()).hexdigest()
    return int(h[:8], 16)


# ---------------------------------------------------------------- filters
def _biquad_lowpass(x, fc, q=0.707):
    """RBJ lowpass. Vectorised via lfilter-style recursion."""
    w0 = 2 * math.pi * min(fc, SR * 0.49) / SR
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    b0 = (1 - cw) / 2
    b1 = 1 - cw
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * cw
    a2 = 1 - alpha
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1.0, a1 / a0, a2 / a0]
    return _apply_df(x, b, a)


def _biquad_highpass(x, fc, q=0.707):
    w0 = 2 * math.pi * min(fc, SR * 0.49) / SR
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2 * q)
    b0 = (1 + cw) / 2
    b1 = -(1 + cw)
    b2 = b0
    a0 = 1 + alpha
    a1 = -2 * cw
    a2 = 1 - alpha
    b = [b0 / a0, b1 / a0, b2 / a0]
    a = [1.0, a1 / a0, a2 / a0]
    return _apply_df(x, b, a)


def _apply_df(x, b, a):
    """Direct-form II transposed biquad, applied in chunks to stay fast."""
    n = len(x)
    y = np.empty(n)
    z1 = z2 = 0.0
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    # process in a tight loop; numpy cannot vectorise a recursive filter
    for i in range(n):
        xn = x[i]
        yn = b0 * xn + z1
        z1 = b1 * xn - a1 * yn + z2
        z2 = b2 * xn - a2 * yn
        y[i] = yn
    return y


def _tilt(x, low_gain, mid_gain, high_gain, low_cut, high_cut):
    """Three-band gain. Mid is the residual after low and high are removed."""
    low = _biquad_lowpass(x, low_cut)
    high = _biquad_highpass(x, high_cut)
    mid = x - _biquad_lowpass(x, low_cut) - _biquad_highpass(x, high_cut)
    return low * low_gain + mid * mid_gain + high * high_gain


# ---------------------------------------------------------------- SFX
def sfx(category, name, variant=0, seconds=None):
    """Build one sound effect. Returns float64 mono at SR."""
    seed = _variant_seed(category, name, variant)
    rng = np.random.default_rng(seed)
    # small deterministic variation so repeated hits don't sound robotic
    pitch_jitter = 1.0 + (rng.random() - 0.5) * 0.06

    if category == "pop":
        sec = seconds if seconds else 0.16
        n = int(sec * SR)
        t = np.arange(n) / SR
        f = 420 * pitch_jitter * np.exp(-t * 26)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, sec)

    elif category == "sparkle":
        sec = seconds if seconds else 0.55
        n = int(sec * SR)
        t = np.arange(n) / SR
        # three rising partials = classic "magic shimmer"
        s = np.zeros(n)
        for k, (f0, f1, amp) in enumerate([(1400, 4200, 1.0),
                                           (2100, 6300, 0.5),
                                           (700, 2100, 0.35)]):
            ph = _sweep(n, f0 * pitch_jitter, f1 * pitch_jitter)
            s += amp * np.sin(2 * np.pi * ph)
        s *= _env(n, 0.004, sec) * (0.6 + 0.4 * np.sin(2 * np.pi * 18 * t))

    elif category == "whoosh":
        sec = seconds if seconds else 0.42
        n = int(sec * SR)
        noise = rng.standard_normal(n)
        # simple one-pole lowpass whose cutoff follows the sweep
        ph = _sweep(n, 300 * pitch_jitter, 2600 * pitch_jitter)
        cut = 300 + 2600 * (np.linspace(0, 1, n) ** 1.4)
        alpha = np.clip(cut / SR, 0.01, 0.9)
        y = np.zeros(n)
        acc = 0.0
        for i in range(n):
            acc += alpha[i] * (noise[i] - acc)
            y[i] = acc
        s = y * _env(n, 0.01, sec)

    elif category == "tada":
        # rising arpeggio + noise burst = "reveal"
        sec = seconds if seconds else 0.85
        n = int(sec * SR)
        s = np.zeros(n)
        for i, semis in enumerate((0, 4, 7, 12)):
            t0 = i * 0.085
            i0 = int(t0 * SR)
            ln = n - i0
            if ln <= 0:
                continue
            f = 523.25 * (2 ** (semis / 12)) * pitch_jitter
            t = np.arange(ln) / SR
            note = (np.sin(2 * np.pi * f * t) * 0.7
                    + np.sin(2 * np.pi * f * 2 * t) * 0.2)
            s[i0:] += note * _env(ln, 0.004, 0.30)
        s += rng.standard_normal(n) * 0.05 * _env(n, 0.001, 0.25)

    elif category == "boing":
        sec = seconds if seconds else 0.34
        n = int(sec * SR)
        ph = _sweep(n, 700 * pitch_jitter, 180 * pitch_jitter, curve=1.3)
        s = np.sin(2 * np.pi * ph) * _env(n, 0.003, sec)

    elif category == "coin":
        # two bright partials with a fast ring-down
        sec = seconds if seconds else 0.45
        n = int(sec * SR)
        t = np.arange(n) / SR
        s = (np.sin(2 * np.pi * 1320 * pitch_jitter * t) * 0.6
             + np.sin(2 * np.pi * 1980 * pitch_jitter * t) * 0.3)
        s *= _env(n, 0.001, sec)

    elif category == "click":
        sec = seconds if seconds else 0.06
        n = int(sec * SR)
        s = rng.standard_normal(n) * _env(n, 0.0005, sec)

    else:
        raise ValueError(f"unknown sfx category: {category}")

    # three-tap echo adds space to anything that isn't a UI blip
    if category not in ("click",):
        e = np.zeros(len(s))
        e[::1] = s
        for delay, gain in ((0.055, 0.28), (0.11, 0.14)):
            d = int(delay * SR)
            if d < len(e):
                e[d:] += s[:-d] * gain
        s = e
    return s


# ---------------------------------------------------------------- music
# Pentatonic major. Deliberately narrow: it is very hard to write a wrong note
# in C-major pentatonic, which matters when nobody is listening before publish.
PENTA = [0, 2, 4, 7, 9]


def _note_freq(semitone_from_c4):
    return 261.63 * (2 ** (semitone_from_c4 / 12))


def music_loop(seconds=30.0, bpm=112, seed=7, bright=True):
    """A looping, upbeat kids bed. No drums sample needed - they are synthesised."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    n_bars = max(1, int(seconds / bar))
    total = int(n_bars * bar * SR) + SR
    out = np.zeros(total)

    # ---- bass: root notes, one per bar, walking up the pentatonic
    bass_roots = [0, 0, 5, 7, 0, 0, 5, 7]
    for b in range(n_bars):
        semi = bass_roots[b % len(bass_roots)]
        f = _note_freq(semi - 24)
        i0 = int(b * bar * SR)
        ln = int(bar * SR)
        if i0 + ln > total:
            break
        t = np.arange(ln) / SR
        w = (np.sin(2 * np.pi * f * t) * 0.55
             + np.sin(2 * np.pi * f * 2 * t) * 0.18)
        # pluck envelope so it reads as a bass guitar, not an organ
        w *= np.exp(-t * 2.2)
        out[i0:i0 + ln] += w * 0.32

    # ---- chord stabs on beats 2 and 4
    for b in range(n_bars):
        semi = bass_roots[b % len(bass_roots)]
        for off in (beat, beat * 3):
            i0 = int((b * bar + off) * SR)
            ln = int(beat * 0.9 * SR)
            if i0 + ln > total:
                continue
            t = np.arange(ln) / SR
            chord = np.zeros(ln)
            for iv in (0, 4, 7):
                f = _note_freq(semi + iv)
                chord += np.sin(2 * np.pi * f * t) / 3
            chord *= np.exp(-t * 7)
            out[i0:i0 + ln] += chord * 0.14

    # ---- melody: pentatonic random walk, eighth notes
    step = beat / 2
    n_steps = int(seconds / step)
    deg = 7  # start on the fifth, sounds resolved
    for i in range(n_steps):
        # bias toward stepwise motion, occasional leap
        if rng.random() < 0.72:
            deg += rng.choice([-1, 1])
        else:
            deg += rng.choice([-3, -2, 2, 3])
        deg = max(0, min(14, deg))
        semi = PENTA[deg % 5] + 12 * (deg // 5)
        f = _note_freq(semi)
        i0 = int(i * step * SR)
        ln = int(step * 1.6 * SR)
        if i0 + ln > total:
            continue
        t = np.arange(ln) / SR
        # triangle-ish lead: fundamental + odd harmonic
        lead = np.sin(2 * np.pi * f * t) * 0.7 + np.sin(2 * np.pi * f * 3 * t) * 0.1
        lead *= np.exp(-t * 5.5)
        if bright:
            lead += np.sin(2 * np.pi * f * 2 * t) * 0.12 * np.exp(-t * 6)
        out[i0:i0 + ln] += lead * 0.20

    # ---- synthesised percussion
    for b in range(n_bars):
        for k, (off, kind) in enumerate([(0.0, "kick"), (beat * 0.5, "hat"),
                                         (beat, "kick"), (beat * 1.5, "hat"),
                                         (beat * 2, "snare"), (beat * 2.5, "hat"),
                                         (beat * 3, "kick"), (beat * 3.5, "hat")]):
            i0 = int((b * bar + off) * SR)
            if i0 >= total:
                continue
            if kind == "kick":
                ln = int(0.11 * SR)
                t = np.arange(ln) / SR
                f = 130 * np.exp(-t * 30)
                w = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 22)
                out[i0:i0 + ln] += w * 0.5
            elif kind == "snare":
                ln = int(0.13 * SR)
                w = rng.standard_normal(ln) * np.exp(-np.arange(ln) / SR * 26)
                out[i0:i0 + ln] += w * 0.16
            else:  # hat
                ln = int(0.045 * SR)
                w = rng.standard_normal(ln) * np.exp(-np.arange(ln) / SR * 90)
                out[i0:i0 + ln] += w * 0.09

    # soft-clip so the mix stays musical rather than crunchy
    out = np.tanh(out * 1.1) * 0.85

    # ---- MIX BALANCE -------------------------------------------------
    # A raw additive synth mix comes out top-heavy and thin: on the first
    # build, 51% of energy sat above 4 kHz with only 5% below 80 Hz. On a
    # phone speaker that reads as harsh and weak, so rebalance with a
    # three-band tilt. Kids content is mostly watched on device speakers
    # that cannot reproduce 8 kHz, so that energy is wasted, not just loud.
    out = _tilt(out, low_gain=1.9, mid_gain=1.0, high_gain=0.32,
                low_cut=220.0, high_cut=4200.0)
    # loop-safe: crossfade the tail into the head
    xf = int(0.25 * SR)
    if len(out) > 2 * xf:
        head, tail = out[:xf], out[-xf:]
        ramp = np.linspace(0, 1, xf)
        out[-xf:] = tail * (1 - ramp) + head * ramp
        out = out[:-xf]
    return out


def mix_voice(bed, events, seconds):
    """Overlay SFX events on the music bed.
    events = [(time_in_seconds, category, name, gain), ...]"""
    n = int(seconds * SR)
    track = np.zeros(n)
    if bed is not None:
        m = min(n, len(bed))
        track[:m] = bed[:m]
    for when, cat, name, gain in events:
        s = sfx(cat, name) * gain
        i0 = int(when * SR)
        if i0 >= n:
            continue
        m = min(n - i0, len(s))
        track[i0:i0 + m] += s[:m]
    # gentle limiter
    return np.tanh(track * 1.2) * 0.88


if __name__ == "__main__":
    import sys
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    outdir.mkdir(parents=True, exist_ok=True)

    bed = music_loop(seconds=30.0, bpm=112, seed=7)
    _write(outdir / "music_bed_30s.wav", bed)
    print(f"music bed: 30s -> {outdir/'music_bed_30s.wav'}")

    for cat in ("pop", "sparkle", "whoosh", "tada", "boing", "coin", "click"):
        _write(outdir / f"sfx_{cat}.wav", sfx(cat, "a"))
        print(f"sfx: {cat}")

    ev = [(0.0, "whoosh", "a", 0.5), (1.0, "pop", "a", 0.7), (2.0, "sparkle", "a", 0.6),
          (3.0, "coin", "a", 0.8), (4.0, "tada", "a", 0.9), (5.0, "boing", "a", 0.6)]
    _write(outdir / "mix_demo.wav", mix_voice(bed, ev, 6.0))
    print(f"mix demo: 6s -> {outdir/'mix_demo.wav'}")
