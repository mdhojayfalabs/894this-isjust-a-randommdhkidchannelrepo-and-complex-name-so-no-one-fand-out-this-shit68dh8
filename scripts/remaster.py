#!/usr/bin/env python3
"""Remaster an already-generated Short: lock the framing, fix the exposure.

WHY THIS EXISTS
---------------
The owner rejected the first Short for three specific defects, all confirmed by
measurement rather than opinion:

  1. the character zooms and drifts inside the frame,
  2. there is no background,
  3. so much light behind him that he cannot be made out.

Generation cannot be re-run right now (Pixazo returns 402 Insufficient Balance),
so this recovers what can be recovered in post from the footage that exists.

Two of the three are fixable here and are fixed here:

  * framing - per-frame subject rotoscoping gives a bounding box; a smoothed
    affine is then planned that puts that box in the same place every frame.
    Measured on the original: character height swung 428-640px (36% of mean) and
    vertical centre swung 92px. This removes that without freezing the
    character's own performance.
  * exposure - the generated frames carry a blown-out key behind the subject.
    A tone curve pulls the highlights back in and restores local contrast so the
    character reads against whatever is behind him.

The third - a real background - is NOT fixable from this footage, because the
character fills the frame edge to edge (head at row 0, feet at row 1279, hands
at both margins). There is no room to reveal a scene, and scaling him down to
make room would push an already-soft 704px-tall source below usable detail. That
one needs a regeneration with a properly framed key frame, which is what
scripts/build_short.py now does via bg_plates + composite.

Stdlib + numpy + opencv only.
"""
import cv2
import numpy as np

import composite as comp


# ---------------------------------------------------------------- rotoscope
def roto_sequence(frames, key_bgr, iters=4):
    """Per-frame character mask by GrabCut, seeded from the previous frame.

    Seeding from the previous mask rather than a fixed rectangle is what keeps
    the mask glued to the character while the virtual camera moves.
    """
    kh, kw = key_bgr.shape[:2]
    seed = comp._core_matte(key_bgr)
    ys, xs = np.nonzero(seed > 127)
    sx0, sx1, sy0, sy1 = xs.min(), xs.max(), ys.min(), ys.max()

    masks = []
    prev = None
    for f in frames:
        fh, fw = f.shape[:2]
        # where the character was last frame, grown a little
        if prev is None:
            # The key frame is a tight crop - the character touches all four
            # edges - so a rectangle init leaves GrabCut with no background
            # samples and it aborts. Seed from the key frame's own matte
            # instead, which supplies both classes.
            seed_r = cv2.resize(seed, (fw, fh), interpolation=cv2.INTER_NEAREST)
            core = cv2.erode((seed_r > 127).astype(np.uint8) * 255,
                             np.ones((9, 9), np.uint8))
            band = cv2.dilate((seed_r > 127).astype(np.uint8) * 255,
                              np.ones((21, 21), np.uint8))
            init = np.full((fh, fw), cv2.GC_PR_BGD, np.uint8)
            init[band > 0] = cv2.GC_PR_FGD
            init[core > 0] = cv2.GC_FGD
            bg = np.zeros((1, 65), np.float64)
            fg = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(f, init, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                masks.append((seed_r > 127).astype(np.uint8) * 255)
                prev = masks[-1]
                continue
            cur = np.where((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        else:
            # seed with the previous mask: sure-fg inside, sure-bg outside a
            # dilated band, probable-fg on the boundary
            grown = cv2.dilate(prev, np.ones((25, 25), np.uint8))
            shrunk = cv2.erode(prev, np.ones((11, 11), np.uint8))
            init = np.full(f.shape[:2], cv2.GC_PR_BGD, np.uint8)
            init[grown > 0] = cv2.GC_PR_FGD
            init[shrunk > 0] = cv2.GC_FGD
            init[prev == 0] = cv2.GC_BGD
            bg = np.zeros((1, 65), np.float64)
            fg = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(f, init, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                # not enough samples in a class - keep the previous mask
                masks.append(prev)
                continue
            cur = np.where((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # keep only the largest blob so stray props do not skew the bbox
        n, lab, stats, _ = cv2.connectedComponentsWithStats((cur > 0).astype(np.uint8), 8)
        if n > 1:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[best, cv2.GC_STAT_AREA if False else cv2.CC_STAT_AREA] > 300:
                cur = (lab == best).astype(np.uint8) * 255
        masks.append(cur)
        prev = cur
    return masks


def boxes_from_masks(masks):
    out = []
    for m in masks:
        ys, xs = np.nonzero(m > 127)
        if not len(xs):
            out.append(None)
            continue
        out.append((int(xs.min()), int(ys.min()),
                    int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min())))
    return out


# ---------------------------------------------------------------- lock
def lock_transforms(boxes, target_frac=0.74, target_cy=0.50, smooth=0.35,
                    W=None, H=None, max_gain=1.25):
    """Smoothed affine per frame that holds the subject in one framing.

    The subject is allowed to breathe: the scale correction is damped to
    max_gain so a genuine step toward the camera still reads as a step, it just
    no longer becomes a 40% zoom.
    """
    H = H or 720
    W = W or 1280
    ok = [b for b in boxes if b]
    if not ok:
        return [np.float32([[1, 0, 0], [0, 1, 0]])] * len(boxes)
    rh = float(np.median([b[3] for b in ok]))
    rw = float(np.median([b[2] for b in ok]))
    rcx = float(np.median([b[0] + b[2] / 2 for b in ok]))
    rcy = float(np.median([b[1] + b[3] / 2 for b in ok]))

    base = (target_frac * H) / max(rh, 1)
    raw = []
    for b in boxes:
        if b is None:
            raw.append(None)
            continue
        bw, bh = b[2], b[3]
        cx, cy = b[0] + bw / 2, b[1] + bh / 2
        rel = float(np.clip(rh / max(bh, 1), 1.0 / max_gain, max_gain))
        s = base * rel
        raw.append((W * 0.5 - s * cx, H * target_cy - s * cy, s))

    filled, last = [], (W * 0.5 - base * rcx, H * target_cy - base * rcy, base)
    for r in raw:
        if r is not None:
            last = r
        filled.append(last)
    arr = np.array(filled, float)
    if len(arr) > 2:
        fwd = arr.copy()
        for i in range(1, len(fwd)):
            fwd[i] = smooth * fwd[i - 1] + (1 - smooth) * fwd[i]
        bwd = arr.copy()
        for i in range(len(bwd) - 2, -1, -1):
            bwd[i] = smooth * bwd[i + 1] + (1 - smooth) * bwd[i]
        arr = 0.5 * (fwd + bwd)
    return [np.float32([[s, 0, tx], [0, s, ty]]) for tx, ty, s in arr]


# ---------------------------------------------------------------- grade
def grade(bgr, lift=0.0, gamma=0.94, gain=1.06, sat=1.12, warm=1.03,
          highlight_rolloff=0.55):
    """Pull the blown-out key back in and give the subject local contrast.

    highlight_rolloff compresses everything above ~0.72 so the white void
    behind the character stops reading as a light source.
    """
    a = bgr.astype(np.float32) / 255.0
    # tone curve: lift shadows slightly, compress highlights
    a = np.clip(a, 0, 1)
    hi = a > 0.72
    a[hi] = 0.72 + (a[hi] - 0.72) * (1.0 - highlight_rolloff)
    a = np.clip(a * gain + lift, 0, 1) ** gamma
    # saturation
    g = a.mean(2, keepdims=True)
    a = np.clip(g + (a - g) * sat, 0, 1)
    # warmth: a touch more red, a touch less blue
    a[:, :, 2] = np.clip(a[:, :, 2] * warm, 0, 1)
    a[:, :, 0] = np.clip(a[:, :, 0] / warm, 0, 1)
    # gentle unsharp so the 704px source holds together after all this
    blur = cv2.GaussianBlur(a, (0, 0), 1.1)
    a = np.clip(a + (a - blur) * 0.55, 0, 1)
    return (a * 255).astype(np.uint8)


def upscale(bgr, factor=1.5):
    h, w = bgr.shape[:2]
    return cv2.resize(bgr, (int(w * factor), int(h * factor)),
                      interpolation=cv2.INTER_LANCZOS4)
