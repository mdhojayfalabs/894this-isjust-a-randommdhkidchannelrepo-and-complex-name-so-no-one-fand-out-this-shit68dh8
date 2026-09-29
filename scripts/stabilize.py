#!/usr/bin/env python3
"""Subject lock: stop the character swimming around inside the frame.

WHY THIS EXISTS
---------------
LTX image-to-video on the Pixazo gateway cannot be told to hold the camera.
Verified by A/B test: the parameters `end_image`, `camera_motion`,
`negative_prompt` and `seed` are all accepted with HTTP 202 and then silently
ignored. Over a 5-second clip the character's on-screen width shrank by 33-47%
as the virtual camera pulled back, and the framing jumped between clips from
"torso fills the frame" to "full body, small". That is the "zoom tum" the owner
rejected.

Prompts do not fix it either. The documented behaviour is that with no camera
instruction the model defaults to handheld drift, and re-describing the scene in
an image-to-video prompt makes the drift worse, not better. So the camera is
uncontrollable at generation time and must be corrected afterwards.

WHAT IT DOES
------------
Per frame:
  1. find the character's bounding box (the only saturated, non-white region)
  2. compute the affine transform that would put that box where it belongs
  3. smooth that transform over time so the correction itself never jitters
Then re-render every frame through the smoothed transform.

The result is a subject that holds a constant scale and position while still
moving naturally inside it. It does not freeze the character - only the camera.

Normalising to one target framing also makes the cut between two clips
invisible, because clip N+1 starts from the same framing clip N ended on.
"""
import cv2
import numpy as np


def _character_mask(bgr):
    """Boolean mask of the character: saturated and not blown-out white.

    The generator emits the character over a near-white void, so "coloured and
    not white" separates subject from background cleanly. Deliberately generous
    - a slightly loose mask still gives a correct bounding box, whereas a tight
    one can clip the character's extremities and bias the transform.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    coloured = (s > 28) & (v > 55)
    not_white = ~((v > 238) & (s < 14))
    m = (coloured & not_white).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    # keep the largest blob only: props and sparkles are small and stray
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    # ignore label 0 (background); pick by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if areas.max() < 200:
        return None
    return (lab == best).astype(np.uint8) * 255


def character_box(bgr):
    """Bounding box (x, y, w, h) of the character, or None."""
    m = _character_mask(bgr)
    if m is None:
        return None
    x, y, w, h = cv2.boundingRect(m)
    if w < 8 or h < 8:
        return None
    return x, y, w, h


def track(frames):
    """Per-frame character boxes. `frames` is a list of BGR arrays."""
    boxes = []
    for f in frames:
        boxes.append(character_box(f))
    return boxes


def _median_box(boxes):
    ok = [b for b in boxes if b]
    if not ok:
        return None
    return tuple(int(np.median([b[i] for b in ok])) for i in range(4))


def plan_transforms(frames, target_frac=0.62, target_cy=0.46,
                    smooth=0.35, W=None, H=None):
    """Build a per-frame affine matrix that locks the subject.

    target_frac  - the character's height as a fraction of frame height
    target_cy    - the character's vertical centre as a fraction of height
    smooth       - temporal smoothing of the correction, 0 = none, 1 = frozen
    """
    boxes = track(frames)
    H = H or frames[0].shape[0]
    W = W or frames[0].shape[1]
    ref = _median_box(boxes)
    if ref is None:
        return [np.float32([[1, 0, 0], [0, 1, 0]])] * len(frames), boxes

    rx, ry, rw, rh = ref
    # scale so the median character height becomes target_frac of the frame
    want_h = target_frac * H
    s = want_h / max(rh, 1)
    # cap it: never blow the character past 1.15x, never shrink below 0.75x
    s = float(np.clip(s, 0.75, 1.15))

    # where the median box centre sits after that scale
    rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
    tx = W * 0.5 - s * rcx
    ty = H * target_cy - s * rcy

    raw = []
    for b in boxes:
        if b is None:
            raw.append(None)
            continue
        bx, by, bw, bh = b
        # this frame's character, scaled by the same s, must land on target
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        # keep the *scale* correction relative to the median so a genuinely
        # closer/further subject still reads as closer/further, just damped
        rel = np.clip(rh / max(bh, 1), 0.85, 1.18)
        ss = s * rel
        raw.append((W * 0.5 - ss * cx, H * target_cy - ss * cy, ss))

    # forward-fill missing, then smooth
    filled = []
    last = (tx, ty, s)
    for r in raw:
        if r is not None:
            last = r
        filled.append(last)
    arr = np.array(filled, dtype=np.float64)
    if len(arr) > 2:
        # exponential smoothing in both directions so the correction ramps in
        # and out instead of snapping on the first frame
        a = smooth
        fwd = arr.copy()
        for i in range(1, len(fwd)):
            fwd[i] = a * fwd[i - 1] + (1 - a) * fwd[i]
        bwd = arr.copy()
        for i in range(len(bwd) - 2, -1, -1):
            bwd[i] = a * bwd[i + 1] + (1 - a) * bwd[i]
        arr = 0.5 * (fwd + bwd)

    mats = []
    for tx_, ty_, ss in arr:
        mats.append(np.float32([[ss, 0, tx_], [0, ss, ty_]]))
    return mats, boxes


def apply(frames, mats, W=None, H=None):
    """Re-render frames through the planned transforms."""
    out = []
    H = H or frames[0].shape[0]
    W = W or frames[0].shape[1]
    for f, m in zip(frames, mats):
        out.append(cv2.warpAffine(f, m, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE))
    return out


def lock(frames, **kw):
    """Convenience: plan and apply in one call. Returns (frames, boxes)."""
    mats, boxes = plan_transforms(frames, **kw)
    return apply(frames, mats), boxes
