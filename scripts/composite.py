#!/usr/bin/env python3
"""Put the character into a real scene, and take him back out again.

WHY THIS EXISTS
---------------
The character key frame is a clean studio shot on white. Handing that to the
video model asks it to invent the world around him, and it invents a different
one every clip - which is the "nothing behind him / too much light behind him"
defect. Compositing him onto a drawn plate first means the model receives a
complete, correctly lit scene and has structure to hold onto.

The second half matters just as much. Because the plate is known, the
character's matte in the generated clip is recoverable by differencing against
the plate: |generated - plate| is large exactly where the character is. That
gives a far more reliable subject mask than colour thresholding, which on this
material matched 99% of the frame and produced garbage tracking data.
"""
import cv2
import numpy as np

import bg_plates


def _core_matte(key_bgr, thresh=232):
    """Hard (unfeathered) character mask from the white key frame."""
    hsv = cv2.cvtColor(key_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    bg = (v > thresh) & (s < 26)
    m = (~bg).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=4)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == best).astype(np.uint8) * 255
    # Fill every interior hole by filling the OUTER contour solid. Flood-filling
    # from the border does not work here: the character has white parts (socks,
    # shoe highlights, glasses) that connect to the white studio background, so
    # the "hole" reaches the edge and is never filled - which left a white block
    # standing behind his legs in the composite.
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        solid = np.zeros_like(m)
        cv2.drawContours(solid, cnts, -1, 255, thickness=-1)
        m = solid
    return m


def character_matte(key_bgr, thresh=232, feather=9):
    """Feathered matte. For geometry use core_bbox: the feather bleeds stray
    pixels to the frame edge, which makes a bbox taken from the blurred mask
    span the entire image and breaks any GrabCut initialised from it."""
    m = _core_matte(key_bgr, thresh)
    if feather:
        m = cv2.GaussianBlur(m, (feather * 2 + 1, feather * 2 + 1), 0)
    return m


def core_bbox(key_bgr, thresh=232, pad=0):
    """Bounding box (x, y, w, h) of the character, from a clean mask."""
    m = _core_matte(key_bgr, thresh)
    ys, xs = np.nonzero(m > 127)
    if not len(xs):
        return None
    h, w = m.shape
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
    return (x0, y0, x1 - x0, y1 - y0)


def composite(key_bgr, plate_bgr, target_h_frac=0.62, cx_frac=0.50,
              cy_frac=0.50, shade=0.16, scale=1.0, backdrop=0.30):
    """Place the character on the plate, preserving his aspect ratio.

    The key frame is portrait (9:16) and the plate is landscape, so the
    character is scaled to a fraction of the plate's HEIGHT and centred - never
    stretched to the plate's box, which would squash him.

    Returns (image, matte, box) where box is the character's (x, y, w, h) in the
    output. The box comes from the matte, not from re-masking the composite:
    re-masking cannot work because the drawn plate is full of colour, so
    "bright and neutral" stops separating subject from background and the mask
    swallows the whole frame.
    """
    ph, pw = plate_bgr.shape[:2]
    kh, kw = key_bgr.shape[:2]

    m = character_matte(key_bgr)
    ys, xs = np.nonzero(m > 127)
    if not len(xs):
        raise RuntimeError("no character found in the key frame")
    kx0, ky0 = int(xs.min()), int(ys.min())
    kx1, ky1 = int(xs.max()), int(ys.max())
    ch, cw = ky1 - ky0, kx1 - kx0

    # target height in the output, aspect preserved
    want_h = int(target_h_frac * ph * scale)
    f = want_h / max(ch, 1)
    new_w, new_h = max(1, int(cw * f)), max(1, int(ch * f))
    if new_w > pw * 0.92:                       # keep him inside the frame
        f = (pw * 0.92) / max(cw, 1)
        new_w, new_h = int(cw * f), int(ch * f)

    sub = key_bgr[ky0:ky1 + 1, kx0:kx1 + 1]
    subm = m[ky0:ky1 + 1, kx0:kx1 + 1]
    sub = cv2.resize(sub, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    subm = cv2.resize(subm, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    a = (subm.astype(np.float32) / 255.0)[:, :, None]

    x0 = int(cx_frac * pw - new_w / 2.0)
    y0 = int(cy_frac * ph - new_h / 2.0)
    x0 = int(np.clip(x0, 0, pw - new_w))
    y0 = int(np.clip(y0, 0, ph - new_h))

    # Separation: darken the room just behind where the character will stand, so
    # he reads against it. This MUST happen before compositing - applied after,
    # it darkens the character too and the surround ends up brighter than he is.
    # Without it the drawn plate and the bright character average to nearly the
    # same value and he dissolves into the background, which is the "so much
    # light behind him you can't make him out" defect all over again.
    plate = plate_bgr.astype(np.float32).copy()
    if backdrop:
        yy, xx = np.mgrid[0:ph, 0:pw]
        bx = x0 + new_w / 2.0
        by = y0 + new_h / 2.0
        d = np.sqrt(((xx - bx) / (new_w * 1.15)) ** 2
                    + ((yy - by) / (new_h * 0.70)) ** 2)
        behind = np.clip(1.0 - d, 0.0, 1.0) ** 1.4 * backdrop
        plate *= (1.0 - behind)[:, :, None]

    out = plate
    roi = out[y0:y0 + new_h, x0:x0 + new_w]
    out[y0:y0 + new_h, x0:x0 + new_w] = roi * (1 - a) + sub.astype(np.float32) * a

    full_m = np.zeros((ph, pw), np.uint8)
    full_m[y0:y0 + new_h, x0:x0 + new_w] = subm

    # contact shadow so the feet are grounded rather than pasted on
    band = max(6, int(new_h * 0.05))
    foot = np.zeros((ph, pw), np.float32)
    fy1 = min(ph, y0 + new_h + band)
    foot[max(0, y0 + new_h - band):fy1, x0:x0 + new_w] = 1.0
    foot = cv2.GaussianBlur(foot, (0, 0), band * 1.8)
    out *= (1.0 - shade * foot)[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8), full_m, (x0, y0, new_w, new_h)


def scene(scene_name, key_bgr, seed=7, **kw):
    """One-call: build a plate and put the character in it."""
    p = bg_plates.plate(scene_name, seed)
    return composite(key_bgr, p, **kw)


def diff_matte(frame_bgr, plate_bgr, thresh=26, feather=7):
    """Recover the character mask from a generated frame by differencing.

    Works because the plate is known. Anywhere the generated frame disagrees
    with the plate by more than `thresh` is character.
    """
    if frame_bgr.shape[:2] != plate_bgr.shape[:2]:
        frame_bgr = cv2.resize(frame_bgr, (plate_bgr.shape[1], plate_bgr.shape[0]))
    d = np.abs(frame_bgr.astype(np.int16) - plate_bgr.astype(np.int16)).sum(2)
    m = (d > thresh * 3).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[best, cv2.CC_STAT_AREA] > 400:
            m = (lab == best).astype(np.uint8) * 255
    if feather:
        m = cv2.GaussianBlur(m, (feather * 2 + 1, feather * 2 + 1), 0)
    return m


if __name__ == "__main__":
    import sys
    key = cv2.imread("assets/hoji_key_720.png")
    for name in bg_plates.SCENES:
        img, m, _ = scene(name, key)
        cv2.imwrite(f"/tmp/comp_{name}.png", img)
        print(f"{name}: composite {img.shape} matte coverage "
              f"{(m > 8).mean() * 100:.1f}%")
