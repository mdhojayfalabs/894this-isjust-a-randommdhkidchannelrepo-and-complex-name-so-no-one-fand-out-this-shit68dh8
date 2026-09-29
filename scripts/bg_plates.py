#!/usr/bin/env python3
"""Procedural background plates, landscape-first.

WHY THIS EXISTS
---------------
Feeding the video model a character floating on a plain background produced the
three defects the owner rejected:

  * "nothing behind him" - with no scene to hold, the model invented a different
    one per clip (white void, then orange),
  * "so much light behind that you can't make him out" - no lighting described,
    so it defaulted to a blown-out key behind the subject,
  * the framing jumping - with no structure in frame the virtual camera had
    nothing to anchor to.

There is no image endpoint on the Pixazo subscription (probed: every
text-to-image path 404s), so the plate is drawn here. That is not a compromise:
a drawn plate is perfectly stable across every clip, carries zero generation
flicker, and lets the lighting be set so the subject reads against it. For a
flat, graphic kids-cartoon look that is the better tool anyway.

LANDSCAPE ON PURPOSE
--------------------
The model emits 1280x704 landscape no matter what is asked of it, so the key
frame handed to it must be landscape too. Composing in portrait and cropping
afterwards would throw away most of the drawn room.
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# The model's native output. Everything is composed for this frame.
W, H = 1280, 704


# ---------------------------------------------------------------- helpers
def _vgrad(w, h, top, bottom):
    t = np.linspace(0.0, 1.0, h)[:, None]
    a = np.array(top, float)[None, :]
    b = np.array(bottom, float)[None, :]
    return np.repeat((a * (1 - t) + b * t)[:, None, :], w, axis=1)


def _radial_glow(arr, cx, cy, radius, colour, strength=1.0):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / max(radius, 1)) ** 2 + ((yy - cy) / max(radius, 1)) ** 2)
    f = np.clip(1.0 - d, 0.0, 1.0) ** 2 * strength
    return np.clip(arr + f[:, :, None] * np.array(colour, float)[None, None, :], 0, 255)


def _soft_shadow(arr, cx, cy, rx, ry, strength=0.35):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    f = np.clip(1.0 - d, 0.0, 1.0) ** 2
    return arr * (1.0 - strength * f[:, :, None])


def _noise(h, w, seed, octaves=4):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        sh, sw = max(2, h >> (o + 3)), max(2, w >> (o + 3))
        up = np.array(Image.fromarray((rng.random((sh, sw)) * 255).astype(np.uint8))
                      .resize((w, h), Image.BICUBIC), float) / 255.0
        out += up * amp
        tot += amp
        amp *= 0.5
    return out / tot


def _canvas(top, bottom, seed, tex=0.05):
    a = _vgrad(W, H, top, bottom)
    a = a + (_noise(H, W, seed) - 0.5)[:, :, None] * 255 * tex
    return a


# ---------------------------------------------------------------- scenes
def playroom(seed=7):
    """Bright playroom: pastel wall, wooden floor, toy shelf, sunlit window."""
    rng = np.random.default_rng(seed)
    floor_y = int(H * 0.74)
    arr = np.vstack([_canvas((214, 186, 216), (188, 152, 190), seed + 1)[:floor_y],
                     _canvas((150, 104, 74), (112, 74, 52), seed + 2, 0.12)[floor_y:]])

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img, "RGBA")

    # floor planks, converging for a hint of perspective
    for i in range(-3, 14):
        y = floor_y + int((max(i, 0) / 10.0) ** 2 * (H - floor_y) * 1.2)
        if y < H:
            dr.line([(0, y), (W, y)], fill=(152, 106, 70, 80), width=2)
    for i in range(-5, 10):
        x = int(W * 0.5 + i * W * 0.13)
        dr.line([(x, floor_y), (int(W * 0.5 + i * W * 0.45), H)],
                fill=(152, 106, 70, 70), width=2)

    # window, right third: warm daylight pooling in
    wx, wy = int(W * 0.66), int(H * 0.10)
    ww, wh = int(W * 0.24), int(H * 0.42)
    dr.rounded_rectangle([wx, wy, wx + ww, wy + wh], radius=22,
                         fill=(176, 214, 232, 255), outline=(96, 150, 178, 255), width=9)
    dr.line([(wx + ww // 2, wy), (wx + ww // 2, wy + wh)], fill=(96, 150, 178, 255), width=7)
    dr.line([(wx, wy + wh // 2), (wx + ww, wy + wh // 2)], fill=(234, 199, 152, 255), width=7)

    # toy shelf, left third
    sy = int(H * 0.34)
    dr.rounded_rectangle([int(W * 0.05), sy, int(W * 0.30), sy + 14], radius=7,
                         fill=(178, 130, 90, 255))
    dr.rectangle([int(W * 0.065), sy + 14, int(W * 0.09), sy + int(H * 0.20)],
                 fill=(152, 108, 74, 255))
    dr.rectangle([int(W * 0.26), sy + 14, int(W * 0.285), sy + int(H * 0.20)],
                 fill=(152, 108, 74, 255))
    cols = [(240, 128, 84), (110, 190, 210), (250, 205, 90),
            (150, 205, 130), (215, 140, 200)]
    for i, c in enumerate(cols):
        bx = int(W * 0.075) + i * int(W * 0.042)
        dr.rounded_rectangle([bx, sy - 54, bx + 40, sy - 4], radius=8, fill=c + (255,))
    dr.ellipse([int(W * 0.10), sy + 24, int(W * 0.10) + 54, sy + 78],
               fill=(238, 118, 96, 255))
    dr.ellipse([int(W * 0.115), sy + 32, int(W * 0.115) + 17, sy + 49],
               fill=(255, 192, 172, 210))
    for r in range(3):
        for c in range(3 - r):
            px = int(W * 0.185) + c * 27 + r * 13
            py = sy + 72 - r * 23
            dr.polygon([(px, py), (px + 25, py), (px + 12, py - 21)],
                       fill=cols[(r * 3 + c) % 5] + (255,))

    # a rug so the floor is not bare
    dr.ellipse([int(W * 0.30), int(H * 0.80), int(W * 0.78), int(H * 1.02)],
               fill=(150, 96, 140, 150))
    dr.ellipse([int(W * 0.36), int(H * 0.84), int(W * 0.72), int(H * 0.99)],
               fill=(186, 132, 172, 150))

    arr = np.asarray(img, float)
    arr = _radial_glow(arr, wx + ww // 2, wy + wh // 2, ww * 1.7, (34, 25, 11), 0.5)
    arr = _soft_shadow(arr, W * 0.5, H * 0.88, W * 0.30, H * 0.055, 0.26)
    arr = _radial_glow(arr, W * 0.5, H * 0.45, W * 1.15, (-24, -20, -18), 0.28)
    return np.clip(arr, 0, 255).astype(np.uint8)


def kitchen(seed=11):
    """Sunny kitchen: tiled splashback, counter, fruit bowl, pendant lamp."""
    floor_y = int(H * 0.70)
    wall = _canvas((150, 186, 196), (110, 148, 162), seed)[:floor_y]
    arr = np.vstack([wall, _canvas((140, 100, 72), (104, 72, 50), seed + 3, 0.11)[floor_y:]])

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img, "RGBA")
    for y in range(0, floor_y, 62):
        for x in range(0, W, 62):
            dr.rounded_rectangle([x + 4, y + 4, x + 58, y + 58], radius=9,
                                 outline=(255, 255, 255, 140), width=4)
    cy = int(H * 0.66)
    dr.rectangle([0, cy, W, cy + 22], fill=(150, 140, 122, 255))
    for i in range(-2, 11):
        x = int(W * 0.5 + i * W * 0.14)
        dr.line([(x, cy + 22), (int(W * 0.5 + i * W * 0.42), H)],
                fill=(168, 134, 96, 110), width=3)
    dr.ellipse([int(W * 0.60), cy - 42, int(W * 0.60) + 150, cy + 12],
               fill=(228, 178, 122, 255))
    for i, c in enumerate([(242, 176, 76), (238, 108, 92), (246, 214, 120),
                           (214, 232, 140)]):
        bx = int(W * 0.615) + i * 33
        dr.ellipse([bx, cy - 72, bx + 42, cy - 30], fill=c + (255,))
    dr.line([(int(W * 0.28), 0), (int(W * 0.28), int(H * 0.16))],
            fill=(92, 92, 98, 255), width=6)
    dr.polygon([(int(W * 0.28) - 70, int(H * 0.26)), (int(W * 0.28) + 70, int(H * 0.26)),
                (int(W * 0.28) + 38, int(H * 0.16)), (int(W * 0.28) - 38, int(H * 0.16))],
               fill=(251, 228, 152, 255))
    arr = np.asarray(img, float)
    arr = _radial_glow(arr, W * 0.28, H * 0.22, W * 0.95, (38, 30, 13), 0.46)
    arr = _soft_shadow(arr, W * 0.5, H * 0.80, W * 0.28, H * 0.05, 0.24)
    return np.clip(arr, 0, 255).astype(np.uint8)


def garden(seed=23):
    """Backyard: sky, hedge, grass, bunting, flower bed."""
    gy = int(H * 0.62)
    arr = np.vstack([_canvas((84, 148, 202), (128, 182, 220), seed)[:gy],
                     _canvas((76, 132, 66), (48, 96, 48), seed + 4, 0.14)[gy:]])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rounded_rectangle([-20, gy - 78, W + 20, gy + 18], radius=52,
                         fill=(58, 104, 54, 255))
    rng = np.random.default_rng(seed)
    for _ in range(150):
        x = int(rng.integers(0, W)); y = int(rng.integers(gy - 74, gy + 10))
        r = int(rng.integers(9, 24))
        dr.ellipse([x - r, y - r, x + r, y + r],
                   fill=(122 + int(rng.integers(0, 40)), 178 + int(rng.integers(0, 40)),
                         94 + int(rng.integers(0, 30)), 255))
    cols = [(242, 128, 100), (250, 206, 96), (128, 196, 214), (176, 148, 214)]
    pts = [(int(W * i / 8.0), int(H * 0.14 + 34 * math.sin(math.pi * i / 8.0)))
           for i in range(9)]
    dr.line(pts, fill=(255, 255, 255, 220), width=5)
    for i in range(8):
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2 + 22
        dr.polygon([(x0 + 4, y0), (x1 - 4, y1), (mx, my)], fill=cols[i % 4] + (255,))
    for i in range(30):
        x = int(i / 29.0 * W + rng.integers(-12, 12)); y = int(H * 0.93 + rng.integers(-20, 20))
        dr.ellipse([x - 11, y - 11, x + 11, y + 11], fill=(250, 214, 96, 255))
        dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(250, 160, 80, 255))
    arr = np.asarray(img, float)
    arr = _radial_glow(arr, W * 0.5, H * 0.16, W * 1.2, (24, 18, 5), 0.30)
    arr = _soft_shadow(arr, W * 0.5, H * 0.84, W * 0.26, H * 0.05, 0.22)
    return np.clip(arr, 0, 255).astype(np.uint8)


SCENES = {"playroom": playroom, "kitchen": kitchen, "garden": garden}


def plate(scene, seed=7, w=W, h=H):
    """Build one plate. Unknown names fall back to the playroom.

    Returns BGR, because everything downstream is OpenCV. The scenes are
    composed in RGB (PIL needs that for drawing), so the channels are swapped
    here. Without this the plate's red and blue trade places and a brown wooden
    floor renders blue - which is exactly what happened before this was caught.
    """
    fn = SCENES.get(scene, playroom)
    a = fn(seed)
    if (a.shape[1], a.shape[0]) != (w, h):
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    return a[:, :, ::-1]


if __name__ == "__main__":
    import cv2
    for name, fn in SCENES.items():
        a = fn()
        cv2.imwrite(f"/tmp/plate_{name}.png", a)
        print(f"{name}: {a.shape[1]}x{a.shape[0]} mean={a.mean():.0f} "
              f"min={a.min()} max={a.max()}")
