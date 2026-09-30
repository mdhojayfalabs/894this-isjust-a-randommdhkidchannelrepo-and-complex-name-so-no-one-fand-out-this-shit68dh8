#!/usr/bin/env python3
"""Video generation providers behind one interface.

WHY THIS EXISTS
---------------
Pixazo worked until the wallet emptied (402 Insufficient Balance), and the
subscription has no image endpoint. Rather than hard-coding one vendor again,
every provider sits behind `generate(key_frame, prompt) -> bytes`.

WHAT THE AUDIT FOUND (September 2026)
-------------------------------------
Free, card-free, scriptable, and with enough GPU are mutually exclusive - every
route fails at least one:

| route | free | no card | scriptable | enough GPU |
|---|---|---|---|---|
| Pixazo | no - wallet | - | yes | yes |
| Gemini API Veo | no | - | yes | yes |
| Gemini API image | **no free tier at all** | - | yes | yes |
| Google Flow (Veo + Nano Banana 2) | yes, 50 credits/day | yes | **no - web UI** | yes |
| HF Inference Providers | $0.10/month credits | yes | yes | ~1 clip/month |
| HF ZeroGPU free | 3.5 min/day H200 | yes | yes | no |
| self-host Wan 2.2 | after hardware | - | yes | yes |
| GitHub Actions | unlimited minutes | yes | yes | **no GPU on runners** |
| this sandbox | - | - | yes | **2 cores, 1 GB RAM, no GPU** |

So the choice is which constraint to drop. Pixazo (or any paid API) drops
"free". Google Flow drops "automatic" - it is free and high quality but a web
UI, so it needs a few minutes of a human per day.

WHAT IS IMPLEMENTED
-------------------
`HuggingFaceProvider` routes through HF Inference Providers to fal-ai. That
reaches Lightricks/LTX-2 (19B) and Wan-AI/Wan2.2-I2V-A14B, both far stronger
than the LTX-Video Pixazo serves. One HF token, no separate provider account.

It is NOT free at volume: HF's free tier is $0.10/month of credits, which is
about one clip. It is implemented anyway because it is the best quality-per-
effort route and the only one where a small spend removes every other problem.
"""
import base64
import json
import time
import urllib.error
import urllib.request

HF_ROUTER = "https://router.huggingface.co"

# Model ids as HF knows them -> the provider task path fal-ai serves.
HF_VIDEO_MODELS = {
    "ltx2": "Lightricks/LTX-2",            # fal-ai/ltx-2-19b/distilled/image-to-video
    "wan22": "Wan-AI/Wan2.2-I2V-A14B",     # fal-ai/wan/v2.2-a14b/image-to-video
    "hunyuan": "tencent/HunyuanVideo-I2V",
    "minimax": "MiniMaxAI/MiniMax-H3",
}


class HuggingFaceProvider:
    """Route image-to-video through HF Inference Providers.

    One token, no provider account. Monthly credits apply; past them it is
    pay-as-you-go at the provider's own rate with no HF markup.
    """

    name = "huggingface"

    def __init__(self, token, model="ltx2", provider="fal-ai", timeout=900):
        self.token = token
        self.model = HF_VIDEO_MODELS.get(model, model)
        self.provider = provider
        self.timeout = timeout

    def _post(self, payload):
        d = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{HF_ROUTER}/hf-inference/models/{self.model}",
            data=d, method="POST",
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json",
                     "Accept": "application/json",
                     "X-Provider": self.provider})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def generate(self, key_frame_bgr, prompt, **_):
        """Return raw video bytes, or raise."""
        import cv2
        ok, buf = cv2.imencode(".png", key_frame_bgr)
        if not ok:
            raise RuntimeError("could not encode the key frame")
        st, body = self._post({
            "image": "data:image/png;base64,"
                     + base64.b64encode(buf.tobytes()).decode(),
            "prompt": prompt,
        })
        if st != 200:
            raise RuntimeError(f"HF {self.provider} returned {st}: {body[:200]!r}")
        # providers answer with either raw video bytes or a JSON envelope
        ctype = ""
        try:
            ctype = json.loads(body).get("content_type", "")
        except Exception:
            pass
        if body[:4] == b"\x1a\x45\xdf\xa3" or "video" in ctype:
            return body
        try:
            j = json.loads(body)
            for k in ("video", "output", "url"):
                if k in j:
                    u = j[k]
                    if isinstance(u, list):
                        u = u[0]
                    if isinstance(u, str) and u.startswith("http"):
                        return self._download(u)
        except Exception:
            pass
        return body

    @staticmethod
    def _download(url):
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        subprocess.run(["curl", "-sL", "-A", "Mozilla/5.0", "-o", tmp, url],
                       check=True)
        return open(tmp, "rb").read()

    def wait(self, *_a, **_k):
        """Interface compatibility with the Pixazo client."""
        return None


class StillsProvider:
    """No video model at all: build motion from stills with ffmpeg + numpy.

    This is the only route that is free, card-free, scriptable AND has enough
    compute, because it needs no GPU - it runs in this sandbox on 2 cores.

    It is a real format for this content, not a fallback apology. The pacing
    research says a Short wants a visual change every 2-4 seconds and CoComelon
    cuts every 1-3 seconds, so a sequence of composed stills with hard cuts,
    Ken Burns motion and procedural parallax is inside the format's norms. What
    it cannot do is generate new imagery, so every shot is a recomposition of
    frames we already have.
    """

    name = "stills"

    def __init__(self, **_kw):
        pass

    def generate(self, key_frame_bgr, prompt, **_):  # pragma: no cover
        raise NotImplementedError(
            "StillsProvider does not call a model; use build_short_stills.py")


PROVIDERS = {
    "pixazo": None,          # filled in by build_short to avoid a cycle
    "huggingface": HuggingFaceProvider,
    "stills": StillsProvider,
}


def make(name, **kw):
    cls = PROVIDERS.get(name)
    if cls is None:
        raise SystemExit(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return cls(**kw)
