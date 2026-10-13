#!/usr/bin/env python3
"""
HunyuanVideo-1.5 8.3B image-to-video clip renderer on Kaggle.

THE FIX THAT MATTERS (v24, 2026-10-11)
--------------------------------------
v16 through v23 all loaded the transformer from `tencent/HunyuanVideo-1.5`,
which is the NATIVE Tencent format. Its weights are named
`double_blocks.0.txt_attn_v.bias`, `double_blocks.23.img_mod.linear.bias`, ...
because HunyuanVideo-1.5 uses a 54-block DUAL-STREAM MMDiT that keeps the image
and text tokens in separate streams.

diffusers 0.36+ models that model as `HunyuanVideo15Transformer3DModel` with a
SINGLE-STREAM joint layout: `transformer_blocks.32.attn.to_q.bias`,
`transformer_blocks.21.norm1_context.linear.bias`, `attn.add_q_proj.*`.

The two are not the same architecture. Forcing the 0.35 config through a
0.36+ class name produced a model whose blocks were almost entirely randomly
initialised (diffusers logged hundreds of "newly initialized" keys), and the
first forward pass died at

    transformer_hunyuan_video15.py:153  hidden_states = self.proj(hidden_states)
    RuntimeError: Given groups=1, weight of size [2048, 32, 1, 1, 1],
    expected input[1, 65, 31, 53, 30] to have 32 channels, but got 65 instead

65 = 32 latents + 32 image condition + 1 mask, i.e. exactly what
`concat_condition: true` means - but the checkpoint's x_embedder was built for
32 because the random init came from the wrong schema.

The correct source is the DIFFUSERS-FORMAT mirror, which is a complete,
self-contained pipeline repo:

    hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v

It ships transformer/, vae/, text_encoder/, text_encoder_2/, tokenizer/,
tokenizer_2/, image_encoder/, feature_extractor/, guider/, scheduler/ and
model_index.json - no manual component assembly, no config translation, and
no separate Qwen2.5-VL / byt5-small / SigLIP downloads.

Reference: huggingface.co/docs/diffusers/api/pipelines/hunyuan_video15
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
        torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

Also carried forward from v22/v23, which DID fix a real and separate bug:

    huggingface/accelerate#3542, closed as completed 2026-03-02 (PR #3941)
    "out_proj of torch.nn.MultiheadAttention is a MODULE, not a parameter.
     Submodules won't be loaded by AlignDevicesHook.pre_forward if
     self.place_submodules is False. cpu_offload needs the
     preload_module_classes argument to set place_submodules to True."

    MHA.forward() never calls out_proj.forward() - it reaches into
    out_proj.weight / out_proj.bias directly - so the child hook never fires
    and those two tensors stay on meta. Every offload call below therefore
    passes preload_module_classes=["MultiheadAttention"].
"""

import gc
import os as _os

# Must be set BEFORE torch touches CUDA. In v22 this was done with
# os.environ.setdefault() *after* import torch, so it never took effect and the
# run ended with "738.90 MiB is reserved by PyTorch but unallocated" - pure
# fragmentation that expandable_segments would have absorbed.
_os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import torch


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


T0 = time.time()


def run(cmd):
    log("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        log(r.stdout.strip()[-2000:])
    if r.returncode != 0 and r.stderr:
        log("STDERR: " + r.stderr.strip()[-2000:])
    return r


def report_env():
    log(f"python {sys.version.split()[0]}")
    try:
        import torch

        log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    except Exception as e:
        log(f"torch import failed: {e}")
    # report_accelerator() crashed an earlier run with FileNotFoundError:
    # 'nvidia-smi' - guard it.
    if shutil.which("nvidia-smi"):
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            log(f"GPU: {line.strip()}")
    else:
        log("nvidia-smi not found")


def find_input(name_part, must_end=()):
    for root in (Path("/kaggle/input"), Path("/kaggle/usr")):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if name_part not in p.name:
                continue
            if must_end and not p.name.endswith(must_end):
                continue
            return p
    return None


def download_model(repo_id, cache_dir):
    """Snapshot-download a whole HF repo into /tmp, return the snapshot path."""
    from huggingface_hub import snapshot_download

    log(f"downloading {repo_id} -> {cache_dir}")
    t0 = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        # Only what the i2v pipeline needs. The native-format variants
        # (480p_t2v, 1080p_sr_distilled, ...) are 100+ GB of dead weight.
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "transformer/*",
            "vae/*",
            "text_encoder/*",
            "text_encoder_2/*",
            "tokenizer/*",
            "tokenizer_2/*",
            "image_encoder/*",
            "feature_extractor/*",
            "guider/*",
            "scheduler/*",
        ],
    )
    log(f"download complete in {(time.time() - t0) / 60:.1f} min: {path}")
    return path


def tree(path, max_lines=60):
    p = Path(path)
    if not p.exists():
        log(f"TREE of {p} (missing)")
        return
    log(f"TREE of {p}")
    n = 0
    for f in sorted(p.rglob("*")):
        if f.is_file():
            try:
                sz = f.stat().st_size / 1e6
            except OSError:
                sz = 0.0
            log(f"  {sz:10.1f} MB  {f.relative_to(p)}")
            n += 1
            if n >= max_lines:
                log(f"  ... ({max_lines} shown)")
                break


MOTION_PROMPT = (
    "A 3D animated children's cartoon scene, Pixar-style rendering, two young "
    "Muslim children standing together in a cosy warmly lit bedroom. On the left "
    "a small girl in a bright coral pink dress with a cream hijab; on the right a "
    "small boy in a sky-blue shirt and dark trousers. Behind them a large bed with "
    "a soft teal blanket, a wooden shelf holding colourful stacked books, a round "
    "wall clock, a potted green plant and a small desk lamp casting warm light. "
    "Saturated cheerful colours, clean soft shadows, no harsh backlight. The camera "
    "slowly pushes in a gentle dolly movement. Both children breathe naturally, "
    "blink, and shift their weight slightly; the boy turns his head toward the girl "
    "and she looks up at him. Subtle ambient motion in the blanket and plant leaves. "
    "Continuous stable framing, no zoom jitter, no flicker."
)

NEGATIVE_PROMPT = (
    "overexposed, blown out highlights, washed out background, bright white "
    "background, static, no motion, frozen, blurred, low quality, worst quality, "
    "JPEG compression artifacts, ugly, deformed, disfigured, bad anatomy, extra "
    "limbs, missing limbs, fused fingers, too many fingers, poorly drawn hands, "
    "poorly drawn face, mutated, watermark, signature, text, subtitles, caption, "
    "frame border, camera shake, zoom jitter, scale jitter, flickering, "
    "inconsistent character identity, morphing, duplicate characters, "
    "many people in the background, walking backwards"
)

# v24 (720p_i2v) got all the way into the denoise loop and then died on
# activation memory, not weights:
#
#   transformer_hunyuan_video15.py:483   attn_output = self.attn(...)
#   attention_processor.py:607            self.processor(...)
#   transformer_hunyuan_video15.py:79     apply_rotary_emb(...)
#   embeddings.py:1232                    out = (x.float()*cos + ...)
#   torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 862.00 MiB.
#   GPU 0 has a total capacity of 14.56 GiB of which 404.81 MiB is free.
#
# The architecture was right - that was the repo fix - and only ~862 MiB was
# missing out of 14 GB. Activation size scales with token count, and 480p has
# ~2.3x fewer spatial tokens than 720p. Same 8.3B weights, same model, just a
# smaller working resolution. The demo clip is upscaled to 1080x1920 at mux.
REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"


# ----------------------------------------------------------------------
# Chunked SDPA.
#
# v31 settled the attention question once and for all. On this T4 (sm75) at the
# real sequence length N=33,390:
#
#   ALL (default)    -> CUDA OOM, tried to allocate 66.45 GiB
#   FLASH only       -> "No available kernel"
#   EFFICIENT only   -> "No available kernel"
#   CUDNN only       -> "No available kernel"
#   FLASH+EFFICIENT  -> "No available kernel"
#   xformers         -> wants attn_bias shaped [1, 16, N, N], i.e. per head,
#                       which at N=33,390 is 17.8 GB - worse than the mask
#
# 66.45 GiB is exactly H * N^2 * 4 with H=16: the MATH backend materialises the
# whole score matrix. No fused kernel is reachable on Turing at this length.
#
# So chunk the QUERY instead. The score matrix becomes H * C * N for a chunk of
# C queries, and with C=1024 that is 16 * 1024 * 33,390 * 4 = 2.03 GiB, which
# fits. The attn_mask rows are sliced to match, which is exact - this is the
# same arithmetic, just not all at once.
# ----------------------------------------------------------------------
SDPA_CHUNK = 512
_orig_sdpa = None


def _install_chunked_sdpa(chunk=SDPA_CHUNK):
    global _orig_sdpa
    if _orig_sdpa is not None:
        return _orig_sdpa
    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    def chunked(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False):
        n = query.shape[-2]
        if attn_mask is None or n <= chunk:
            return _orig_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, enable_gqa=enable_gqa)
        outs = []
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            # attn_mask broadcasts over the batch/head dims; keep those dims.
            m = attn_mask[..., i:j, :]
            outs.append(_orig_sdpa(query[..., i:j, :], key, value,
                                   attn_mask=m, dropout_p=dropout_p,
                                   is_causal=is_causal, scale=scale,
                                   enable_gqa=enable_gqa))
        return torch.cat(outs, dim=-2)

    torch.nn.functional.scaled_dot_product_attention = chunked
    return _orig_sdpa


def render(model_dir, key_png, out_dir):
    import torch
    from diffusers import HunyuanVideo15ImageToVideoPipeline

    log(f"loading pipeline from {model_dir}")
    t0 = time.time()
    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True leaves the weights on the meta device, and then
        # every offload hook dies with "Cannot copy out of meta tensor; no data!".
        low_cpu_mem_usage=False,
    )
    log(f"pipeline loaded in {(time.time() - t0) / 60:.1f} min")
    log(f"components: {sorted(pipe.components.keys())}")
    log(f"transformer: {type(pipe.transformer).__name__}, "
        f"in_channels={getattr(pipe.transformer.config, 'in_channels', '?')}, "
        f"target_size={getattr(pipe.transformer.config, 'target_size', '?')}")

    # ---------------------------------------------------------------- offload
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA")

    _preload = ["MultiheadAttention"]
    _gpu0 = torch.device("cuda:0")
    _gpu1 = (torch.device("cuda:1")
             if torch.cuda.device_count() > 1 else _gpu0)
    log(f"GPUs visible: {torch.cuda.device_count()} -> "
        f"cuda:0 for small components + transformer, cuda:1 for text_encoder")

    # v23 measured this exact split and it fit:
    #   cuda:0 after dispatch: alloc=9.59 GiB, reserved=9.60 GiB
    #   cuda:1 after offloads: alloc=0.00 GiB
    # The Qwen2.5-VL text model is ~14 GB bf16, so it gets GPU 1 to itself.
    from accelerate import cpu_offload, dispatch_model, infer_auto_device_map

    _te = getattr(pipe, "text_encoder", None)
    if isinstance(_te, torch.nn.Module):
        try:
            cpu_offload(_te, _gpu1, preload_module_classes=_preload)
            log("module-level offload: text_encoder -> cuda:1 (preload=MHA)")
        except Exception as e:
            log(f"offload of text_encoder to cuda:1 failed: {e}")

    for _cname in ("text_encoder_2", "image_encoder", "vae"):
        _comp = getattr(pipe, _cname, None)
        if isinstance(_comp, torch.nn.Module):
            try:
                cpu_offload(_comp, _gpu0, preload_module_classes=_preload)
                log(f"module-level offload: {_cname} -> cuda:0 (preload=MHA)")
            except Exception as e:
                log(f"offload of {_cname} failed: {e}")

    for _i in range(torch.cuda.device_count()):
        log(f"cuda:{_i} after offloads: "
            f"alloc={torch.cuda.memory_allocated(_i) / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved(_i) / 2**30:.2f} GiB")

    _nsplit = getattr(type(pipe.transformer), "_no_split_modules", None) or [
        "HunyuanVideo15TransformerBlock"]
    _dmap = infer_auto_device_map(
        pipe.transformer,
        max_memory={0: "7GiB", "cpu": "28GiB"},
        no_split_module_classes=_nsplit,
    )
    log(f"device map: {len(_dmap)} entries, no_split={_nsplit}")
    pipe.transformer = dispatch_model(
        pipe.transformer,
        device_map=_dmap,
        preload_module_classes=_preload,
    )
    log("transformer dispatched with preload_module_classes=[MultiheadAttention]")

    # The meta tensors accelerate reports are INTENTIONAL - they are the
    # CPU-offloaded blocks, materialised on demand by AlignDevicesHook. v22 tried
    # to "repair" them out of the safetensors shards; that was wrong twice over
    # (it defeats the offload and it silently matched nothing).
    _n_meta = sum(1 for _p in pipe.transformer.parameters()
                  if _p.device.type == "meta")
    log(f"transformer meta params (= CPU-offloaded blocks): {_n_meta}")
    for _i in range(torch.cuda.device_count()):
        log(f"cuda:{_i} after dispatch: "
            f"alloc={torch.cuda.memory_allocated(_i) / 2**30:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved(_i) / 2**30:.2f} GiB")

    for _name, _comp in pipe.components.items():
        if isinstance(_comp, torch.nn.Module):
            _m = sum(1 for _mod in _comp.modules()
                     for _t in _mod.parameters(recurse=False)
                     if _t.device.type == "meta")
            if _m:
                log(f"{_name}: {_m} META tensors (offloaded, loaded on demand)")

    for fn in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, fn)()
            log(f"vae {fn} enabled")
        except Exception:
            pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- key frame
    from PIL import Image

    key = Image.open(key_png).convert("RGB")
    log(f"key frame {key_png.name}: {key.size[0]}x{key.size[1]}")

    # 480p_i2v is roughly 480x848. It is NOT 9:16, so the key frame is
    # letterboxed-free centre-cropped to the model's own aspect and the finished
    # 5 s clip is scaled to 1080x1920 at mux time.
    width, height = 480, 848
    key = key.resize((width, height), Image.LANCZOS)
    # v24 (720p, 121f) and v25 (480p, 121f) BOTH died inside the denoise loop on
    # the SSTA attention mask, not on weights:
    #
    #   transformer_hunyuan_video15.py:106
    #       attention_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
    #   torch.OutOfMemoryError: Tried to allocate 2.36 GiB.
    #   GPU 0 has a total capacity of 14.56 GiB of which 1.33 GiB is free.
    #
    # That mask is N x N bool, N = spatial_tokens * temporal_tokens. At 480p
    # spatial is ~1590 (30x53) and 121 frames give ceil(121/4)=31 temporal
    # slices, so N ~= 47,700 and N^2 = 2.3 GB - and it is built three times over
    # (mask_1, mask_2, then the AND) before anything else runs.
    #
    # N scales with FRAME COUNT, not with wall-clock duration. So the demo can
    # still be ~5 seconds: 81 frames at 16 fps is 5.06 s, and N drops to
    # 1590 * ceil(81/4) = 33,390, making the mask 1.11 GB and the three-way peak
    # ~3.3 GB, which fits the ~6.5 GiB left after an 8 GiB transformer budget.
    num_frames = 81
    fps = 16  # 81 / 16 = 5.06 s - the agreed demo length
    log(f"render target {width}x{height}, {num_frames} frames @ {fps} fps "
        f"= {num_frames / fps:.2f} s")

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(20261011)

    # ----------------------------------------------------------------------
    # v26 got all the way into the denoise loop with the CORRECT architecture
    # (log line: "transformer: HunyuanVideo15Transformer3DModel,
    # in_channels=65, target_size=640") and died on the attention backend:
    #
    #   diffusers/models/attention_dispatch.py:3707 in _native_attention
    #       out = torch.nn.functional.scaled_dot_product_attention(...)
    #   torch.OutOfMemoryError: Tried to allocate 71.96 GiB.
    #   GPU 0 has a total capacity of 14.56 GiB of which 126.81 MiB is free.
    #
    # 71.96 GiB is exactly H * N^2 * 4 bytes with H=16 heads and N=33,390
    # tokens - i.e. PyTorch picked the MATH backend, which materialises the
    # whole attention score matrix in float32. That is O(N^2) memory.
    #
    # The T4 is sm75. Flash attention v2 needs sm80+, so SDPBackend.FLASH is
    # unavailable here - but SDPBackend.EFFICIENT_ATTENTION (the
    # xformers-derived cutlass kernel) supports sm50+ and is O(N) instead of
    # O(N^2). It also accepts an explicit attn_mask, which HunyuanVideo-1.5
    # needs for its variable-length sequences.
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # v27 forced SDPBackend.EFFICIENT_ATTENTION and it hard-failed BEFORE the
    # denoise loop even started, inside the SigLIP image encoder:
    #
    #   pipeline_..._image2video.py:400  image_encoder(**image)
    #   RuntimeError: No available kernel. Aborting execution.
    #
    # PyTorch's memory-efficient backend also does not accept enable_gqa=True,
    # which is exactly what the transformer's attention passes. So neither
    # EFFICIENT nor FLASH (T4 is sm75, flash needs sm80+) is usable directly.
    #
    # The backend that DOES work on Turing and does accept an explicit
    # attn_mask plus grouped queries is xformers'
    # memory_efficient_attention, which diffusers exposes as
    # attention_backend("xformers"). Install it, then probe each candidate
    # against a realistic call (bf16, bool attn_mask, grouped query) and keep
    # the first that actually runs - rather than gambling the whole 10 minute
    # model load on one guess.
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # v27 forced EFFICIENT_ATTENTION alone and hard-failed inside the SigLIP
    # image encoder ("No available kernel"). v28 installed xformers and
    # flash_hub and diffusers rejected BOTH:
    #   flash_hub -> "the `kernels` package isn't available"
    #   xformers  -> "missing package or the version is too old"
    # so it fell back to native, which is the O(N^2) MATH path and OOMs at
    # exactly 71.96 GiB = H * N^2 * 4 with H=16, N=33,390.
    #
    # Two things were never actually tested:
    #   1. whether EFFICIENT_ATTENTION supports the TRANSFORMER's call, which
    #      passes enable_gqa=True (v27 died in SigLIP before reaching it), and
    #   2. whether MATH as a *fallback* rather than the only option is enough -
    #      SigLIP's sequence is tiny, so MATH is harmless there.
    #
    # So: probe every candidate against the exact call shape the transformer
    # makes, log what actually works, and only then pick.
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # v29's probe was conclusive but its install was blind: `pip install -q`
    # swallowed the reason xformers and kernels never imported. And the probe
    # itself was malformed - the "no gqa" case passed q with 16 heads against
    # k with 4, which is a shape mismatch, not a backend-selection failure.
    #
    # v30: install verbosely, then test the libraries DIRECTLY rather than
    # through diffusers' availability check, and probe SDPA with matched heads
    # so a real answer comes back.
    # ----------------------------------------------------------------------
    # The v31 probe is deliberately GONE from this path. It answered the
    # question - on this T4 at N=33,390 there is no fused attention kernel and
    # MATH needs 66.45 GiB - but it also left 13.72 GiB of the 14.56 GiB card
    # reserved, which is most of the headroom the render itself needs.
    # xformers 0.0.35 and kernels 0.17.2 are already on the image from v30.
    log("skipping the backend probe (answered in v31); installing chunked SDPA")

    _install_chunked_sdpa(SDPA_CHUNK)
    _N = 1590 * max(1, (num_frames + 3) // 4)
    log(f"installed chunked SDPA (chunk={SDPA_CHUNK}); "
        f"N={_N}, score matrix per chunk ~= "
        f"{16 * SDPA_CHUNK * _N * 4 / 2**30:.2f} GiB")

    log("=" * 70)
    log("RENDERING")
    log("=" * 70)
    t0 = time.time()
    # NOTE: HunyuanVideo15ImageToVideoPipeline.__call__ takes NO height/width.
    # Resolution comes from transformer.config.target_size, derived from this
    # checkpoint's own `ideal_resolution: "720p"`. Passing height= or width=
    # raises TypeError: ... got an unexpected keyword argument 'height'
    out = pipe(
        image=key,
        prompt=MOTION_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=num_frames,
        num_inference_steps=30,
        generator=generator,
        output_type="np",
    )
    log(f"render finished in {(time.time() - t0) / 60:.1f} min")

    frames = out.frames[0] if isinstance(out.frames, (list, tuple)) else out.frames
    arr = np.asarray(frames)
    log(f"frames shape {arr.shape}")

    # ---- frame-to-frame motion measurement --------------------------------
    # This is the number that caught the day_05 collage (1.62/255 = a still
    # image with a pan). A real animated shot measures 5-20+. Report it every
    # run so a regression is impossible to miss.
    if arr.ndim == 4 and arr.shape[0] > 1:
        g = arr[:, ::4, ::4, :].astype(np.float32).mean(axis=3)
        d = np.abs(np.diff(g, axis=0)).mean()
        log(f"frame-to-frame mean abs diff: {d:.2f} / 255 "
            f"(a static collage measures ~1.6, real motion 5-20+)")
        if d < 3.0:
            log("WARNING: motion is far too low - this looks like a still image")

    # ---- write mp4 -------------------------------------------------------
    import imageio.v2 as imageio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_u8 = np.clip((arr * 255).round(), 0, 255).astype("uint8")

    video_path = out_dir / "clip.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p")
    for fr in frames_u8:
        writer.append_data(fr)
    writer.close()
    log(f"wrote {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")

    # ---- audio: HunyuanVideo-1.5 is video-only --------------------------
    audio = getattr(out, "audio", None)
    if audio is not None:
        try:
            import soundfile as sf

            a = audio[0] if isinstance(audio, (list, tuple)) else audio
            a = np.asarray(a)
            if a.ndim > 1:
                a = a.squeeze()
            sf.write(str(out_dir / "clip.wav"), a, 16000)
            log(f"wrote clip.wav {a.shape}")
        except Exception as e:
            log(f"audio write failed: {e}")
    else:
        log("no audio in output (HunyuanVideo-1.5 is video-only) -> "
            "the procedural audio track is muxed later, off-GPU")

    log("=" * 70)
    log("DONE")
    log("=" * 70)


def main():
    report_env()

    key_png = find_input("scene_pair_key.png", (".png",))
    if key_png is None:
        key_png = find_input("scene_pair_key", (".png",))
    if key_png is None:
        log("STOP no key frame found under /kaggle/input")
        raise SystemExit("STOP no key frame")
    log(f"key frame: {key_png}")

    # /tmp, never /kaggle/working: Kaggle collects /kaggle/working as kernel
    # output and a 33 GB model there exceeds the output cap and kills the run.
    cache = Path("/tmp/hy15_d")
    sentinel = cache / ".snapshot_path"
    if sentinel.exists() and Path(sentinel.read_text().strip()).exists():
        model_dir = Path(sentinel.read_text().strip())
        log(f"reusing cached snapshot: {model_dir}")
    else:
        model_dir = Path(download_model(REPO, cache))
        sentinel.write_text(str(model_dir))

    log(f"model dir: {model_dir}")
    tree(model_dir)

    render(model_dir, key_png, "/kaggle/working")

    dst = Path("/kaggle/working/clip.mp4")
    if dst.exists():
        log(f"final artefact: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
