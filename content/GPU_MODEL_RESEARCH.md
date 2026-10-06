# GPU Model Research — "Are those the biggest model available in public?"

**Date:** 2026-10-06
**Question asked:** whether the models previously discussed (Wan 2.2 5B, LTX-Video, HunyuanVideo 13B) were
the biggest open/public models available, and which is the biggest *local* model usable as a backup.

**Short answer: No — they were not the biggest. The biggest public/open-weight video models are
33B, 30B and 22B. None of them fit the free Kaggle T4. The biggest one that *does* fit is
HunyuanVideo 1.5 at 8.3B.**

---

## 1. The biggest public / open-weight video models (2026)

| Rank | Model | Org | Params | Max output | Native audio | Licence | Fits free T4? |
|---|---|---|---|---|---|---|---|
| 1 | **MiniMax H3** (Hailuo 3.0) | MiniMax | **33.1B** | 4–15 s, 2K API | ✅ stereo | Community, <$20M rev | ❌ |
| 2 | **Step-Video-T2V** | StepFun | **30B** | 204 frames | ❌ | MIT | ❌ (and T2V-only) |
| 3 | **LTX-2.5** | Lightricks/LTX | **22B** | 4K, 20 s | ✅ native | free <$10M ARR | ⚠️ borderline (~37 GB INT8) |
| 4 | LTX-2 | Lightricks | 19B | 4K, 20 s | ✅ native | dual | ❌ (83 GB all-in) |
| 5 | Wan 2.2 (MoE) | Alibaba | 27B total / 14B active | 720p, 5 s | ❌ | **Apache 2.0** | ❌ |
| 6 | HunyuanVideo 1.5 | Tencent | 13B (8.3B actual) | 720p, 15 s | ✅ | Tencent community | ✅ **YES** |
| 7 | HunyuanVideo (v1) | Tencent | 13B | 720p, 5 s | ❌ | Tencent community | ❌ (60–80 GB) |
| 8 | Mochi 1 | Genmo | 10B | 480p | ❌ | Apache 2.0 | ❌ (60 GB) |
| 9 | LTX-Video 0.9.x | Lightricks | 2B | 768×512 | ❌ | OpenRAIL-M | ✅ |
| 10 | Wan 2.1/2.2 TI2V-5B | Alibaba | 5B | 720p, 5 s | ❌ | Apache 2.0 | ✅ (T2V only, see §3) |

Notes on the top three:

- **MiniMax H3** — released 2026-07-31, open-sourced 2026-08-03. Unified multimodal transformer reading
  text/image/video/audio in one context. The **text encoder alone is 48 GB at full precision
  (14.6 GB quantised)**. Explicitly not a consumer-GPU model. Licence caps free commercial use at
  $20M annual revenue.
- **Step-Video-T2V** — the largest open *text-to-video* model at 30B, clean MIT licence, but
  **text-to-video only** — it cannot be conditioned on our composed key frame, which is the entire
  point of the route.
- **LTX-2.5** — the most interesting of the three: 22B, **image-to-video**, **native synchronised
  audio generated in the same pass as the video**, native multishot generation that preserves
  *character identity, lighting, environment, sound and visual style* across consecutive shots in a
  single generation. Distilled checkpoint runs an **8-step** denoise; FP8 quantisation cuts VRAM by
  ~40%; documented to start at **16 GB VRAM** with CPU offload. 912,729 HF downloads by 2026-08-28.
  Local deployment via ComfyUI and Diffusers.

---

## 2. Measured free-Kaggle runner limits (this account, measured not assumed)

| Resource | Measured value |
|---|---|
| GPU | **Tesla T4 × 2, 15360 MiB each** (type cannot be chosen) |
| torch | `2.11.0+cu128`, `cuda True`, `devices 2` |
| System RAM | **31 GB** |
| Disk | **1.1 TB free** |
| CPUs | 4 |
| Internet | **reachable** (confirmed by a successful runtime pip download) |
| GPU quota | 30.00 h/week; 2.80 h used, 27.20 h left, refresh 2026-10-10 |

---

## 3. What actually fits, and why each big model fails

### Wan 2.2 TI2V-5B — fits, but is **text-to-video only**

This cost 2.80 GPU-h to discover. Despite the "TI2V" name, the diffusers port carries **no
`image_encoder/` subfolder** (verified from the HF API with `?blobs=true`: 29 files, `image_encoder`
absent). `model_index.json` declares:

```json
"_class_name": "WanPipeline",
"boundary_ratio": null,
"transformer_2": [null, null]
```

and `transformer/config.json` has `"image_dim": null`. `WanPipeline.__call__` takes no `image=`
(confirmed by reading the signature in `pipeline_wan.py`). It therefore **cannot be conditioned on
`scene_pair_key.png`** — which is the whole route. `WanImageToVideoPipeline` exists in diffusers but
requires a CLIP `image_encoder` this model does not ship.

### Wan 2.1 I2V 14B — has the image encoder, is far too big

`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` *does* ship `image_encoder/model.safetensors` (1.26 GB) and
`image_processor/`, so it is genuinely image-conditioned. But its transformer is **64.7 GB** across 14
shards (plus a 22.7 GB text encoder). Even FP8 leaves ~32 GB, which exceeds both 16 GB VRAM and
31 GB system RAM.

### LTX-2 19B — diffusers-native, but 83 GB all-in

`Lightricks/LTX-2` is a clean diffusers repo (`model_index.json`, `text_encoder/` 12 shards,
`audio_vae/`, `connectors/`, `scheduler/`). Total requirement:

- transformer `ltx-2-19b-distilled.safetensors` — **43.3 GB bf16** (19 GB as FP8)
- text encoder Gemma 3 12B — **~40 GB bf16** (no quantised variant published on that repo)

43 + 40 = **83 GB** against 31 GB of system RAM. Dead on the free T4.

### LTX-2.5 22B — closest, still short

`kizitoezirim/ltx-2-5-transformer-int8-convrot` on Kaggle holds the **entire LTX-2.5 distilled stack
in one 49 GB dataset**:

| File | Size |
|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21.5 GB |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.4 GB |
| `ltx-2.5-video-vae-conv-bf16.safetensors` | 1.45 GB |
| `ltx-2.5-audio-vae-bf16.safetensors` | 365 MB |
| `ltx-2.5-duration-head-bf16.safetensors` | 3.8 MB |

21.5 + 15.4 + 1.5 ≈ **38 GB** against 31 GB RAM. Borderline but over. It is also ComfyUI-layout
single-file weights, not a diffusers tree, so it needs conversion. **Retained as the first upgrade
target if a bigger free GPU ever appears.**

### HunyuanVideo 1.5 (8.3B) — **fits. This is the chosen route.**

`tencent/HunyuanVideo-1.5` is **fully diffusers-format on Hugging Face**:

```
transformer/720p_i2v/diffusion_pytorch_model.safetensors          33306.6 MB
transformer/720p_i2v_distilled/...                                 33306.6 MB
transformer/480p_i2v_step_distilled/...                            33325.5 MB
vae/diffusion_pytorch_model.safetensors                             5042.6 MB
scheduler/scheduler_config.json
config.json                     (_class_name: HunyuanVideo_1_5_Pipeline)
```

diffusers 0.40 ships `HunyuanVideo15ImageToVideoPipeline`, `HunyuanVideo15Transformer3DModel` and
`AutoencoderKLHunyuanVideo15`, so **no ComfyUI and no GGUF conversion are needed**.

Memory budget with FP8 transformer + `enable_model_cpu_offload()`:

| Component | Size |
|---|---|
| transformer (FP8) | ~17 GB |
| text encoder Qwen2.5-VL 7B | ~9.4 GB |
| byT5 glyph encoder | 0.44 GB |
| VAE | 2.5 GB (fp16) |
| SigCLIP image encoder (i2v conditioning) | 0.86 GB |
| **total** | **~30 GB** → fits 31 GB RAM, streams through 16 GB VRAM |

The pipeline is genuinely image-conditioned:
`model_cpu_offload_seq = "image_encoder->text_encoder->transformer->vae"` — the `image_encoder`
(SigCLIP) is what Wan 2.2 5B was missing.

**API (read from `pipeline_hunyuan_video1_5_image2video.py`, line 652):**

```python
pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16,
    scheduler=FlowMatchEulerDiscreteScheduler(flow_shift=7.0))   # config.json asks for flow_shift 7.0
pipe.transformer.to(torch.float8_e4m3fn)
pipe.enable_model_cpu_offload(); pipe.vae.enable_tiling(); pipe.vae.enable_slicing()

out = pipe(
    image=key_frame_pil,          # 720x1280
    prompt=MOTION_PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=1280, width=720,
    num_frames=121,               # ~5 s at 24 fps
    num_inference_steps=30,
    generator=generator,          # fixed seed 20261006
    output_type="np",
)
# out.frames -> video.  No audio: HunyuanVideo-1.5 is video-only.
```

**Important:** `config.json` names custom classes
(`hyvideo.schedulers.scheduling_flow_match_discrete.FlowMatchDiscreteScheduler`,
`hyvideo.models.text_encoders.TextEncoder`) that are **not importable from PyPI**. The kernel
therefore constructs a diffusers-native `FlowMatchEulerDiscreteScheduler(flow_shift=7.0)` and passes
it explicitly, so `from_pretrained` never tries to import `hyvideo`.

**A ready-made Kaggle stack also exists** if ComfyUI is ever preferred:
`selenagomezrealai/hunyuanvideo-1-5-8-3b-distilled-q4-stack-v1` (16.2 GB) — `diffusion_models/
hunyuanvideo1.5_480p_i2v_step_distilled-Q4_K_M.gguf` (5.1 GB), `text_encoders/
qwen_2.5_vl_7b_fp8_scaled.safetensors` (9.4 GB), `text_encoders/byt5_small_glyphxl_fp16.safetensors`
(0.44 GB), `vae/hunyuanvideo15_vae_fp16.safetensors` (2.5 GB), `clip_vision/
sigclip_vision_patch14_384.safetensors` (0.86 GB). GGUF only needs ComfyUI; the safetensors files
drop straight into the diffusers layout.

---

## 4. Decision and the fallback chain

| Tier | Route | Model | Why |
|---|---|---|---|
| **1** | Kaggle T4 kernel `mdhkids/hy-clip` | **HunyuanVideo 1.5 8.3B i2v** | biggest i2v model that fits 16 GB VRAM / 31 GB RAM; diffusers-native; conditions on `scene_pair_key.png` |
| **2** | same kernel, swap weights | LTX-2.5 22B INT8 | bigger + native audio + multishot, needs ~38 GB — only if RAM/VRAM grows |
| **3** | Kaggle T4 kernel | LTX-Video 2B i2v | guaranteed fit, ~LTX quality tier, same key-frame route |
| **4** | local stills pipeline (proven) | composite + `build_short.py` v2 | free, scriptable, already works end to end |
| **5** | GitHub Actions CPU | LTX-Video 0.9.5 | backup only: 4 vCPU, no CUDA, 6 h job cap |

---

## 5. What is still unresolved

1. `mdhkids/hy-clip` reaches the render loop but dies on VRAM. Fifteen kernel versions
   were pushed; every non-memory bug is fixed. The remaining problem is a single
   memory-granularity issue, documented in §6.
2. The 33 GB HF download happens at runtime every run. Pre-staging a Kaggle dataset from
   `tencent/HunyuanVideo-1.5` (720p_i2v + vae + text encoders only) would remove that cost
   and is the obvious next optimisation.
3. `720p_i2v_distilled` and `480p_i2v_step_distilled` variants exist and are the same size —
   a distilled variant would cut the 30-step denoise substantially and is worth testing once
   the base path renders.

---

## 6. Kernel log — the 15-version debugging record

Every bug below was found by pushing and reading `hy-clip.log`. The kernel now runs on
`mdhkids/hy-clip` (Kernel version 15), source at `kaggle/hyclip/hy_clip.py`.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | kernel killed mid-download | model was written to `/kaggle/working`, which Kaggle collects as **output**; the 33 GB transformer exceeded the output cap | download to `/tmp/hf_hy15` instead. `/kaggle/working` is only **21 GB**; `/tmp` is on the larger volume |
| 2 | `TypeError: FlowMatchEulerDiscreteScheduler.__init__() got an unexpected keyword argument 'flow_shift'` | diffusers kwarg is `shift`, not `flow_shift` | `FlowMatchEulerDiscreteScheduler(shift=7.0)` |
| 3 | `FileNotFoundError: model-00001-of-00005.safetensors` | `allow_patterns` omitted `*.safetensors`, so only json/tokenizer files landed | added `*.safetensors` **and** `*.bin` |
| 4 | `OSError: no file named model.safetensors, or pytorch_model.bin` (byt5-small) | that repo ships `pytorch_model.bin` | added `*.bin` to the patterns |
| 5 | `TypeError: empty(): argument 'size' failed to unpack the object at pos 4` | **config schema drift**: 0.35.0 used `patch_size: [t,h,w]`; 0.40.0 uses int `patch_size` + int `patch_size_t` | translate `patch_size[0] -> patch_size_t`, `patch_size[1] -> patch_size` |
| 6 | `ValueError: unknown qk_norm: True` | second schema drift: 0.35 used bool `qk_norm` + `qk_norm_type`; 0.40 wants a string | `qk_norm=True` + `qk_norm_type="rms"` → `qk_norm="rms_norm"` |
| 7 | `TypeError: ConfigMixin.from_config() got multiple values for argument 'config'` | `from_pretrained(config=...)` collides with the path-derived config | write the translated config to `/tmp/hy15_transformer_fixed` and load from there |
| 8 | `Cannot copy out of meta tensor; no data!` | `from_pretrained`'s default `low_cpu_mem_usage=True` left weights on the **meta** device | `low_cpu_mem_usage=False` on every `from_pretrained`. Verified fixed: `transformer params on real device: 1801/1801` |
| 9 | `Cannot copy out of meta tensor` (again) | a plain `.to(torch.float8_e4m3fn)` is a silent no-op on meta tensors, and torch cannot move fp8 tensors between devices at all | quantise with `torchao` instead of a dtype cast |
| 10 | `RuntimeError: Tensor on device meta is not on the expected device cuda:0!` | a meta straggler invisible to a top-level check | recursive walk over `named_modules()` checking **parameters and buffers** |
| 11 | `TypeError: __call__() got an unexpected keyword argument 'height'` | `HunyuanVideo15ImageToVideoPipeline.__call__` takes **no** `height`/`width`; resolution comes from `transformer.config.target_size`, derived from `ideal_resolution` | drop `height=`/`width=` from the call |
| 12 | **CUDA OOM, 14.24 GiB of 14.56 GiB — CURRENT** | `enable_model_cpu_offload` treats the whole transformer as one unit; int8 leaves it at ~16.6 GB, which cannot fit the T4 | see below |

### The remaining problem

`model_cpu_offload_seq = "image_encoder->text_encoder->transformer->vae"` offloads at
module granularity, and accelerate is moving the entire ~16.6 GB int8 transformer onto the
14.56 GiB card in one go. `torchao`'s `int4_weight_only()` — which would bring it to ~8.3 GB
and fit — raises a meta-registration error in the torchao version available on the runner.

**Next attempt, in order of likelihood:**

1. **Force block-level offload.** Set `transformer._no_split_modules = ["HunyuanVideo15TransformerBlock"]`
   (or confirm the config declares it) so accelerate moves one block at a time instead of the
   whole 54-block stack. This is the cheapest fix and needs no re-quantisation.
2. **Quantise the text encoder too.** The Qwen2.5-VL language tower is ~15 GB in bf16; int8
   brings it to ~7.5 GB, which leaves headroom once the transformer is also smaller.
3. **Load the transformer with `device_map="auto"`, `max_memory={0: "13GiB", "cpu: "28GiB"}`**,
   letting accelerate split it across GPU and CPU itself, and drop the pipeline-level offload.
4. **Use the 480p variant** (`transformer/480p_i2v`) and lower `num_frames` — smaller activations,
   same weight size. Only helps if the OOM is activation-driven rather than weight-driven.
5. **Retry int4** with the API the installed torchao actually exposes (`int4_weight_only(group_size=...)`,
   or `quantize_(m, int4_quantize(...))`) — the current call form is what fails.

GPU budget is healthy for this: **4.97 h used, 25.03 h of 30.00 h remaining, refresh 2026-10-10.**

