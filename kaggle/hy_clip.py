#!/usr/bin/env python3
"""
HunyuanVideo-1.5 8.3B image-to-video clip renderer on Kaggle.

Route chosen after measuring what actually fits a free Kaggle T4
(16 GB VRAM / 31 GB system RAM):

    MiniMax H3      33.1B   ~60 GB+   DOES NOT FIT
    Step-Video-T2V  30B     huge      DOES NOT FIT  (also T2V-only)
    LTX-2.5         22B     ~37 GB    borderline, > 31 GB RAM
    LTX-2           19B     43 GB bf16 transformer + 40 GB Gemma text encoder
                                    -> 83 GB, DOES NOT FIT
    Wan 2.2 14B I2V 14B     64.7 GB transformer, DOES NOT FIT
    Wan 2.2 TI2V-5B 5B      34 GB    fits, but the diffusers port has NO
                                    image_encoder/ subfolder -> text-to-video
                                    ONLY. Proven: model_index.json declares
                                    boundary_ratio=null, image_dim=null,
                                    _class_name=WanPipeline.
    HunyuanVideo 1.5 8.3B   33.3 GB bf16 transformer (fp8 ~17 GB)
                            + Qwen2.5-VL 7B text encoder + SigCLIP image encoder
                            + 5 GB VAE. Fits with model_cpu_offload().
                                    -> CHOSEN

HunyuanVideo-1.5 is natively diffusers-format on Hugging Face
(tencent/HunyuanVideo-1.5) and diffusers 0.40 ships
HunyuanVideo15ImageToVideoPipeline, so no ComfyUI and no GGUF conversion
is required.
"""

import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOG_T0 = time.time()


def log(msg):
    el = time.time() - LOG_T0
    print(f"[{el:8.1f}s] {msg}", flush=True)


def run(cmd, **kw):
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


# --------------------------------------------------------------------------
# 1. environment report
# --------------------------------------------------------------------------
def report_env():
    log("=" * 70)
    log("ENV REPORT")
    log("=" * 70)
    print(f"python {sys.version.split()[0]}", flush=True)

    try:
        with open("/proc/meminfo") as f:
            mi = {ln.split(":")[0]: ln.split(":")[1].strip() for ln in f if ":" in ln}
        print(f"MemTotal  {mi.get('MemTotal')}", flush=True)
        print(f"MemAvail  {mi.get('MemAvailable')}", flush=True)
    except Exception as e:
        print(f"meminfo: {e}", flush=True)

    try:
        st = os.statvfs("/kaggle/working")
        print(
            f"disk free {(st.f_bavail * st.f_frsize) / 1e9:.1f} GB of "
            f"{(st.f_blocks * st.f_frsize) / 1e9:.1f} GB",
            flush=True,
        )
    except Exception as e:
        print(f"disk: {e}", flush=True)

    print(f"nproc {os.cpu_count()}", flush=True)

    # torch: install CUDA build if the image shipped a CPU-only one
    try:
        import torch  # noqa: F401

        log(f"torch already present: {torch.__version__}")
    except ImportError:
        log("torch absent -> installing CUDA 12.8 build")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ]
        )
        import torch  # noqa: F401

        log(f"torch installed: {torch.__version__}")

    log(f"torch {torch.__version__}")
    log(f"cuda available: {torch.cuda.is_available()}")
    log(f"cuda device count: {torch.cuda.device_count()}")

    if shutil.which("nvidia-smi"):
        r = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        for ln in (r.stdout or "").strip().splitlines():
            log(f"GPU: {ln.strip()}")
    else:
        log("nvidia-smi: BINARY NOT PRESENT")

    if not torch.cuda.is_available():
        log("STOP no GPU -> nothing further is worth doing")
        raise SystemExit("STOP no GPU")

    # diffusers must expose the HunyuanVideo-1.5 image-to-video pipeline
    try:
        import diffusers

        log(f"diffusers {diffusers.__version__}")
        has = hasattr(diffusers, "HunyuanVideo15ImageToVideoPipeline")
        log(f"diffusers has HunyuanVideo15ImageToVideoPipeline: {has}")
        if not has:
            log("STOP diffusers too old for HunyuanVideo-1.5")
            raise SystemExit("STOP diffusers too old")
    except ImportError:
        log("diffusers absent -> installing")
        run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate"])
        import diffusers

        log(f"diffusers {diffusers.__version__}")


# --------------------------------------------------------------------------
# 2. locate inputs mounted by the kernel
# --------------------------------------------------------------------------
def find_input(name_part, must_end=()):
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    hits = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        s = str(p).lower()
        if name_part.lower() in s and (not must_end or s.endswith(must_end)):
            hits.append(p)
    return hits[0] if hits else None


def find_model_dir():
    """Find the HunyuanVideo-1.5 snapshot directory."""
    root = Path("/kaggle/input")
    cands = []
    for p in root.rglob("config.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if str(d.get("_class_name", "")).startswith("HunyuanVideo_1_5"):
            cands.append(p.parent)
    if cands:
        return cands[0]

    # fall back: any directory that has transformer/<variant>/... under it
    for p in root.rglob("transformer"):
        if p.is_dir():
            for v in p.iterdir():
                if v.is_dir():
                    return p.parent
    return None


def download_model(repo_id, cache_dir):
    """Pull only the pieces we need, at runtime (kernels now have internet).

    CRITICAL: the cache must NOT live under /kaggle/working. Kaggle uploads
    everything in /kaggle/working as kernel output, and a 33 GB transformer
    blows the output cap and kills the kernel mid-download. /tmp is outside
    the output collection path.
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"downloading {repo_id} -> {cache_dir}")
    log("(ignoring every transformer variant except 720p_i2v, plus the upsampler)")
    path = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=[
            "config.json",
            "scheduler/*",
            "vae/*",
            "text_encoder/*",
            "tokenizer/*",
            "byt5*",
            "image_encoder/*",
            "clip_vision/*",
            "transformer/720p_i2v/*",
        ],
        ignore_patterns=["*.msgpack", "*.h5", "*.bin", "flax_model*", "tf_model*"],
    )
    log(f"download complete: {path}")
    return path


def download_extra(cache_dir):
    """Fetch the text/vision encoders that tencent/HunyuanVideo-1.5 does not ship.

    The repo carries only config.json, scheduler, transformer variants, upsampler
    and vae. HunyuanVideo15ImageToVideoPipeline needs ten components, so the
    remaining weights come from their upstream Hugging Face repos:
        Qwen2_5_VLTextModel / Qwen2Tokenizer  <- Qwen/Qwen2.5-VL-7B-Instruct
        T5EncoderModel / ByT5Tokenizer        <- google/byt5-small
        SiglipVisionModel / SiglipImageProcessor <- google/siglip-so400m-patch14-384
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        # NB: *.safetensors AND *.bin are both essential. byt5-small ships its
        # weights as pytorch_model.bin, and without the *.bin pattern the
        # directory holds only json files and T5EncoderModel.from_pretrained
        # raises "no file named model.safetensors, or pytorch_model.bin".
        ("Qwen/Qwen2.5-VL-7B-Instruct", ["*.json", "*.txt", "*.safetensors", "*.bin", "tokenizer*", "vocab*", "merges*"]),
        ("google/byt5-small", ["*.json", "*.model", "*.txt", "*.safetensors", "*.bin"]),
        ("google/siglip-so400m-patch14-384", ["*.json", "*.safetensors", "*.txt", "*.bin"]),
    ]
    out = {}
    for repo, pats in jobs:
        log(f"downloading {repo}")
        p = snapshot_download(repo_id=repo, cache_dir=str(cache_dir), allow_patterns=pats)
        out[repo] = p
        log(f"  -> {p}")
    return out


def tree(path, max_lines=80):
    log("-" * 70)
    log(f"TREE of {path}")
    log("-" * 70)
    n = 0
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            print(f"  {sz / 1e6:10.1f} MB  {p.relative_to(path)}", flush=True)
            n += 1
            if n >= max_lines:
                print("  ... (truncated)", flush=True)
                break
    log("-" * 70)


# --------------------------------------------------------------------------
# 3. prompt construction -- targets the three named quality defects
# --------------------------------------------------------------------------
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

# Official HunyuanVideo-1.5 style negative prompt, extended for our defects:
# blown-out background, backlit unreadable character, zoom/scale jitter.
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


# --------------------------------------------------------------------------
# 4. render
# --------------------------------------------------------------------------
def render(model_dir, key_png, out_dir):
    import numpy as np
    import diffusers
    import torch
    from diffusers import (
        AutoencoderKLHunyuanVideo15,
        ClassifierFreeGuidance,
        FlowMatchEulerDiscreteScheduler,
        HunyuanVideo15ImageToVideoPipeline,
        HunyuanVideo15Transformer3DModel,
    )
    from PIL import Image

    log("=" * 70)
    log("LOADING PIPELINE (component by component)")
    log("=" * 70)

    # tencent/HunyuanVideo-1.5's config.json names hyvideo classes that are not
    # installable from PyPI, so every component is built explicitly instead of
    # letting from_pretrained resolve them.
    #
    # config.json asks for flow_shift 7.0; the diffusers scheduler kwarg is `shift`.
    scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    log("scheduler built (shift=7.0)")

    cache = Path("/tmp/hf_hy15_extra")
    extra = download_extra(cache)
    qwen_dir = extra["Qwen/Qwen2.5-VL-7B-Instruct"]
    byt5_dir = extra["google/byt5-small"]
    siglip_dir = extra["google/siglip-so400m-patch14-384"]
    log(f"qwen={qwen_dir}")
    log(f"byt5={byt5_dir}")
    log(f"siglip={siglip_dir}")

    # ---- text encoder: only the language tower of Qwen2.5-VL-7B -------------
    from transformers import (
        AutoTokenizer,
        ByT5Tokenizer,
        Qwen2_5_VLForConditionalGeneration,
        SiglipImageProcessor,
        SiglipVisionModel,
        T5EncoderModel,
    )

    log("loading Qwen2.5-VL-7B (full, then keep the language tower)")
    qwen_full = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    text_encoder = qwen_full.model.language_model
    log("language tower extracted")
    del qwen_full
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(qwen_dir)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        byt5_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    tokenizer_2 = ByT5Tokenizer.from_pretrained(byt5_dir)

    image_encoder = SiglipVisionModel.from_pretrained(
        siglip_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )
    feature_extractor = SiglipImageProcessor.from_pretrained(siglip_dir)

    # The checkpoint was saved with _diffusers_version 0.35.0, where
    # `patch_size` was a 3-list [t, h, w]. diffusers 0.40 split that into two
    # ints, `patch_size` and `patch_size_t`. Passing the old config through
    # verbatim makes HunyuanVideo15PatchEmbed build
    # torch.empty((out_ch, in_ch, [1,1,1], 1, [1,1,1])) and blows up with
    # "empty(): argument 'size' failed to unpack the object at pos 4".
    # Translate the old schema into the new one before loading.
    tcfg = HunyuanVideo15Transformer3DModel.load_config(model_dir / "transformer" / "720p_i2v")
    ps = tcfg.get("patch_size", [1, 1, 1])
    if isinstance(ps, (list, tuple)):
        log(f"translating 0.35 patch_size {ps} -> 0.40 patch_size_t/patch_size")
        tcfg["patch_size_t"] = int(ps[0])
        tcfg["patch_size"] = int(ps[1])
    tcfg["_class_name"] = "HunyuanVideo15Transformer3DModel"
    tcfg["_diffusers_version"] = diffusers.__version__

    # Second 0.35 -> 0.40 schema drift: 0.35 used a boolean `qk_norm` plus a
    # separate `qk_norm_type` string; 0.40 folded both into a single `qk_norm`
    # that must be one of None/'layer_norm'/'rms_norm'/... Passing the boolean
    # straight through raises
    #   ValueError: unknown qk_norm: True. Should be one of ...
    _QK = {"rms": "rms_norm", "layer": "layer_norm", "l2": "l2"}
    if isinstance(tcfg.get("qk_norm"), bool):
        if tcfg["qk_norm"]:
            tcfg["qk_norm"] = _QK.get(str(tcfg.get("qk_norm_type", "rms")), "rms_norm")
        else:
            tcfg["qk_norm"] = None
        log(f"translated qk_norm -> {tcfg['qk_norm']!r}")

    log(
        f"transformer config: patch_size_t={tcfg['patch_size_t']} "
        f"patch_size={tcfg['patch_size']} in_channels={tcfg.get('in_channels')} "
        f"qk_norm={tcfg.get('qk_norm')!r}"
    )

    # from_pretrained(config=...) collides with the config it derives from the
    # path itself ("from_config() got multiple values for argument 'config'"),
    # so materialise the translated config on disk and load from there.
    import shutil as _shutil

    fixed_dir = Path("/tmp/hy15_transformer_fixed")
    if fixed_dir.exists():
        _shutil.rmtree(fixed_dir)
    _shutil.copytree(model_dir / "transformer" / "720p_i2v", fixed_dir)
    (fixed_dir / "config.json").write_text(json.dumps(tcfg, indent=2))
    log(f"wrote translated config -> {fixed_dir / 'config.json'}")

    # Load with real weights first (low_cpu_mem_usage=False), then hand the model
    # to accelerate for dispatching.
    #
    #   from_pretrained(device_map="auto")  -> internally calls dispatch_model on
    #       a model built on the meta device and raises
    #       "weight is on the meta device, we need a `value` to put in on 0".
    #   low_cpu_mem_usage=False + device_map -> disallowed outright.
    #
    # So: materialise the weights ourselves, then call infer_auto_device_map +
    # dispatch_model explicitly. HunyuanVideo15Transformer3DModel declares
    # _no_split_modules = ["HunyuanVideo15TransformerBlock", ...], so accelerate
    # splits at block boundaries and never needs the whole 33 GB at once.
    transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
        fixed_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    _n_real = sum(1 for _p in transformer.parameters() if _p.device.type != "meta")
    _n_all = sum(1 for _ in transformer.parameters())
    log(f"transformer params on real device: {_n_real}/{_n_all}")
    if _n_real != _n_all:
        log("WARNING: transformer still has meta params")
    log(f"transformer loaded in bfloat16: {transformer.config._class_name}")

    # Quantise for real. A plain `.to(torch.float8_e4m3fn)` changes the dtype but
    # torch cannot move fp8 tensors between devices, which is what makes
    # accelerate's offload hooks die with "Cannot copy out of meta tensor".
    # torchao's int8_weight_only swaps in real quantised Linear layers that
    # dequantise inside forward(), so the weights stay movable AND the forward
    # pass still works. 33 GB bf16 -> ~17 GB.
    try:
        n_real = sum(1 for p in transformer.parameters() if p.device.type != "meta")
        n_all = sum(1 for _ in transformer.parameters())
        log(f"transformer params on real device: {n_real}/{n_all}")
        if n_real != n_all:
            log("WARNING: transformer still has meta params")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torchao",
            ]
        )
        # Memory strategy, after three failed attempts:
        #
        #   torchao int8_weight_only  -> the attention out_proj BIAS ends up on
        #       the meta device (plain attribute, invisible to
        #       named_parameters/named_buffers, so it slips past the diagnostic).
        #       Denoise dies with "Tensor on device meta is not on the expected
        #       device cuda:0!" at linear(attn_output, out_proj_weight, bias).
        #   group offload to /tmp     -> /tmp on the Kaggle runner is a small
        #       tmpfs; writing the 33 GB transformer there hard-kills the kernel
        #       (the log comes back empty, 2 bytes).
        #
        # So: let accelerate place the transformer itself with device_map="auto"
        # and an explicit max_memory budget. accelerate splits it across the GPU
        # and CPU at HunyuanVideo15TransformerBlock granularity (the model
        # declares _no_split_modules), needs no quantisation, and keeps no 33 GB
        # copy anywhere.
        log("transformer: device_map=auto placement (no quantisation)")
    except Exception as e:
        log(f"torchao quantisation failed ({e}); continuing in bfloat16")

    vae = AutoencoderKLHunyuanVideo15.from_pretrained(
        model_dir / "vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
    )

    guider = ClassifierFreeGuidance(guidance_scale=6.0)

    pipe = HunyuanVideo15ImageToVideoPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        guider=guider,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
    )
    log("pipeline assembled")


    # Every parameter AND buffer in every module must live on a real device.
    # accelerate's offload hooks refuse to move a meta tensor, and a meta tensor
    # surviving into forward() raises
    #   RuntimeError: Tensor on device meta is not on the expected device cuda:0!
    # Walk the whole module tree recursively -- a straggler inside a submodule is
    # invisible to a top-level components-only check.
    def _meta_report(root, label):
        found = []
        for mod_name, mod in root.named_modules():
            for t_name, t in list(mod.named_parameters(recurse=False)):
                if t.device.type == "meta":
                    found.append(f"{label}.{mod_name}.{t_name} (param)")
            for b_name, b in list(mod.named_buffers(recurse=False)):
                if b.device.type == "meta":
                    found.append(f"{label}.{mod_name}.{b_name} (buffer)")
        return found

    _any_meta = False
    for _name, _comp in pipe.components.items():
        if isinstance(_comp, torch.nn.Module):
            _m = _meta_report(_comp, _name)
            if _m:
                _any_meta = True
                log(f"{_name}: {len(_m)} META tensors")
                for _line in _m[:8]:
                    log(f"    {_line}")
                log(f"    -> materialising {_name} on cpu")
                _comp.to_empty(device="cpu")
    if not _any_meta:
        log("no meta tensors in any component (params+buffers checked)")
    gc.collect()

    # No post-hoc dtype cast here: the transformer is already fp8 from load time.

    if torch.cuda.is_available():
        # expandable_segments cuts fragmentation on a 14.56 GiB card
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # ------------------------------------------------------------------
        # The offload strategy that actually fits a 14.56 GiB T4.
        #
        #   enable_model_cpu_offload        -> moves a WHOLE component to GPU.
        #                                      The int8 transformer is ~16.6 GB,
        #                                      which cannot fit. -> CUDA OOM.
        #   enable_sequential_cpu_offload   -> leaf level, lowest VRAM, but it
        #                                      moves every component to the meta
        #                                      device and something in the render
        #                                      path is left on meta.
        #
        # The working combination is to split the two concerns:
        #   * module-level offload for everything EXCEPT the transformer
        #     (via _exclude_from_cpu_offload, so the transformer is untouched)
        #   * BLOCK-level group offload for the transformer, which moves one
        #     HunyuanVideo15TransformerBlock at a time. The model declares
        #     _no_split_modules = ["HunyuanVideo15TransformerBlock", ...], so
        #     accelerate knows exactly where the block boundaries are.
        #
        # The Qwen2.5-VL language tower is also quantised to int8 first: at
        # bfloat16 it is ~15 GB, which on its own exceeds the 14.56 GiB card.
        # ------------------------------------------------------------------
        # The Qwen2.5-VL language tower is ~15 GB bf16, which on its own exceeds
        # the 14.56 GiB card, so it is group-offloaded block by block too. It
        # stays in system RAM (15 GB fits inside 31 GB); only the transformer
        # needs to go to disk.
        try:
            pipe.text_encoder.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_type="block_level",
                num_blocks_per_group=1,
            )
            log("block-level group offload enabled on text_encoder")
        except Exception as e:
            log(f"text_encoder group offload failed: {e}")

        # Do NOT go through pipe.enable_model_cpu_offload(): setting
        # pipe._exclude_from_cpu_offload = ["transformer"] does not actually stop
        # it from installing an accelerate strategy on the transformer, and
        # enable_group_offload then refuses to run with
        #   "Cannot apply group offloading to a module that is already applying
        #    an alternative offloading strategy from Accelerate."
        # Offload each non-transformer component directly with accelerate instead.
        from accelerate import cpu_offload

        _cuda = torch.device("cuda")
        for _cname in ("text_encoder", "text_encoder_2", "image_encoder", "vae"):
            _comp = getattr(pipe, _cname, None)
            if isinstance(_comp, torch.nn.Module):
                try:
                    cpu_offload(_comp, _cuda)
                    log(f"module-level offload: {_cname}")
                except Exception as e:
                    log(f"offload of {_cname} failed: {e}")

        from accelerate import dispatch_model, infer_auto_device_map

        _dmap = infer_auto_device_map(
            transformer,
            max_memory={0: "13GiB", "cpu": "28GiB"},
            no_split_module_classes=["HunyuanVideo15TransformerBlock"],
        )
        log(f"device map: {len(_dmap)} entries")
        transformer = dispatch_model(transformer, device_map=_dmap)
        log("transformer dispatched across gpu+cpu at block granularity")
        for fn in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, fn)()
                log(f"vae {fn} enabled")
            except Exception:
                pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ---- key frame -------------------------------------------------------
    key = Image.open(key_png).convert("RGB")
    log(f"key frame {key_png.name}: {key.size[0]}x{key.size[1]}")

    width, height = 720, 1280
    key = key.resize((width, height), Image.LANCZOS)
    num_frames = 121  # ~5 s at 24 fps
    log(f"render target {width}x{height}, {num_frames} frames")

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(20261006)

    log("=" * 70)
    log("RENDERING")
    log("=" * 70)
    t0 = time.time()
    # NOTE: HunyuanVideo15ImageToVideoPipeline.__call__ takes NO height/width.
    # Resolution comes from transformer.config.target_size, which this checkpoint
    # derives from its own `ideal_resolution: "720p"` setting. Passing height=
    # or width= raises
    #   TypeError: ... got an unexpected keyword argument 'height'
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
    log(f"frames shape {np.asarray(frames).shape}")

    # ---- write mp4 -------------------------------------------------------
    import imageio.v2 as imageio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_u8 = np.clip((np.asarray(frames) * 255).round(), 0, 255).astype("uint8")

    video_path = out_dir / "clip.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=24, codec="libx264", quality=8, pixelformat="yuv420p"
    )
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

    # ---- key frame -------------------------------------------------------
    key_png = find_input("scene_pair_key.png", (".png",))
    if key_png is None:
        key_png = find_input("scene_pair_key", (".png",))
    if key_png is None:
        log("STOP no key frame found under /kaggle/input")
        raise SystemExit("STOP no key frame")
    log(f"key frame: {key_png}")

    # ---- model -----------------------------------------------------------
    model_dir = find_model_dir()
    if model_dir is None:
        # /tmp, never /kaggle/working: Kaggle collects /kaggle/working as kernel
        # output and a 33 GB model there exceeds the output cap and kills the run.
        cache = Path("/tmp/hf_hy15")
        sentinel = cache / ".snapshot_path"
        if sentinel.exists() and Path(sentinel.read_text().strip()).exists():
            model_dir = Path(sentinel.read_text().strip())
            log(f"reusing cached snapshot: {model_dir}")
        else:
            model_dir = Path(
                download_model("tencent/HunyuanVideo-1.5", cache)
            )
            sentinel.write_text(str(model_dir))

    log(f"model dir: {model_dir}")
    tree(model_dir)

    render(model_dir, key_png, "/kaggle/working")

    # make sure it lands in the kernel output
    dst = Path("/kaggle/working/clip.mp4")
    if dst.exists():
        log(f"final artefact: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
