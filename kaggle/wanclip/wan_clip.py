"""Wan 2.2 image-to-video clip generator for Kaggle.

WHY THIS EXISTS
---------------
Pixazo's wallet emptied (HTTP 402) and the user ruled out paying. Google Flow
is free and good but a web UI, which breaks the zero-manual-work constraint.
Kaggle is the remaining route that is free, card-free and scriptable: push a
kernel with `kaggle kernels push`, poll `kaggle kernels status`, download with
`kaggle kernels output`.

WHAT THE PROBES ESTABLISHED (2026-10-05)
----------------------------------------
The first two kernels ran with `enable_gpu: true` and `machine_shape:
NvidiaTeslaT4` recorded, yet nvidia-smi was absent and torch was 2.11.0+cpu.
They got 4 CPUs, 31 GB RAM and 8 TB disk - which Kaggle users identify as the
CPU-only profile. GPU requires phone verification, and until that is done every
push silently lands on CPU. So this script reports the accelerator loudly at
the top: if it says CPU, stop and read that line rather than waiting an hour
for a clip that will not finish.

The default Kaggle image ships a CPU-only torch, so CUDA torch is installed
here. Internet is off by default; `enable_internet` is set in the metadata but
still failed on the probes, so the wheels in mdh-video-deps are the fallback.

WHAT IT DOES
------------
Reads a composed key frame from /kaggle/input, runs Wan 2.2 image-to-video with
a MOTION-ONLY prompt, and writes the clip to /kaggle/working so
`kaggle kernels output` can fetch it.

The prompt must describe motion and nothing else. Re-describing what is already
in the frame is the single largest documented cause of image-to-video drift -
fal.ai's own guide calls it "the most common mistake". The key frame already
carries the characters, the room and the props; the model only needs to know
what moves.
"""
import glob
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"


def say(*a):
    print(" ".join(str(x) for x in a), flush=True)


# ---------------------------------------------------------------- accelerator
def report_accelerator():
    import shutil
    import torch
    say("ACCEL torch", torch.__version__, "cuda", torch.cuda.is_available(),
        "devices", torch.cuda.device_count())
    # nvidia-smi is absent on a CPU session, and subprocess.run raises
    # FileNotFoundError rather than returning an error - which is what crashed
    # the first push. Check for the binary before calling it.
    smi_bin = shutil.which("nvidia-smi")
    if smi_bin:
        smi = subprocess.run([smi_bin, "--query-gpu=name,memory.total",
                              "--format=csv"], capture_output=True, text=True)
        say("ACCEL nvidia-smi:", (smi.stdout or smi.stderr).strip() or "ABSENT")
    else:
        say("ACCEL nvidia-smi: BINARY NOT PRESENT (CPU session)")
    if not torch.cuda.is_available():
        say("ACCEL VERDICT: CPU-ONLY SESSION - GPU was not attached.")
        say("ACCEL Kaggle gives the CPU profile until the account is phone")
        say("ACCEL verified. Verify at kaggle.com -> Settings -> Phone")
        say("ACCEL Verification, then re-push. GPU quota showing 30h does NOT")
        say("ACCEL mean access is granted.")
    else:
        say("ACCEL VERDICT: GPU", torch.cuda.get_device_name(0),
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")


# ------------------------------------------------------------------ torch cuda
def ensure_cuda_torch():
    """Install a CUDA build if the image shipped a CPU-only one."""
    import torch
    if torch.cuda.is_available() or "+cpu" not in torch.__version__:
        return
    say("TORCH installing CUDA build (image shipped CPU-only)")
    for cmd in (["pip", "install", "-q", "torch",
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                ["pip", "install", "-q", "--no-index", "--find-links",
                 "/kaggle/input/mdh-video-deps", "torch"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        say("TORCH", cmd[3], "rc", r.returncode, (r.stderr or "")[-300:])
        if r.returncode == 0:
            break


# ------------------------------------------------------------------- deps
def ensure_deps():
    """Install the pre-staged wheels from the dataset, offline-first."""
    wheels = glob.glob("/kaggle/input/mdh-video-deps/*.whl")
    if not wheels:
        say("DEPS no wheels in dataset; relying on internet")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "diffusers", "transformers", "accelerate",
                        "sentencepiece", "ftfy", "einops"], timeout=1800)
        return
    say("DEPS installing", len(wheels), "staged wheels")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--no-index", "--find-links",
                        "/kaggle/input/mdh-video-deps", "diffusers",
                        "transformers", "accelerate", "sentencepiece",
                        "ftfy", "einops", "imageio", "imageio-ffmpeg",
                        "av"], capture_output=True, text=True, timeout=1800)
    say("DEPS rc", r.returncode, (r.stderr or "")[-300:])


# ------------------------------------------------------------------ locate
def find_model_dir():
    """Find the Wan weights, whether attached as a model or a dataset."""
    for p in sorted(glob.glob("/kaggle/input/**/model_index.json",
                              recursive=True)):
        say("MODEL found via model_index.json", p)
        return os.path.dirname(p)
    for pat in ("/kaggle/input/models/*/wan*", "/kaggle/input/*/wan*"):
        for p in sorted(glob.glob(pat)):
            if os.path.isdir(p):
                say("MODEL candidate", p)
                return p
    raise SystemExit("MODEL not found under /kaggle/input")


def find_keyframe():
    for pat in ("/kaggle/input/**/scene_pair_key.png",
                "/kaggle/input/**/*key*.png",
                "/kaggle/input/**/*.png"):
        for p in sorted(glob.glob(pat, recursive=True)):
            say("KEY", p)
            return p
    raise SystemExit("no key frame under /kaggle/input")


# -------------------------------------------------------------------- render
def render(key_path, model_dir, prompt, out_path,
           seconds=5, fps=24, steps=30, guidance=5.0, seed=7):
    import torch
    from diffusers import WanPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    say("RENDER loading", model_dir)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = WanPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()      # keep VRAM flat on a 16 GB card
    say("RENDER pipeline", type(pipe).__name__, dtype)

    img = Image.open(key_path).convert("RGB")
    say("RENDER key", img.size)

    n_frames = int(seconds * fps) + 1
    gen = torch.Generator("cpu").manual_seed(seed)
    t0 = time.time()
    out = pipe(
        image=img,
        prompt=prompt,
        negative_prompt=(
            "camera zoom, camera pan, camera orbit, dolly, push in, pull out, "
            "tilt, whip pan, jump cut, flicker, morphing, extra limbs, "
            "deformed hands, text, watermark, subtitles"),
        height=img.size[1] - img.size[1] % 8,
        width=img.size[0] - img.size[0] % 8,
        num_frames=n_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    ).frames[0]
    say("RENDER done in", round(time.time() - t0, 1), "s frames", len(out))
    export_to_video(out, out_path, fps=fps)
    say("RENDER wrote", out_path, os.path.getsize(out_path), "bytes")
    return out_path


def main():
    spec = {"prompt": "Hoji and Mariyam stand in the playroom, waving their "
                      "hands and bouncing gently, looking at each other and "
                      "laughing. The room stays still.",
            "key": "scene_pair_key.png", "seconds": 5, "seed": 7}
    spec_path = "/kaggle/input/mdh-keyframes/job.json"
    if os.path.exists(spec_path):
        spec.update(json.load(open(spec_path)))
    say("JOB", json.dumps(spec))

    report_accelerator()
    # Stop before spending an hour. Video diffusion on CPU is not merely slow,
    # it will not finish inside a Kaggle session, and a silent CPU session looks
    # identical to a working one until it is far too late.
    import torch
    if not torch.cuda.is_available():
        say("STOP CPU-only session. Kaggle attaches the CPU profile until the")
        say("STOP account is phone verified: kaggle.com -> Settings -> Phone")
        say("STOP Verification. Showing 30h of GPU quota does NOT grant access.")
        raise SystemExit("STOP no GPU")
    ensure_deps()
    ensure_cuda_torch()
    report_accelerator()

    model_dir = find_model_dir()
    key_path = find_keyframe()
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, "clip.mp4")
    render(key_path, model_dir, spec["prompt"], out,
           seconds=spec.get("seconds", 5), seed=spec.get("seed", 7))
    say("DONE", out)


if __name__ == "__main__":
    main()
