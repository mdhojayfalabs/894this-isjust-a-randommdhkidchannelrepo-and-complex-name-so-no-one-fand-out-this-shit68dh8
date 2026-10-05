# Kaggle GPU route — status and what it proved

## What is here
- `wanclip/` — the kernel: Wan 2.2 TI2V-5B image-to-video, motion-only prompt.
- `keyframes/` — the dataset carrying the composed two-character key frame and
  the job spec the kernel reads.

## The plumbing works, end to end
`kaggle kernels push` → `kaggle kernels status` → `kaggle kernels output` all
run from this sandbox with the stored token. Kernel logs parse cleanly. The
kernel reads its job spec and key frame from the attached dataset, finds the
model weights, and exits with a diagnosis instead of hanging.

## The blocker: no GPU is ever attached
Three pushes with `enable_gpu: true` and `machine_shape: NvidiaTeslaT4`
recorded all landed on the CPU profile: `torch 2.11.0+cpu`, `cuda_available
false`, `nvidia-smi` binary absent, 4 CPUs, 31 GB RAM, 8 TB disk.

That profile is what Kaggle gives an account that has not been phone verified.
Showing 30 h of GPU quota is not the same as having GPU access — the quota is
displayed to every account. Kaggle's own docs and support threads state the
accelerator option is gated behind phone verification.

**Fix, one time, in a browser:** kaggle.com → Settings → Phone Verification.
Then re-push `wanclip/`.

## Two other things the probes settled
- The default Kaggle image ships a **CPU-only torch**, so a CUDA build must be
  installed in the kernel. `ensure_cuda_torch()` does it.
- **Internet is off** even with `enable_internet: true`. The wheels in the
  `mdh-video-deps` dataset are the offline fallback.

## Corrected assumptions
Earlier notes said Kaggle gives 16 GB RAM and 20 GB disk. The measured profile
is **31 GB RAM and 1.1 TB free disk**. RAM was never the constraint — VRAM is,
and a T4 has 16 GB. So Wan 2.2 TI2V-5B (~8 GB) fits comfortably; the A14B
variants (~24 GB FP8) do not fit natively and need offload.
