#!/usr/bin/env python3
"""Standalone inference for the KLA Phase 2 restoration task.

    python run.py <input_dir> <output_dir> [--ckpt models/best.pth]

Reads every degraded image in <input_dir>, restores it at 2x resolution and
writes the result into <output_dir> under the same filename.

End-to-end wall time is scored, so the pipeline is shaped by three measured
facts rather than by guesswork:

  * The network costs ~22 ms/image at 128->256 on an RTX 3050, i.e. roughly
    2-3 ms on an H100. For a test set of a few hundred images the fixed cost of
    `import torch` plus CUDA context creation is LARGER than all of the
    inference put together.
  * So: nothing here imports torchvision, lpips or skimage (those are training
    and scoring dependencies, and cost ~1-2 s of startup); there is no
    torch.compile (30-60 s of compile time to save well under 1 s of compute);
    and file reads run on worker threads that are started BEFORE torch is
    imported, so disk I/O overlaps CUDA initialisation.
  * cuDNN autotuning (`torch.backends.cudnn.benchmark`) is chosen from the input
    count, not pinned. The usual advice is to enable it for fixed input shapes,
    but that is wrong for a short-lived process: the autotune cost is paid once
    and only amortises over a very large run. Swept on an RTX 3050, one fresh
    process per row, the same pixels, only N changing:

        N        inference s          observed ms/img
                 off      on          off      on     on/off
          128    3.66   29.80        28.6   232.8      8.14x
          479   12.70   42.69        26.5    89.1      3.36x
         1916   48.36   63.10        25.2    32.9      1.30x
         5748  140.66  153.16        24.5    26.7      1.09x

        least squares:  off = 1.04 s + 24.33 ms/img   (residual RMS 0.44 s)
                        on  = 27.6 s + 21.58 ms/img   (residual RMS 3.85 s)

    So autotuning does find better kernels -- 11% faster steady state -- but it
    costs ~26.5 s up front and only repays that at ~9,650 images. It is still
    losing at 5,748. Hence BENCHMARK_MIN_IMAGES = 10000: for any realistic test
    set this evaluates to False, and the constant only earns its keep if the set
    turns out to be enormous. `--cudnn-benchmark on|off` overrides.

    CAVEAT: measured on a 4 GB laptop 3050, where benchmark=True also triples
    workspace memory and can trigger the OOM batch-halving path below. An H100
    has no such pressure and trials each algorithm faster, so its true crossover
    is probably LOWER than 9,650 -- this threshold is deliberately conservative
    rather than calibrated for the scoring hardware, which we cannot measure.
  * The problem statement specifies two scales, 128->256 and 256->512. Images
    are grouped by shape so each batch is uniform, and any size is accepted:
    inputs are reflect-padded up to a multiple of 8 for the two stride-2 stages
    and cropped back afterwards.
  * TensorRT is optional (`--backend auto|trt|torch`, `--trt-build never|auto`).
    An engine is tied to one GPU architecture and one TensorRT version, so it is
    looked up by name in models/ (see trt_backend.py). If one exists it runs the
    128x128 inputs from TRT_MIN_IMAGES images up; on ANY problem --
    tensorrt missing, no/corrupt/mismatched engine, failed first-batch self-check
    against the checkpoint, NaN output, an input size with no engine -- the
    affected images run on models/best.pth instead, so the output is always
    produced. Measured on the RTX 3050 (whole run.py, val images repeated, fresh
    process per row, fp16 both):

        N        torch total s    trt total s      (fixed batch 32)
           30       2.95            3.58
          120       5.21            4.91
          480      13.91           11.71
         1200      30.25           25.06

        TensorRT saves ~4.5 ms/img and costs ~0.5 s of extra setup, so it breaks
        even at roughly 100-130 images; TRT_MIN_IMAGES = 200 leaves a margin.

    The H100 was not available: its break-even is unmeasured and probably higher
    (a faster GPU saves fewer ms per image against the same ~0.7 s of extra
    setup). Re-measure there with `--backend torch` vs `--backend trt`.
"""
import time

_T0 = time.perf_counter()

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# cuDNN autotuning pays a large one-off cost and then runs faster per image, so
# whether it is worth it depends only on how many images there are. Chosen at
# runtime from the input count. See the module docstring for the measurement.
BENCHMARK_MIN_IMAGES = 10000

# TensorRT costs a fixed startup (import + engine load + self-check) and saves
# time per image, so like cuDNN autotuning it only pays off past some image
# count. `--backend auto` uses an engine only from this many images up;
# `--backend trt` ignores it. RTX 3050 break-even measured at ~100-130 images (README);
# 200 leaves a margin. The H100 crossover could not be measured and is likely
# higher, since a faster GPU saves fewer ms per image against the same setup.
TRT_MIN_IMAGES = 200
# An engine is only trusted if, on the first batch, it agrees with the
# checkpoint's own fp16 output to this mean absolute error (images are in
# [0,1]; the shipped fp16 engine measures ~2e-4, a wrong engine is far above).
TRT_SELFCHECK_TOL = 3e-3

# Fixed batch size, used by both the TensorRT and the .pth path. Memory at 128x128 is
# ~2.2 GB for the whole process at 32 on the RTX 3050 (README), and 32 is also the
# largest batch of the shipped engines. An OOM still halves it. `--batch 0` restores
# the old free-VRAM estimate (auto_batch below).
DEFAULT_BATCH = 32

NPY_EXT = (".npy",)
IMG_EXT = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
PAD_TO = 8


# --------------------------------------------------------------------------
# I/O.  Every reader returns (float32 HxW scaled to ~[0,1], meta) and `meta`
# carries exactly what the writer needs to put the file back in its original
# form, so the output format always matches the input format.
# --------------------------------------------------------------------------
def _read(path):
    if path.lower().endswith(NPY_EXT):
        a = np.load(path)
        if a.ndim == 3:
            a = a[..., 0] if a.shape[-1] <= 4 else a[0]
        if np.issubdtype(a.dtype, np.integer):
            mx = float(np.iinfo(a.dtype).max)
            return (a.astype(np.float32) / mx), ("npy", a.dtype, mx)
        return np.ascontiguousarray(a, dtype=np.float32), ("npy", np.float32, None)

    from PIL import Image
    with Image.open(path) as im:
        fmt, a = im.format, np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    mx = float(np.iinfo(a.dtype).max) if np.issubdtype(a.dtype, np.integer) else 1.0
    return (a.astype(np.float32) / mx), ("img", a.dtype, mx, fmt)


def _write(path, arr, meta):
    if meta[0] == "npy":
        dtype, mx = meta[1], meta[2]
        np.save(path, (arr * mx).round().astype(dtype) if mx else arr.astype(np.float32))
    else:
        from PIL import Image
        dtype, mx, fmt = meta[1], meta[2], meta[3]
        out = (arr * mx).round().clip(0, mx).astype(dtype) if mx != 1.0 else arr
        Image.fromarray(out).save(path, format=fmt)


# Peak activation memory measured at ~2.8 kB per INPUT pixel (batch 8 and 16,
# 128x128 and 256x256, fp16 autocast, cudnn.benchmark off). 4 kB/px is that
# figure with headroom; half of free VRAM keeps well clear of fragmentation.
BYTES_PER_INPUT_PIXEL = 4096


def load_trt_runners(here, groups, build, backend, log, warn):
    """One TensorRT Runner per square input size that has a usable engine.

    Returns {} (=> everything runs on the .pth) on any problem. Nothing heavy is
    imported unless an engine file actually exists or a build was requested, so
    a machine with no engine pays no TensorRT startup cost.
    """
    sizes = sorted({h for (h, w) in groups if h == w})
    try:
        sys.path.insert(0, here)
        import trt_backend
        sizes = [s for s in sizes if s in trt_backend.SIZES]
        if not build:
            have = [s for s in sizes if trt_backend.has_candidates(s)]
            for s in sizes:
                if s not in have:
                    (warn if backend == "trt" else log)(
                        f"[trt] no engine for {s}x{s} on this GPU (build one: "
                        f"python trt_backend.py build --size {s}, or pass --trt-build auto); "
                        f"using the PyTorch checkpoint")
            sizes = have
        if not sizes:
            return {}
        import tensorrt  # noqa: F401
    except Exception as e:
        warn(f"[trt] TensorRT unavailable ({type(e).__name__}: {e}); using the PyTorch checkpoint")
        return {}
    if build:
        # Build every missing engine BEFORE loading any: a loaded engine holds a
        # large activation buffer, and a build needs that memory free.
        for s in sizes:
            if not trt_backend.has_candidates(s):
                try:
                    trt_backend.build_engine(s, log=log)
                except Exception as e:
                    warn(f"[trt] {s}x{s}: engine build failed ({e}); using the PyTorch checkpoint for this size")
    runners = {}
    for s in sizes:
        try:
            runners[s] = trt_backend.load_runner(s, build=build, log=log)
        except Exception as e:
            warn(f"[trt] {s}x{s}: {e}; using the PyTorch checkpoint for this size")
    return runners


def auto_batch(torch, device, pixels, requested):
    if requested:
        return max(1, requested)
    if device.type != "cuda":
        return 8
    free, _ = torch.cuda.mem_get_info()
    return int(min(256, max(1, (free * 0.5) // (BYTES_PER_INPUT_PIXEL * pixels))))


def main():
    ap = argparse.ArgumentParser(description="2x restoration of degraded SEM images")
    ap.add_argument("input_dir", help="directory of degraded input images")
    ap.add_argument("output_dir", help="directory to write restored images into")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (default: models/best.pth next to this script)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                    help=f"images per forward pass (default {DEFAULT_BATCH}, fixed for both the "
                         f"TensorRT and .pth paths; an out-of-memory error halves it). "
                         f"0 = size it from free VRAM instead")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 autocast")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--cudnn-benchmark", choices=("auto", "on", "off"), default="auto",
                    help="cuDNN autotuning. auto = enable only when there are enough "
                         "images to amortise the one-off autotune cost")
    ap.add_argument("--backend", choices=("auto", "trt", "torch"), default=None,
                    help="auto (default, or $KLA_BACKEND) = TensorRT engine when one exists for "
                         "this GPU and there are enough images, else the .pth; trt = try the "
                         "engine regardless of image count; torch = never touch TensorRT. Any "
                         "TensorRT problem falls back to the .pth automatically.")
    ap.add_argument("--trt-build", choices=("never", "auto"), default=None,
                    help="when no engine exists for this GPU: never (default, or $KLA_TRT_BUILD) "
                         "= use the .pth, auto = build one from models/model_fp16_<size>.onnx "
                         "first (1-3 min, cached for later runs)")
    ap.add_argument("--quiet", action="store_true", help="suppress the timing summary")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ckpt = args.ckpt or os.path.join(here, "models", "best.pth")
    if not os.path.isfile(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    if not os.path.isdir(args.input_dir):
        sys.exit(f"input directory not found: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(NPY_EXT + IMG_EXT))
    if not names:
        sys.exit(f"no .npy or image files found in {args.input_dir}")

    # Reads start here, on threads, and are collected only after torch has
    # finished importing and building the model -- that is the whole point.
    pool = ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4)))
    pending = [pool.submit(_read, os.path.join(args.input_dir, n)) for n in names]

    import torch
    import torch.nn.functional as F
    sys.path.insert(0, here)
    from models.minirestormer import (MiniRestormer, config_from_state_dict,
                                      load_state_dict_compat)

    t_import = time.perf_counter()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and not args.fp32
    if device.type == "cuda":
        bench = (len(names) >= BENCHMARK_MIN_IMAGES if args.cudnn_benchmark == "auto"
                 else args.cudnn_benchmark == "on")
        torch.backends.cudnn.benchmark = bench
        torch.backends.cuda.matmul.allow_tf32 = True

    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = MiniRestormer(**config_from_state_dict(sd, ck if isinstance(ck, dict) else None))
    load_state_dict_compat(model, sd)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    del ck, sd
    t_model = time.perf_counter()

    arrays, metas = [], []
    for f in pending:
        a, m = f.result()
        arrays.append(a)
        metas.append(m)
    t_read = time.perf_counter()

    # The outputs are clamped to [0,1] because ground truth lives there. If the
    # test set ever ships float arrays on a 0-255 scale instead, that clamp
    # would silently flatten every output to white, so say so loudly rather than
    # guess a rescale -- speckle legitimately pushes real inputs above 1.0 (the
    # training data reaches 2.24), so a value over 1 is NOT itself suspicious.
    hi = max(float(a.max()) for a in arrays)
    if hi > 4.0:
        print(f"WARNING: input maximum is {hi:.1f}. This pipeline expects data "
              f"scaled to [0,1] (speckle may exceed 1). Outputs are clamped to "
              f"[0,1] and will be wrong if the inputs are on a 0-255 scale.",
              file=sys.stderr)

    # Uniform batches: group by input shape, so 128x128 and 256x256 inputs in
    # the same test set never share a padded batch.
    groups = {}
    for i, a in enumerate(arrays):
        groups.setdefault(a.shape, []).append(i)

    def log(msg):
        if not args.quiet:
            print(msg, file=sys.stderr)

    def warn(msg):
        print(msg, file=sys.stderr)

    # Optional TensorRT engines, one per square input size. Anything that goes
    # wrong (no tensorrt, no engine for this GPU, wrong TensorRT version,
    # corrupt file, failed self-check, NaN output) just means that size runs on
    # the .pth checkpoint instead; the result is the same restoration either way.
    backend = args.backend or os.environ.get("KLA_BACKEND", "auto")
    build = (args.trt_build or os.environ.get("KLA_TRT_BUILD", "never")) == "auto"
    runners = {}
    if backend == "trt" or (backend == "auto" and len(names) >= TRT_MIN_IMAGES):
        if device.type != "cuda" or args.fp32:
            if backend == "trt":
                warn("[trt] TensorRT engines are fp16 and need CUDA; using the PyTorch checkpoint")
        else:
            runners = load_trt_runners(here, groups, build, backend, log, warn)
    elif backend == "auto":
        log(f"[trt] {len(names)} images < TRT_MIN_IMAGES ({TRT_MIN_IMAGES}); using the PyTorch checkpoint")
    t_trt = time.perf_counter()

    counts = {"trt": 0, "torch": 0}
    verified = set()

    def torch_forward(x):
        if device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            return model(x)

    def trt_forward(runner, x, first):
        y = runner(x)
        if not bool(torch.isfinite(y).all()):
            raise RuntimeError("engine produced NaN/Inf")
        if first:
            # Cross-check the first batch against the checkpoint itself, so a
            # stale, mismatched or mis-built engine can never silently ship.
            n = min(2, x.shape[0])
            diff = (y[:n] - torch_forward(x[:n]).float()).abs().mean(dim=(1, 2, 3)).max().item()
            if diff > TRT_SELFCHECK_TOL:
                raise RuntimeError(f"engine output differs from the checkpoint "
                                   f"(mean |diff| {diff:.2e} > {TRT_SELFCHECK_TOL:.0e})")
        return y

    writes = []
    with torch.inference_mode():
        for shape, idxs in sorted(groups.items()):
            h, w = shape
            ph, pw = (-h) % PAD_TO, (-w) % PAD_TO
            runner = runners.get(h) if h == w else None
            bs = auto_batch(torch, device, h * w, args.batch)
            if runner is not None:
                bs = min(bs, runner.max_batch)
            pos = 0
            while pos < len(idxs):
                chunk = idxs[pos:pos + bs]
                try:
                    x = torch.from_numpy(np.stack([arrays[i] for i in chunk]))
                    x = x.unsqueeze(1).to(device, non_blocking=True)
                    if ph or pw:
                        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
                    y = None
                    if runner is not None:
                        try:
                            y = trt_forward(runner, x, h not in verified)
                            verified.add(h)
                            counts["trt"] += len(chunk)
                        except Exception as e:
                            warn(f"[trt] {h}x{w}: {e}; falling back to the PyTorch checkpoint")
                            runners.pop(h, None)
                            runner = None
                            bs = auto_batch(torch, device, h * w, args.batch)
                    if y is None:
                        y = torch_forward(x)
                        counts["torch"] += len(chunk)
                    y = y.float().clamp_(0, 1)
                    if ph or pw:
                        y = y[..., : h * 2, : w * 2]
                    out = y.squeeze(1).cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    continue
                for k, i in enumerate(chunk):
                    writes.append(pool.submit(
                        _write, os.path.join(args.output_dir, names[i]), out[k], metas[i]))
                pos += len(chunk)
    t_infer = time.perf_counter()

    for f in writes:
        f.result()
    pool.shutdown(wait=True)
    t_end = time.perf_counter()

    if not args.quiet:
        n = len(names)
        print(f"{n} images -> {args.output_dir}", file=sys.stderr)
        print(f"  import+init {t_import - _T0:6.2f} s\n"
              f"  build model {t_model - t_import:6.2f} s\n"
              f"  read        {t_read - t_model:6.2f} s  (overlapped with the above)\n"
              f"  trt setup   {t_trt - t_read:6.2f} s\n"
              f"  inference   {t_infer - t_trt:6.2f} s  ({(t_infer - t_trt) / n * 1e3:.2f} ms/img)\n"
              f"  write flush {t_end - t_infer:6.2f} s\n"
              f"  TOTAL       {t_end - _T0:6.2f} s\n"
              f"  backend     TensorRT {counts['trt']} img, PyTorch {counts['torch']} img",
              file=sys.stderr)


if __name__ == "__main__":
    main()
