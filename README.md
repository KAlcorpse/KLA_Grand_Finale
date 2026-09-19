# Initial submission — restoration of degraded SEM images

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Checkpoint: `models/best.pth` = `runs/r7_combined_scratch/best.pth` (this
project's `NOTES.md`) — a MiniRestormer (6.81M params) trained from scratch
on the union of this phase's 1197 pairs and the previous phase's 4785 pairs
(5862 train images total, 90/10 split held out from this phase's own data
only). `run.py` is the previous phase's submission plus an optional TensorRT path
(see "TensorRT engine" below); with no engine, or with `--backend torch`, it
imports only `torch`/`numpy` exactly as before.

## What's validated, and what isn't

**128→256: fully validated.** Beats the previous phase's submitted model on
all three official metrics on this phase's 120-image held-out split, every
CI excluding zero:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| previous phase's model (zero-shot on this data) | 23.991 | 0.6384 | 0.1531 |
| **this submission** | **24.039** | **0.6408** | **0.1493** |

Also checked against an independently-built 100-image OOD probe
(`final_test/`, fresh NFFA-EUROPE content across 7 degradation-severity
profiles) — roughly tied with the previous model on PSNR/SSIM, ahead on
LPIPS on most profiles, including the hardest one. Full detail, every
ablation tried and rejected, and every number's provenance are in
`../NOTES.md`.

**256→512: architecturally supported, NOT quality-validated.** `run.py`
handles any input size (reflect-pads to a multiple of 8, MDTA's channel
attention has no fixed spatial size), and a shape-only smoke test (feeding
256×256 images and confirming exactly 512×512 comes out, no crash) passes.
But **no real 256→512 training or test pairs exist for either phase**, so
there is no accuracy number for this scale — only the previous phase's
finding that size sensitivity was small (0.264 dB over a 4x range) and
favoured *larger* inputs, which is encouraging but not a measurement at this
specific scale.

## Known limitation, inherited and unresolved

Same as the previous phase: handed a *clean* low-resolution input, the model
denoises anyway and loses to plain bicubic (`../NOTES.md`, oracle-test
section). Not fixed here; not expected to matter for a task where inputs are
degraded by construction.

## TensorRT engine (optional) and automatic fallback to the `.pth`

`run.py` can run the network through a TensorRT FP16 engine instead of PyTorch.
It is optional and built so that a missing or bad engine cannot make a run fail:
if anything about the engine is wrong, `run.py` prints a `[trt]` warning and
processes those images with `models/best.pth`, exactly as before. (A native crash
inside TensorRT itself cannot be caught from Python; `--backend torch` never loads
TensorRT at all.)

### What ships

| file | what it is |
|---|---|
| `models/model_fp16_128.onnx` | portable, self-contained FP16 ONNX (dynamic batch, 128x128 input). The source every engine is built from |
| `models/model_fp16_128_sm86_trt11.3.0.99.engine` | built on the dev RTX 3050 (SM 8.6), batch 1 to 32, 20 MB. Loads only on SM 8.6 with TensorRT 11.3.0.99 |
| `models/model_fp16_128_ampere_plus_trt11.3.0.99.engine` | the same network built with TensorRT hardware compatibility (`AMPERE_PLUS`), 59 MB, meant to load on any Ampere-or-newer GPU including an H100 with the same TensorRT version. **Not verified on an H100.** On the 3050 it loads, matches the checkpoint (mean absolute difference 2.8e-4) and runs as fast as the native engine (19.1 to 19.2 ms per image against 19.4 to 20.5 over 480 images). |
| `trt_backend.py` | export, build and load code (`python trt_backend.py --help`) |
| `smoke_test.py` | checks the engine path and every fallback |

**Only 128x128 inputs (the validated 128 to 256 scale) have an engine.** An engine
built the same way for 256x256 disagreed with the checkpoint by about 4e-3 mean
absolute error, roughly 40 times the 128x128 engine's 3e-4 (both measured against
FP32, on real and synthetic inputs), so `run.py`'s self-check rejected it. The
cause was not investigated, so no 256x256 engine or ONNX is shipped and 256x256
inputs, like every other size, run on the `.pth`.

An engine reserves activation memory for its largest batch (about 36 MiB per 128x128
image, so 1160 MiB for batch 32). Both shipped engines cover batch 1 to 32, matching
`run.py`'s fixed batch. When `build` is run without `--max-batch` it sizes the profile from
the GPU's free memory instead (16 on the 4 GiB RTX 3050, 32 on an H100); the shipped 3050
engine was built with `--max-batch 32` explicitly.

### Why there is no H100 engine here

A TensorRT engine is compiled for one GPU architecture and one TensorRT version:
an SM 8.6 engine does not load on an H100 (SM 9.0), and no H100 was available.
The H100 engine therefore has to be built on the H100, from the ONNX (87 seconds on
the RTX 3050):

```bash
pip install tensorrt-cu13==11.3.0.99         # or the -cu12 build matching that machine's CUDA
python trt_backend.py build --size 128       # -> models/model_fp16_128_sm90_trt11.3.0.99.engine
python smoke_test.py --input-dir <dir-of-.npy-inputs>
```

If the machine has a different TensorRT version, the shipped engines are simply
not used (the version is part of the file name) and `build` makes matching ones.
The ONNX does not depend on the TensorRT version. TensorRT 11 has no FP16 builder
flag, so the ONNX itself is written in FP16: convolutions and matmuls in FP16,
LayerNorm statistics, `F.normalize`, the attention softmax and the final add in
FP32 (the same split `torch.autocast` makes).

### How run.py chooses

| Input size | Backend |
|---|---|
| 128x128 (the validated 128 to 256 scale) | TensorRT engine if one loads, passes the self-check, and there are at least 200 images (or `--backend trt`); otherwise the `.pth` |
| 256x256 (256 to 512), and every other size | always the `.pth` |

The steps, per input size:

1. `--backend torch` never touches TensorRT. `--fp32` and `--device cpu` also mean `.pth`.
2. `--backend auto` (default, or `$KLA_BACKEND`) tries an engine only from
   `TRT_MIN_IMAGES = 200` images up. `--backend trt` ignores that threshold.
3. Engine lookup: the native engine for this exact GPU and TensorRT version, then
   the `ampere_plus` engine, then (only with `--trt-build auto` or
   `$KLA_TRT_BUILD=auto`) a fresh build from the ONNX, cached in `models/` or
   `~/.cache/kla_trt_engines`. A build takes 1 to 3 minutes and is deliberately
   not the default, because it would dwarf the whole inference in a timed run.
4. Checks on every run: no `tensorrt`, no engine, a corrupt or wrong-version
   engine, or an input size without an engine all fall back to the `.pth`. The
   first batch is cross-checked against the checkpoint (mean absolute difference
   must be under 3e-3; the engine measures about 5e-4), and every batch is checked
   for NaN/Inf. Any failure switches that size to the `.pth` for the rest of the run.
5. The summary printed at the end says which backend processed how many images.

### Batch size

The batch size is fixed at **32** (`DEFAULT_BATCH` in `run.py`) for both the TensorRT and
the `.pth` path; `--batch N` changes it. The TensorRT path runs in chunks of at most the
engine's largest batch (32 for both shipped engines). If a batch runs out of GPU memory it
is halved until it fits, so a smaller GPU degrades gracefully instead of failing: 40 images
at 256x256 (no engine, about 4 times the memory of 128x128, over 5 GB at batch 32) finished
correctly on the 4 GiB RTX 3050 this way, only slower. `--batch 0` restores the old
estimate from free GPU memory, `min(256, 0.5 x free VRAM / (4096 bytes x input pixels))`,
which by that formula gives 21 on the RTX 3050 and 256 on an 80 GB H100 at 128x128 (the
H100 value is from the formula, not a measurement).

Measured GPU memory of the `.pth` path (fp16, 128x128, one fresh process per row):

| batch | tensors peak | whole process (nvidia-smi) |
|--:|--:|--:|
| 1 | 72 MB | **216 MB** (26 MB of it weights) |
| 8 | 363 MB | 648 MB |
| 16 | 691 MB | 1146 MB |
| 32 | 1349 MB | 2158 MB |

Each extra image adds about 41 MB of tensors; batch 32 (2.2 GB) fits the 4 GiB RTX 3050 with
the desktop running. On the 3050 the speed is the same for every batch size (the GPU is the limit). TensorRT memory follows the engine's batch profile,
not the batch actually used: a static batch-1 engine needs 36 MiB of scratch (288 MiB for
the process), the shipped batch 1 to 32 engines reserve about 1160 MiB.

### What "FP16" means here

It is not INT8-style quantization: there are no integer weights, scale factors or
calibration data. The model was trained in FP32 (no AMP) and `models/best.pth` holds FP32
weights.

- **PyTorch path (default):** `torch.autocast(fp16)`, i.e. automatic mixed precision at
  inference. The weights stay FP32 in memory and are cast to FP16 per operation;
  convolutions and matmuls run in FP16 while the LayerNorm statistics, `F.normalize`,
  the attention softmax and the final add stay FP32. `--fp32` turns it off.
- **TensorRT path:** the same split, written into the ONNX ahead of time (TensorRT 11
  has no automatic FP16 switch). Conv and matmul weights are rounded from FP32 to FP16
  once; the FP32 operations stay FP32. A pure FP32 engine and a BF16 engine were also
  measured and not shipped (see the table below).

Rounding to FP16 changes the output by about 2e-4 to 3e-4 on [0,1] images; the effect on
the metrics is in the quality table below.

### Measured on the RTX 3050 (4 GiB laptop GPU, 30 W cap)

| | PyTorch FP16 | TensorRT FP16 |
|---|--:|--:|
| latency, batch 1, kernel time only (interleaved runs, fixed batch-1 engine from `../trt/`) | 21.07 ms | 17.03 ms (1.24x) |
| whole `run.py` (batch 32), 30 images | 2.95 s | 3.58 s |
| whole `run.py` (batch 32), 120 images | 5.21 s | 4.91 s |
| whole `run.py` (batch 32), 480 images | 13.91 s | 11.71 s |
| whole `run.py` (batch 32), 1200 images | 30.25 s | 25.06 s |

TensorRT saves about 4.5 ms per image but costs about 0.5 s more up front (import,
engine load, self-check), so on this GPU it breaks even at roughly 100 to 130 images and is
15 to 17 percent faster at 480 to 1200. **On an H100 the crossover is unmeasured
and probably higher**, because a faster GPU saves fewer milliseconds per image
against the same setup cost. Measure it there before relying on the default:

```bash
python run.py <inputs> out_torch --backend torch
python run.py <inputs> out_trt   --backend trt      # compare the two TOTAL lines
```

and change `TRT_MIN_IMAGES` in `run.py` if the crossover differs.

Quality through `run.py`, 120 held-out images, official scorer:

| | PSNR | SSIM | LPIPS |
|---|--:|--:|--:|
| PyTorch FP16 (`--backend torch`) | 24.039 | 0.6408 | 0.1493 |
| TensorRT FP16 (`--backend trt`) | 24.042 | 0.6408 | 0.1496 |

The paired differences are +0.003 dB PSNR, 0.0000 SSIM and +0.0003 LPIPS
(the LPIPS interval excludes zero, but the change is 0.2 percent).

Other precisions were measured on the 3050 and not shipped: TensorRT FP32 is
24.75 ms (PyTorch FP32 34.39 ms), and TensorRT BF16 is 75.2 ms, slower than
PyTorch BF16 (26.6 ms) because TensorRT has no native BF16 convolution kernels for
SM 8.6. INT8 was not evaluated. The full benchmark is in `../trt/results/`.

### Testing it

```bash
python smoke_test.py --input-dir <dir-of-.npy-inputs> [--with-build]
```

runs `run.py` in ten scenarios (after a plain PyTorch reference run) and checks which
backend processed the images and that the output matches the PyTorch path: engine
forced; `auto` below and above the threshold; forced self-check failure; only the
`ampere_plus` engine present; corrupt engine; missing engine; `tensorrt` not
importable; 128, 256 and 100 pixel images in one directory; and (with
`--with-build`) an on-the-fly build of the engine. All eleven runs (the reference plus ten scenarios) passed on the RTX 3050 (Python 3.12, torch 2.13.0+cu132, TensorRT 11.3.0.99).

## Reproducing the checkpoint

```bash
cd train
env EPOCHS=36 FULLRES_EPOCHS=8 SKIP=learned W_LPIPS=0.10 SYNTH_PROB=0.0 \
    MAX_HOURS=4.0 WORKERS=6 \
    EXTRA_GT_DIR=<previous-phase-GT-dir> EXTRA_NOISY_DIR=<previous-phase-NoisyLR-dir> \
    python train.py <this-phase-GT-dir> <this-phase-NoisyLR-dir> --out-dir runs/r7_combined_scratch
```

178.7 min on an RTX 3050 Laptop (4 GiB). `EXTRA_GT_DIR`/`EXTRA_NOISY_DIR`
append the previous phase's 4785 pairs to the train split only — the 120-pair
val split (90/10 of this phase's own 1197 pairs, `SPLIT_SEED=42`) is
untouched, so the number above is not inflated by that extra data. Every
other setting is an environment-variable override (see `train.py`'s own
config block); no source edit is needed to reproduce this run or vary it.

| | |
|---|---|
| architecture | `dim=64`, blocks `(2,2,2)`, heads `(1,2,4)`, wide head 32, learned skip — 6.81M params |
| input | 64×64 random crops of NoisyLR paired with the corresponding 128×128 of GT, last 8 of 36 epochs at full resolution |
| loss | `0.50·Charbonnier + 0.35·(1−SSIM) + 0.15·gradient + 0.10·LPIPS` |
| optimiser | AdamW, lr 1e-4, weight decay 1e-4, 3-epoch warmup then cosine to 1e-6 |
| stabilisers | weight EMA 0.999, gradient clipping 1.0, loss-spike rejection |
| selection | best held-out `SSIM − LPIPS`, raw and EMA both checked each epoch |

`train/model.py` recovers the architecture from a checkpoint's own weight
shapes at load time (`config_from_state_dict`), so a checkpoint can never be
silently loaded into the wrong network — same mechanism `run.py` uses.

## Repository layout

```
run.py                    inference entry point: run.py <input-dir> <output-dir>
trt_backend.py            optional TensorRT export / build / load (imported lazily)
smoke_test.py             checks the TensorRT path and every fallback
requirements.txt
README.md                 this file
models/
  __init__.py
  minirestormer.py         the architecture; imports only torch
  best.pth                 submitted checkpoint, 6.81M parameters
  model_fp16_128.onnx      portable FP16 ONNX, 128x128 input (source of the engines)
  model_fp16_128_sm86_trt11.3.0.99.engine          RTX 3050 engine (SM 8.6 only)
  model_fp16_128_ampere_plus_trt11.3.0.99.engine   portable-across-Ampere+ engine (unverified on H100)
train/
  train.py                 reproduces the submitted checkpoint (command above)
  dataset.py                paired loader, RAM cache, crops, dihedral augmentation
  degrade.py                 the fitted forward degradation model (this phase's data)
  model.py                   same architecture as models/minirestormer.py
```

## Status

Initial submission for tonight. Two further directions are mid-investigation
and not yet in this checkpoint — see `../NOTES.md`'s open items (structured
reparameterization; a real 256→512 data source). ONNX Runtime's CUDA execution provider measured *slower* than this
submission's plain PyTorch + fp16-autocast pipeline on our dev hardware
(`../NOTES.md`, ONNX section) and is not used. TensorRT, built from an ONNX of the
same network, is a different runtime and is included as an optional path (see
"TensorRT engine" above); it has not been run on an H100.
