# Initial submission — restoration of degraded SEM images

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Checkpoint: `models/best.pth` = `runs/r7_combined_scratch/best.pth` (this
project's `NOTES.md`) — a MiniRestormer (6.81M params) trained from scratch
on the union of this phase's 1197 pairs and the previous phase's 4785 pairs
(5862 train images total, 90/10 split held out from this phase's own data
only). `run.py` is unmodified from the previous phase's submission
(self-contained, imports only `torch`/`numpy`, no source edits needed).

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
requirements.txt
README.md                 this file
models/
  __init__.py
  minirestormer.py         the architecture; imports only torch
  best.pth                 submitted checkpoint, 6.81M parameters
train/
  train.py                 reproduces the submitted checkpoint (command above)
  dataset.py                paired loader, RAM cache, crops, dihedral augmentation
  degrade.py                 the fitted forward degradation model (this phase's data)
  model.py                   same architecture as models/minirestormer.py
```

## Status

Initial submission for tonight. Two further directions are mid-investigation
and not yet in this checkpoint — see `../NOTES.md`'s open items (structured
reparameterization; a real 256→512 data source). ONNX export was tried and
rejected: ONNX Runtime's CUDA execution provider measured *slower* than this
submission's plain PyTorch + fp16-autocast pipeline on our dev hardware
(`../NOTES.md`, ONNX section) — kept as a documented negative result, not
included here.
