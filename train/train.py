import argparse
import os

# Must be set BEFORE torch initialises CUDA. Fixes the "epoch 1 fine, epoch 2
# OOM" fragmentation pattern: training crops and full-res validation allocate
# very differently-shaped blocks, and the default allocator can't reuse them.
#os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import gc
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RestorationDataset
import importlib
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# MODEL_MODULE picks the architecture file -- "model" (default, unchanged
# behaviour) or "model_2" (the one-fewer-encoder/decoder-level ablation).
# NUM_BLOCKS/NUM_HEADS below must be set to match (e.g. 2,2 / 1,2 for
# model_2's intended depth) -- this only selects which file's MiniRestormer
# gets those args, it does not change the args themselves.
_MODEL_MODULE = os.environ.get("MODEL_MODULE", "model")
_model_mod = importlib.import_module(_MODEL_MODULE)
MiniRestormer, load_state_dict_compat = _model_mod.MiniRestormer, _model_mod.load_state_dict_compat

# ---------------------------- configuration ----------------------------------
# The data directories are command-line arguments, so nothing in this file has
# to be edited to train on a different machine:
#
#     python train.py <gt-dir> <noisylr-dir>
#
# Every other value below is an environment-variable override, so an experiment
# never needs a source edit either. The defaults reproduce the submitted
# checkpoint exactly.
_e = os.environ.get


def _cli():
    ap = argparse.ArgumentParser(
        description="Train the restoration model on paired GT / NoisyLR .npy directories.")
    ap.add_argument("gt_dir", nargs="?", default=_e("GT_DIR"),
                    help="directory of clean ground-truth .npy files (256x256)")
    ap.add_argument("noisy_dir", nargs="?", default=_e("NOISY_DIR"),
                    help="directory of degraded NoisyLR .npy files (128x128)")
    ap.add_argument("--out-dir", default=_e("RUN_DIR", "runs/dim64_arch"),
                    help="where best.pth, last.pth and train_log.csv are written")
    # parse_known_args, not parse_args: importing this module must never fail,
    # and unknown flags are left for anything that wraps it.
    return ap.parse_known_args()[0], ap


_ARGS, _AP = _cli()
_DEF = os.path.expanduser("~/Downloads/Hack2/semicon_train_data")
# These were parsed from argv/env above and then overwritten by hardcoded paths,
# so the module docstring's promise was false. Now the argument wins if given.
GT_DIR = _ARGS.gt_dir or os.path.join(_DEF, "GT")
NOISY_DIR = _ARGS.noisy_dir or os.path.join(_DEF, "NoisyLR")
# Extra labelled pairs appended to the TRAIN split only; valset/ is untouched.
EXTRA_GT_DIR = _e("EXTRA_GT_DIR") or None
EXTRA_NOISY_DIR = _e("EXTRA_NOISY_DIR") or None
OUT_DIR = _ARGS.out_dir

DIM = int(_e("DIM", 64))
_h = _e("NUM_HEADS", "1,2,4")                 # int, or per-level e.g. "1,2,4"
NUM_HEADS = tuple(int(x) for x in _h.split(",")) if "," in _h else int(_h)
NUM_BLOCKS = tuple(int(x) for x in _e("NUM_BLOCKS", "2,2,2").split(","))
LOG_INPUT = _e("LOG_INPUT", "0") == "1"       # add log(x) channel
NOISE_COND = _e("NOISE_COND", "0") == "1"     # add estimated-noise-level channel
WIDE_HEAD = int(_e("WIDE_HEAD", 32))          # 0 = plain 1-channel head
SKIP = _e("SKIP", "bicubic")                  # bicubic | learned | none
# Fine-tune from an external checkpoint's WEIGHTS ONLY (no optimizer/EMA/epoch
# state -- this is a fresh run that starts from pretrained weights, not a
# resume of a previous run of this same RUN_DIR). Ignored if OUT_DIR/last.pth
# already exists, since that means this run itself is being resumed and that
# takes priority. Architecture must match ARCH below or this fails loudly.
INIT_CKPT = _e("INIT_CKPT") or None


EPOCHS = int(_e("EPOCHS", 400))       # unreachable on purpose: MAX_HOURS ends the run
MAX_HOURS = float(_e("MAX_HOURS", 2.0))
CROP = int(_e("CROP", 64))
BATCH = int(_e("BATCH", 6))           # 0 = probe; the probe does not see LPIPS memory
FULLRES_EPOCHS = int(_e("FULLRES_EPOCHS", 4))
FULLRES_BATCH = int(_e("FULLRES_BATCH", 0))       # 0 = probe
VAL_BATCH = int(_e("VAL_BATCH", 2))   # validation runs at full 256x256; keep small

LR, MIN_LR = float(_e("LR", 1e-4)), 1e-6
WEIGHT_DECAY, WARMUP = 1e-4, int(_e("WARMUP", 3))
SYNTH_PROB, SYNTH_JITTER = float(_e("SYNTH_PROB", 0.0)), 0.2
# Extra degradation on the REAL noisy input (augment.py). Distinct from
# SYNTH_PROB, which replaces the input with one synthesised from GT and has
# never beaten its control. AUG_PROB=0.0 is off and is the default, so every
# earlier run reproduces unchanged.
AUG_PROB = float(_e("AUG_PROB", 0.0))
AUG_STRENGTH = float(_e("AUG_STRENGTH", 0.6))
AUG_MODE = _e("AUG_MODE", "fitted")   # fitted | wide
VAL_SPLIT, WORKERS = 0.1, int(_e("WORKERS", 2))
SEED = int(_e("SEED", 42))    # training randomness: init, shuffle, crops, augment
# The train/val split seed is deliberately NOT env-driven. valset/ -- the 479
# pairs every score in NOTES.md is computed on -- was cut with 42, so varying it
# moves held-out images into training and inflates every metric silently.
SPLIT_SEED = 42
EMA_DECAY, CLIP_GRAD = 0.999, 1.0
# Mixed-precision training (autocast fp16 forward/backward + GradScaler), NOT
# a raw fp16 cast -- AdamW's small moment terms and this project's own SSIM/
# Charbonnier epsilons need fp32 master weights and loss scaling to stay
# stable. Off by default so every existing run reproduces unchanged; when off,
# GradScaler(enabled=False) is a documented no-op, so the code path below is
# identical to plain fp32 training either way. Tried because this GPU is
# memory-bound (full-res batch probes to 1), and AMP roughly halves activation
# memory -- the hoped-for win is a bigger FULLRES_BATCH, not raw FLOPs.
USE_AMP = _e("USE_AMP", "0") == "1"

# Loss weights. SSIM rewards blur (blurring bicubic took SSIM 0.831 -> 0.887),
# so it is down-weighted vs the old 0.5; gradient and LPIPS push the other way.
W_CHAR = float(_e("W_CHAR", 0.50))
W_SSIM = float(_e("W_SSIM", 0.35))    # SSIM is the main blur-pusher; do not raise
W_GRAD = float(_e("W_GRAD", 0.15))
W_LPIPS = float(_e("W_LPIPS", 0.10))  # 0.40 was tested and is worse overall
# Frequency-domain term. 0.0 is off and is the default, so every earlier run
# reproduces unchanged. See freq_loss() for why this is not the HF-ratio loss
# Phase 1 falsified.
W_FREQ = float(_e("W_FREQ", 0.0))
FREQ_MODE = _e("FREQ_MODE", "ffl")    # ffl | logmag
# Which SSIM the loss optimises. "gauss11" = what every run before 2 Sep used;
# "skimage" = an exact match for the metric score.py reports.
SSIM_MODE = _e("SSIM_MODE", "gauss11")
# ------------------------------------------------------------------------------

BATCH_CANDIDATES = [16, 12, 8, 6, 4, 3, 2, 1]


def probe_batch(dim, blocks, crop, safety_step=1, use_amp=False):
    """Largest batch that survives two real optimizer steps, minus a safety step.

    Two steps, because AdamW allocates its moment buffers on the first .step().
    The step-down is deliberate: the probe runs on an empty GPU, whereas real
    training also holds the dataset cache and a fragmented heap.

    `use_amp` must match the real training loop's USE_AMP, or the probe sizes
    batches for the wrong memory footprint -- autocast+GradScaler roughly
    halves activation memory, so a probe that ignores it under-sizes the
    batch and throws away the whole point of enabling AMP.
    """
    if not torch.cuda.is_available():
        return 4
    for i, bs in enumerate(BATCH_CANDIDATES):
        model = opt = x = y = loss = scaler = None
        try:
            model = MiniRestormer(dim=dim, num_blocks=blocks, num_heads=NUM_HEADS,
                                  log_input=LOG_INPUT, noise_cond=NOISE_COND,
                                  wide_head=WIDE_HEAD, skip=SKIP).cuda()
            model = model.to(memory_format=torch.channels_last)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            x = torch.randn(bs, 1, crop, crop, device="cuda").to(memory_format=torch.channels_last)
            y = torch.randn(bs, 1, crop * 2, crop * 2, device="cuda")
            for _ in range(2):
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    loss = F.l1_loss(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            chosen = BATCH_CANDIDATES[min(i + safety_step, len(BATCH_CANDIDATES) - 1)]
            print(f"  crop {crop}: max batch {bs} (peak {peak:.2f} GiB) -> using {chosen}",
                  flush=True)
            return chosen
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" not in str(err).lower():
                raise
        finally:
            del model, opt, x, y, loss
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return 0


# ---------------- losses ----------------
def charbonnier(pred, target, eps=1e-3):
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))


class SSIM(nn.Module):
    """Two modes, because the loss and the reported metric are not the same function.

      "gauss11"  11x11 Gaussian sigma 1.5, biased covariance, zero-padded border.
                 The Wang et al. original, and what every run up to 2 Sep trained on.
      "skimage"  7x7 uniform, sample covariance (N-1), border cropped rather than
                 padded -- an exact match for skimage.metrics.structural_similarity
                 at its defaults, which is what score.py reports and therefore what
                 we are graded on. Measured gap between the two: 0.024 SSIM, ~48x
                 the seed noise floor.
    """

    def __init__(self, mode="gauss11", window=None, sigma=1.5):
        super().__init__()
        assert mode in ("gauss11", "skimage"), f"unknown SSIM mode: {mode}"
        self.mode = mode
        if mode == "gauss11":
            window = window or 11
            c = torch.arange(window, dtype=torch.float32) - window // 2
            g = torch.exp(-c ** 2 / (2 * sigma ** 2))
            g = (g / g.sum()).unsqueeze(1)
            w = (g @ g.t()).view(1, 1, window, window)
            self.pad, self.cov_norm = window // 2, 1.0
        else:
            window = window or 7
            w = torch.full((1, 1, window, window), 1.0 / window ** 2)
            npix = float(window ** 2)
            # skimage crops the border instead of padding, so a valid convolution
            # reproduces exactly the region it averages over.
            self.pad, self.cov_norm = 0, npix / (npix - 1)
        self.register_buffer("w", w)

    def forward(self, a, b):
        w, p, cn = self.w.to(a.dtype), self.pad, self.cov_norm
        mu1, mu2 = F.conv2d(a, w, padding=p), F.conv2d(b, w, padding=p)
        mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        s1 = cn * (F.conv2d(a * a, w, padding=p) - mu1_sq)
        s2 = cn * (F.conv2d(b * b, w, padding=p) - mu2_sq)
        s12 = cn * (F.conv2d(a * b, w, padding=p) - mu12)
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * mu12 + C1) * (2 * s12 + C2)) /
                ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))).mean()


def freq_loss(pred, target, mode="ffl", alpha=1.0):
    """Loss in the frequency domain, aimed at the model's known weakness: it keeps
    only ~47% of ground truth's high-frequency energy.

      "ffl"     Focal Frequency Loss (Jiang et al. 2021). Squared spectral error
                weighted by that same error, detached -- so the loss concentrates
                on whichever frequencies are currently worst, rather than being
                swamped by the DC term.
      "logmag"  L1 on log1p of the magnitude spectrum. Compresses a dynamic range
                that otherwise lets low frequencies dominate.

    NOT the HF-energy-ratio loss Phase 1 falsified before implementing: that was a
    scalar ratio which saturates, so its gradient vanished exactly where it was
    needed. Both of these have gradient at every frequency.
    """
    P = torch.fft.rfft2(pred.float(), norm="ortho")
    T = torch.fft.rfft2(target.float(), norm="ortho")
    if mode == "logmag":
        return F.l1_loss(torch.log1p(P.abs()), torch.log1p(T.abs()))
    d = (P - T).abs()
    w = d.detach() ** alpha
    w = w / w.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return (w * d ** 2).mean()


SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


def gradient_loss(pred, target):
    kx, ky = SOBEL_X.to(pred.device), SOBEL_Y.to(pred.device)
    return (F.l1_loss(F.conv2d(pred, kx, padding=1), F.conv2d(target, kx, padding=1)) +
            F.l1_loss(F.conv2d(pred, ky, padding=1), F.conv2d(target, ky, padding=1)))


_LPIPS, _LPIPS_OK = None, None


def lpips_loss(pred, target):
    """Perceptual distance. Auto-disables (returns 0) if the package or its
    pretrained weights are unavailable -- never kills an overnight run."""
    global _LPIPS, _LPIPS_OK
    if _LPIPS_OK is False:
        return torch.zeros((), device=pred.device)
    if _LPIPS is None:
        try:
            import lpips as _lp
            _LPIPS = _lp.LPIPS(net="alex", verbose=False).to(pred.device).eval()
            for p in _LPIPS.parameters():
                p.requires_grad_(False)
            _LPIPS_OK = True
            print("  LPIPS loss active (alex)", flush=True)
        except Exception as err:                                   # noqa: BLE001
            print(f"  LPIPS unavailable ({type(err).__name__}); continuing without it",
                  flush=True)
            _LPIPS_OK = False
            return torch.zeros((), device=pred.device)
    # grayscale -> 3 channels, [0,1] -> [-1,1], which is what LPIPS expects
    return _LPIPS(pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1,
                  target.repeat(1, 3, 1, 1) * 2 - 1).mean()


def hf_energy(t, cut=0.25):
    """Energy above `cut` cycles/pixel -- a direct proxy for fine texture."""
    f = torch.fft.rfft2(t.float(), norm="ortho").abs()
    H, W = t.shape[-2:]
    fy = torch.fft.fftfreq(H, device=t.device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(W, device=t.device).view(1, -1)
    return float((f * ((fy ** 2 + fx ** 2).sqrt() > cut)).sum())


# ---------------- EMA ----------------
class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.other = {k: v.detach().clone() for k, v in model.state_dict().items()
                      if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        sd = model.state_dict()
        if not all(torch.isfinite(v).all() for v in sd.values() if v.dtype.is_floating_point):
            return                      # never average in a corrupted step
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.other[k] = v.detach().clone()

    def state_dict(self):
        return {**{k: v.clone() for k, v in self.shadow.items()},
                **{k: v.clone() for k, v in self.other.items()}}

    def load_state_dict(self, sd):
        for k, v in sd.items():
            target = self.shadow if k in self.shadow else self.other
            target[k] = v.detach().clone().float() if k in self.shadow else v.detach().clone()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def _val_batches(ds, batch):
    """Yield batches of at most `batch` items that all share one shape.

    Validation runs on whole images (crop=None), so a directory mixing 256 and
    512 GT would make torch.stack raise. Flushing on a shape change keeps every
    batch uniform; with a single resolution the split points are exactly where
    they were before, so nothing about existing runs changes.
    """
    buf = []
    for i in range(len(ds)):
        item = ds[i]
        if buf and (len(buf) == batch or item[0].shape != buf[0][0].shape):
            yield buf
            buf = []
        buf.append(item)
    if buf:
        yield buf


@torch.no_grad()
def validate(model, ds, device, batch):
    """Returns PSNR, SSIM, LPIPS, and the HF ratio (texture kept vs ground truth).
    Plain loop, no DataLoader workers -- one less thing to leak or OOM."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    p = s = lp = n = 0.0
    hf_p = hf_g = 0.0
    for items in _val_batches(ds, batch):
        noisy = torch.stack([a for a, _ in items]).to(device)
        gt_t = torch.stack([b for _, b in items]).to(device)
        pred_t = model(noisy).float().clamp(0, 1)
        if W_LPIPS > 0:
            lp += float(lpips_loss(pred_t, gt_t)) * pred_t.shape[0]
        hf_p += hf_energy(pred_t.cpu())
        hf_g += hf_energy(gt_t.cpu())
        pred, gt = pred_t.cpu().numpy(), gt_t.cpu().numpy()
        for i in range(pred.shape[0]):
            p += psnr(gt[i, 0], pred[i, 0])
            s += structural_similarity(gt[i, 0], pred[i, 0], data_range=1.0)
            n += 1
        del noisy, gt_t, pred_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p / n, s / n, lp / n, hf_p / max(hf_g, 1e-9)


def main():
    if not GT_DIR or not NOISY_DIR:
        _AP.error("give both data directories: python train.py <gt-dir> <noisylr-dir>")
    for d, what in ((GT_DIR, "GT"), (NOISY_DIR, "NoisyLR")):
        if not os.path.isdir(d):
            _AP.error(f"{what} directory not found: {d}")
    print(f"data   : GT {GT_DIR}\n         NoisyLR {NOISY_DIR}", flush=True)

    global BATCH, FULLRES_BATCH
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"device={device}  out={OUT_DIR}  dim={DIM} blocks={NUM_BLOCKS} "
          f"heads={NUM_HEADS} wide_head={WIDE_HEAD} skip={SKIP} "
          f"seed={SEED} split_seed={SPLIT_SEED} noise_cond={NOISE_COND} "
          f"epochs={EPOCHS} fullres_epochs={FULLRES_EPOCHS} "
          f"synth_prob={SYNTH_PROB} aug_prob={AUG_PROB} aug_strength={AUG_STRENGTH} "
          f"ssim_mode={SSIM_MODE} w_freq={W_FREQ} freq_mode={FREQ_MODE} aug_mode={AUG_MODE} "
          f"init_ckpt={INIT_CKPT} use_amp={USE_AMP} model_module={_MODEL_MODULE}",
          flush=True)
    print(f"loss: char={W_CHAR} ssim={W_SSIM} grad={W_GRAD} lpips={W_LPIPS}", flush=True)

    # AMP's memory footprint is less predictable from a 2-step empty-GPU probe:
    # the probe already can't see LPIPS's memory (true with or without AMP),
    # and under AMP that gap cost a real mid-training OOM at the probe's
    # "safe" batch size (measured: batch 3 probed clean, crashed ~180 steps
    # into a real full-res epoch). A wider safety margin under AMP is the fix,
    # not trusting the probe further.
    _safety = 2 if USE_AMP else 1
    if BATCH == 0:
        BATCH = probe_batch(DIM, NUM_BLOCKS, CROP, safety_step=_safety, use_amp=USE_AMP)
        if BATCH == 0:
            raise SystemExit("does not fit even at batch 1 -- lower DIM or NUM_BLOCKS")
    if FULLRES_BATCH == 0 and FULLRES_EPOCHS > 0:
        FULLRES_BATCH = probe_batch(DIM, NUM_BLOCKS, CROP * 2, safety_step=_safety, use_amp=USE_AMP)
    print(f"batch={BATCH}  fullres_batch={FULLRES_BATCH}", flush=True)

    train_ds = RestorationDataset(GT_DIR, NOISY_DIR, "train", VAL_SPLIT, SPLIT_SEED,
                                  crop=CROP, synth_prob=SYNTH_PROB,
                                  synth_jitter=SYNTH_JITTER,
                                  aug_prob=AUG_PROB, aug_strength=AUG_STRENGTH,
                                  aug_mode=AUG_MODE,
                                  extra_gt_dir=EXTRA_GT_DIR,
                                  extra_noisy_dir=EXTRA_NOISY_DIR)
    val_ds = RestorationDataset(GT_DIR, NOISY_DIR, "val", VAL_SPLIT, SPLIT_SEED,
                                crop=None, augment=False)

    ARCH = dict(dim=DIM, num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS,
                log_input=LOG_INPUT, noise_cond=NOISE_COND,
                wide_head=WIDE_HEAD, skip=SKIP)
    model = MiniRestormer(**ARCH).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.3f} M params", flush=True)

    resume = os.path.join(OUT_DIR, "last.pth")
    if INIT_CKPT and not os.path.isfile(resume):
        ck_init = torch.load(INIT_CKPT, map_location=device, weights_only=False)
        sd_init = ck_init["model"] if isinstance(ck_init, dict) and "model" in ck_init else ck_init
        load_state_dict_compat(model, sd_init)
        print(f"initialised from {INIT_CKPT} (weights only; optimizer/EMA/epoch "
              f"start fresh)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ema = ModelEMA(model, EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and USE_AMP))
    ssim_fn = SSIM(mode=SSIM_MODE).to(device)

    start_epoch, best_score, best_psnr, best_ssim = 1, -1.0, 0.0, 0.0
    if os.path.isfile(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        start_epoch = ck["epoch"] + 1
        best_score = ck.get("best_score", -1.0)
        best_psnr, best_ssim = ck.get("best_psnr", 0.0), ck.get("best_ssim", 0.0)
        print(f"resumed at epoch {start_epoch}; delete {resume} to start fresh", flush=True)

    log = os.path.join(OUT_DIR, "train_log.csv")
    if not os.path.exists(log):
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "loss", "psnr", "ssim",
                                    "ema_psnr", "ema_ssim", "minutes",
                                    "lpips", "hf_ratio"])

    def make_loader(crop, bs):
        train_ds.set_crop(crop)
        return DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=WORKERS,
                          pin_memory=device.type == "cuda", drop_last=True,
                          persistent_workers=WORKERS > 0)

    crop_loader = make_loader(CROP, BATCH)
    fullres_loader = None
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        fullres = FULLRES_EPOCHS > 0 and epoch > EPOCHS - FULLRES_EPOCHS
        if fullres:
            if fullres_loader is None:
                del crop_loader
                gc.collect()
                fullres_loader = make_loader(None, max(FULLRES_BATCH, 1))
            train_loader, bs = fullres_loader, max(FULLRES_BATCH, 1)
        else:
            train_ds.set_crop(CROP)
            train_loader, bs = crop_loader, BATCH

        if epoch <= WARMUP:
            lr = LR * epoch / WARMUP
        else:
            by_epoch = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
            by_time = (time.time() - t0) / (MAX_HOURS * 3600)
            prog = min(1.0, max(by_epoch, by_time))
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        total = count = skipped = 0
        recent = []
        for noisy, gt in tqdm(train_loader, desc=f"epoch {epoch} bs={bs}", mininterval=10):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            if device.type == "cuda":
                noisy = noisy.to(memory_format=torch.channels_last)
                gt = gt.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            # AMP only wraps the model's forward -- where the memory pressure
            # actually is. Charbonnier/SSIM's small epsilon terms and LPIPS's
            # pretrained-in-fp32 network are computed outside it, in full
            # fp32, matching model.py's own LayerNorm2d/skip-connection
            # pattern of pinning numerically sensitive math to fp32 regardless
            # of the ambient precision. Measured need for this: wrapping the
            # loss too corrupted the raw model to NaN within one epoch.
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                out = model(noisy)
            out = out.float()
            gt32 = gt.float()
            loss = (W_CHAR * charbonnier(out, gt32) +
                    W_SSIM * (1 - ssim_fn(out.clamp(0, 1), gt32)) +
                    W_GRAD * gradient_loss(out, gt32))
            if W_LPIPS > 0:
                loss = loss + W_LPIPS * lpips_loss(out, gt32)
            if W_FREQ > 0:
                loss = loss + W_FREQ * freq_loss(out, gt32, FREQ_MODE)

            l = loss.item()
            med = sorted(recent[-50:])[len(recent[-50:]) // 2] if len(recent) >= 20 else None
            if not math.isfinite(l) or (med is not None and l > 3 * med):
                skipped += 1                     # one bad batch must not nuke the run
                continue
            recent.append(l)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            total += l * noisy.size(0)
            count += noisy.size(0)

        train_loss = total / max(count, 1)
        val_p, val_s, val_lp, val_hf = validate(model, val_ds, device, VAL_BATCH)
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        ema_p, ema_s, ema_lp, ema_hf = validate(model, val_ds, device, VAL_BATCH)
        model.load_state_dict(raw)

        mins = (time.time() - t0) / 60
        print(f"epoch {epoch}: loss={train_loss:.5f}  "
              f"val {val_p:.2f}dB/{val_s:.4f} lpips {val_lp:.4f} hf {val_hf:.3f}  |  "
              f"ema {ema_p:.2f}dB/{ema_s:.4f} lpips {ema_lp:.4f} hf {ema_hf:.3f}  "
              f"({mins:.1f} min)" + (f"  [{skipped} spikes]" if skipped else ""), flush=True)

        use_ema = ema_s >= val_s
        cp, cs, clp, chf = (ema_p, ema_s, ema_lp, ema_hf) if use_ema else (val_p, val_s, val_lp, val_hf)
        with open(log, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.2e}", f"{train_loss:.6f}",
                                    f"{val_p:.4f}", f"{val_s:.5f}",
                                    f"{ema_p:.4f}", f"{ema_s:.5f}", f"{mins:.1f}",
                                    f"{clp:.5f}", f"{chf:.4f}"])

        # Selection score: SSIM, minus LPIPS if we are optimising perceptually.
        # Pure-SSIM selection would pick the blurriest checkpoint.
        score = cs - (clp if W_LPIPS > 0 else 0.0)
        if score > best_score:
            best_score, best_psnr, best_ssim = score, cp, cs
            torch.save({"model": ema.state_dict() if use_ema else raw,
                        "is_ema": bool(use_ema), "epoch": int(epoch),
                        "val_psnr": float(cp), "val_ssim": float(cs),
                        "val_lpips": float(clp), "val_hf_ratio": float(chf),
                        "dim": DIM, "num_heads": NUM_HEADS,
                        "num_blocks": list(NUM_BLOCKS),
                        "log_input": LOG_INPUT, "noise_cond": NOISE_COND,
                        "wide_head": WIDE_HEAD, "skip": SKIP},
                       os.path.join(OUT_DIR, "best.pth"))
            print(f"  -> best {'ema' if use_ema else 'raw'}  "
                  f"{cp:.2f} dB / {cs:.4f} / lpips {clp:.4f} / hf {chf:.3f}", flush=True)

        torch.save({"model": raw, "opt": opt.state_dict(), "ema": ema.state_dict(),
                    "epoch": int(epoch), "best_score": float(best_score),
                    "best_psnr": float(best_psnr), "best_ssim": float(best_ssim)},
                   os.path.join(OUT_DIR, "last.pth"))

        elapsed = time.time() - t0
        if elapsed + elapsed / (epoch - start_epoch + 1) > MAX_HOURS * 3600:
            print(f"stopping: next epoch would exceed the {MAX_HOURS}h budget", flush=True)
            break

    print(f"done. best {best_psnr:.2f} dB / {best_ssim:.4f} "
          f"({(time.time()-t0)/3600:.2f} h)  ->  {OUT_DIR}/best.pth", flush=True)


if __name__ == "__main__":
    main()