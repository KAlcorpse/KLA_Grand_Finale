"""Forward degradation model, fitted from THIS phase's data.

The model form (Gamma speckle -> Poisson shot -> Gaussian read, split across a
bicubic HR->LR resize, with per-image level jitter) is carried over from a
previous phase on a different SEM dataset. The CONSTANTS are not: this dataset
is confirmed to be different data (same 256/128, .npy, [0,1]-normalised GT
convention and filename scheme, but different pixel content), so everything
below was re-fit against semicon_train_data_excluded/{GT,NoisyLR} rather than
assumed to carry over. See fit_forward_model.py, and NOTES.md "Forward model"
section, for the fitting run and the reasoning.

WHAT THE VARIANCE SAYS
----------------------
Var(y|x) = s^2 x^2 + a x + g^2, fit over 400 pairs / 40 intensity bins, after
excluding the 0.065% of pixels where bicubic downsampling of GT rings slightly
negative (not real signal):

    s^2 x^2        + g^2      R2 = 0.99311
    s^2 x^2 + a x  + g^2      R2 = 0.99911     <- residual error ~10x smaller

Same qualitative story as before -- Poisson shot noise is not optional, and it
dominates at dark pixels. Different quantitatively: the fitted READ_STD came
out ~0 here (the unconstrained fit puts g at the numerical floor). That is a
real difference from the previous dataset's g=0.0293, not a bug -- see NOTES.md
for the direct low-intensity-bin check that supports it. Kept as a small
non-zero floor below for numerical safety, not because the data asked for it.

WHAT THE HIGHER MOMENTS SAY
---------------------------
Same shape story, more pronounced. Measured, dark decile / bright decile:

                              std      ac    skew_dark  kurt_dark  skew_brt  kurt_brt
    REAL                    0.1047  -0.051      1.358      8.275      0.372     0.556
    this model (fitted)     0.1068  -0.050      0.564      1.518      0.401     0.495

The dark-decile tail is heavier here than the previous dataset's REAL numbers
(kurtosis 8.3 vs their 2.5) and the model undershoots it by even more in
relative terms. Checked that this is not a handful of outlier images: pooled
dark-decile kurtosis barely moves (8.28 -> 8.01) after dropping the top 1% of
images by their own per-image kurtosis. It is a real, broad property of this
data's dark pixels, still unexplained by this model. First place to look if
synthetic data underperforms real data here too (it did, consistently, on the
previous dataset).

THE MODEL
---------
    per image:  L ~ level jitter, applied to every noise term
    at HR:      speckle (Gamma) -> shot (Poisson) -> read (Gaussian)
    resize:     bicubic 2x, no anti-aliasing
    at LR:      speckle (Gamma) -> shot (Poisson) -> read (Gaussian)

VAR_AT_HR is the fraction of noise VARIANCE injected before the resize, fit by
grid search against the residual autocorrelation: 0.325 gives ac=-0.0499
against a measured real ac=-0.0507 (mean of the two nearest-neighbour lags).
"""
import numpy as np
import torch
import torch.nn.functional as F

SPECKLE_STD = 0.1589       # s   multiplicative, Gamma        (fitted; prev phase 0.1590 -- coincidentally near-identical)
SHOT_GAIN = 0.01059        # a   Var contribution a*x; photon count is x/a   (fitted; prev phase 0.008018)
READ_STD = 0.003           # g   additive Gaussian   (fit converged to ~0; kept as a small floor, not measured)
VAR_AT_HR = 0.325          # fraction of noise variance injected before the resize   (fitted; prev phase 0.36)
RESIZE_NOISE_GAIN = 0.7204 # K: std of bicubic_downsample_2x(white noise) -- deterministic, matches prev phase's 0.721 exactly
LEVEL_JITTER = 0.12        # per-image spread of the overall level (measured 0.1194, p90/p10=1.33)


def bicubic_downsample(x):
    """(..., 2H, 2W) -> (..., H, W), matching the fitted degradation kernel."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    squeeze = t.ndim == 2
    if squeeze:
        t = t[None]
    out = F.interpolate(t[:, None], scale_factor=0.5, mode="bicubic",
                        align_corners=False, antialias=False)[:, 0].numpy()
    return out[0] if squeeze else out


class Degrader:
    """gt_hr (2H,2W) float32 in [0,1] -> (H,W) float32 degraded.

    `jitter` widens the per-image level distribution beyond the measured 0.10.
    The test set contains out-of-distribution samples, and a model trained at
    one exact noise level has no reason to hold up at another, so the training
    distribution is deliberately made wider than the fitted one.
    """

    def __init__(self, speckle_std=SPECKLE_STD, shot_gain=SHOT_GAIN,
                 read_std=READ_STD, var_at_hr=VAR_AT_HR, jitter=0.2, seed=None):
        self.s, self.a, self.g = speckle_std, shot_gain, read_std
        self.var_at_hr = var_at_hr
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    def _stage(self, x, s, a, g):
        if s > 0:
            k = 1.0 / (s * s)                        # Gamma(k, 1/k): mean 1, std s
            x = x * self.rng.gamma(k, 1.0 / k, x.shape).astype(np.float32)
        if a > 0:
            x = (a * self.rng.poisson(np.maximum(x, 0.0) / a)).astype(np.float32)
        if g > 0:
            x = x + (g * self.rng.standard_normal(x.shape)).astype(np.float32)
        return x.astype(np.float32)

    def __call__(self, gt_hr):
        gt_hr = np.asarray(gt_hr, dtype=np.float32)
        lvl = 1.0
        if self.jitter:
            lvl = max(0.05, 1.0 + self.rng.normal(0.0, self.jitter))

        pv, K = self.var_at_hr, RESIZE_NOISE_GAIN
        s, a, g = self.s * lvl, self.a * lvl * lvl, self.g * lvl
        hr = self._stage(gt_hr, s * np.sqrt(pv) / K, a * pv / (K * K),
                         g * np.sqrt(pv) / K)
        lo = bicubic_downsample(hr)
        return self._stage(lo, s * np.sqrt(1 - pv), a * (1 - pv),
                           g * np.sqrt(1 - pv))
