"""Degradation augmentation applied to the REAL noisy inputs.

Not yet wired into dataset.py -- deliberately. Wiring it now would change the
behaviour of the already-queued v3 runs mid-flight and make them uninterpretable.

WHY THIS EXISTS
---------------
Two rounds of fitting the forward model have not made synthesis competitive with
real pairs (SSIM 0.6037 real-only vs 0.5930 at 25% synthetic, and the v2
autocorrelation fix moved that by 0.00002). Every synthetic pair bets the model
on our forward model being right, and the measurements say it still is not: the
dark decile is too light-tailed, kurtosis 0.59 against a real 2.55.

This sidesteps the bet. Instead of generating an input from GT, take the REAL
NoisyLR -- whose noise is real by construction, dark tail and all -- and add
FURTHER degradation on top. The result is not a sample from the test
distribution, but its errors are real noise plus known extra, rather than
entirely modelled noise. For the OOD robustness the problem statement asks for,
what matters is exposure to a RANGE of degradation, and this gives that without
requiring the forward model to be correct.

THE TRADE-OFF, STATED PLAINLY
-----------------------------
This can only ever make inputs noisier, never cleaner, so the training inputs
are systematically harder than the test inputs. A model trained on them may
over-smooth at test time. Two mitigations, both worth testing rather than
assuming:

  * apply to a fraction of samples only (`p`), keeping most inputs untouched;
  * keep `strength` modest, and consider enabling the model's NOISE_COND input
    channel (off by default) so it can adapt denoising strength per image rather
    than learning one compromise. Level augmentation and noise conditioning are
    natural partners: the first creates the variation, the second gives the
    model a way to respond to it.

Usage (once wired):
    aug = InputAugment(p=0.25, max_strength=0.6, seed=...)
    noisy = aug(noisy)          # gt is untouched
"""
import numpy as np

from degrade import READ_STD, SHOT_GAIN, SPECKLE_STD


class InputAugment:
    """Add extra speckle / shot / read noise on top of an already-noisy LR image.

    `max_strength` is relative to the fitted noise levels: 0.6 means the added
    noise can reach 60% of the real noise level, which combines in quadrature to
    about 1.17x the real total. Each call draws a strength in [0, max_strength],
    so most augmented samples are only mildly harder than the original.
    """

    def __init__(self, p=0.25, max_strength=0.6, seed=None):
        self.p = p
        self.max_strength = max_strength
        self.rng = np.random.default_rng(seed)

    def __call__(self, noisy_lr):
        if self.rng.random() >= self.p:
            return noisy_lr
        x = np.asarray(noisy_lr, dtype=np.float32)
        t = float(self.rng.uniform(0.0, self.max_strength))
        if t <= 0:
            return x
        s, a, g = SPECKLE_STD * t, SHOT_GAIN * t * t, READ_STD * t
        if s > 0:
            k = 1.0 / (s * s)
            x = x * self.rng.gamma(k, 1.0 / k, x.shape).astype(np.float32)
        if a > 0:
            x = (a * self.rng.poisson(np.maximum(x, 0.0) / a)).astype(np.float32)
        if g > 0:
            x = x + (g * self.rng.standard_normal(x.shape)).astype(np.float32)
        return x.astype(np.float32)


class WideAugment:
    """A deliberately DIVERSE degradation family, not a scaled copy of degrade.py.

    WHY THIS EXISTS, AND WHY IT IS NOT `InputAugment`
    -------------------------------------------------
    `InputAugment` adds more of exactly the noise `degrade.py` fits: Gamma
    speckle, then Poisson shot, then Gaussian read, all scaled by one strength.
    Training on that buys robustness to *our own forward model*, and when the
    validation probe was built from the same three constants the result was
    circular -- measured 3 Sep, and it is why the earlier robustness claim was
    retracted.

    The test set contains out-of-distribution samples from different sources.
    Those will not be our forward model at a different amplitude; they will be
    different instruments, different settings, different noise character. So the
    useful thing to train on is *variety of kind*, not variety of amount:

      * additive Gaussian at a random level -- the generic case
      * GAUSSIAN speckle rather than Gamma -- deliberately the wrong shape, so
        the model cannot lean on the skew our fitted model happens to have
      * Poisson shot at a random gain
      * mild Gaussian blur -- a degradation the forward model has NEVER
        contained, so nothing in training has ever anticipated it
      * a random SUBSET in a random ORDER, so the composition varies too

    Each is drawn per sample. Held-out families for evaluation (uniform noise,
    box blur, quantisation) are in `stress.py --family heldout` and appear
    nowhere here, so the probe and the augmenter share no generator.
    """

    def __init__(self, p=0.25, strength=1.0, seed=None):
        self.p = p
        self.strength = strength
        self.rng = np.random.default_rng(seed)

    def _blur(self, x, sigma):
        r = max(1, int(2 * sigma + 0.5))
        k = np.exp(-np.arange(-r, r + 1) ** 2 / (2 * sigma ** 2))
        k /= k.sum()
        pad = np.pad(x, r, mode="reflect")
        out = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, pad)
        return np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 1, out)

    def __call__(self, noisy_lr):
        rng = self.rng
        if rng.random() >= self.p:
            return noisy_lr
        x = np.asarray(noisy_lr, dtype=np.float32)
        s = self.strength

        ops = []
        if rng.random() < 0.60:
            ops.append(("gauss", rng.uniform(0.01, 0.06) * s))
        if rng.random() < 0.50:
            ops.append(("speckle", rng.uniform(0.04, 0.18) * s))
        if rng.random() < 0.40:
            ops.append(("poisson", rng.uniform(0.004, 0.020) * s))
        if rng.random() < 0.30:
            ops.append(("blur", rng.uniform(0.4, 1.1)))
        if not ops:
            ops = [("gauss", rng.uniform(0.01, 0.06) * s)]
        rng.shuffle(ops)

        for kind, lvl in ops:
            if kind == "gauss":
                x = x + (lvl * rng.standard_normal(x.shape)).astype(np.float32)
            elif kind == "speckle":                      # GAUSSIAN, not Gamma
                x = x * (1.0 + lvl * rng.standard_normal(x.shape)).astype(np.float32)
            elif kind == "poisson":
                x = (lvl * rng.poisson(np.maximum(x, 0.0) / lvl)).astype(np.float32)
            elif kind == "blur":
                x = self._blur(x, lvl).astype(np.float32)
        return x.astype(np.float32)
