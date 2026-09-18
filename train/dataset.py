import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from augment import InputAugment, WideAugment
from degrade import Degrader

SCALE = 2


def _npy_shape(path):
    """Shape of a .npy file without reading its data. ~0.3 s per 1000 files."""
    return tuple(np.load(path, mmap_mode="r").shape)


class RestorationDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir, split="train", val_split=0.1, seed=42,
                 crop=64, augment=True, cache=True, synth_prob=0.0, synth_jitter=0.2,
                 aug_prob=0.0, aug_strength=0.6, aug_mode="fitted",
                 extra_gt_dir=None, extra_noisy_dir=None):
        self.gt_dir, self.noisy_dir, self.split = gt_dir, noisy_dir, split
        self.crop = crop
        self.augment = augment and split == "train"

        names = sorted(set(os.listdir(gt_dir)) & set(os.listdir(noisy_dir)))
        names = [f for f in names if f.endswith(".npy")]
        if not names:
            raise RuntimeError(f"no overlapping .npy files in {gt_dir} and {noisy_dir}")
        random.Random(seed).shuffle(names)
        cut = int(len(names) * (1 - val_split))
        self.fnames = names[:cut] if split == "train" else names[cut:]
        # Per-file source directories: extra pairs may live somewhere else and may
        # reuse filenames, so the directory cannot be a single attribute.
        self._gtd = [gt_dir] * len(self.fnames)
        self._nzd = [noisy_dir] * len(self.fnames)

        # Extra pairs join the TRAIN split ONLY, and only after the split above is
        # fixed. Merging them into `names` before the shuffle would change the
        # split, move held-out images into training, and silently invalidate every
        # number ever measured on valset/.
        if split == "train" and extra_gt_dir:
            ex = sorted(set(os.listdir(extra_gt_dir)) & set(os.listdir(extra_noisy_dir)))
            ex = [f for f in ex if f.endswith(".npy")]
            self.fnames += ex
            self._gtd += [extra_gt_dir] * len(ex)
            self._nzd += [extra_noisy_dir] * len(ex)
            print(f"[{split}] +{len(ex)} extra pairs from {extra_gt_dir}", flush=True)

        self.synth_prob = synth_prob if split == "train" else 0.0
        self._degrader = Degrader(jitter=synth_jitter, seed=seed) if self.synth_prob else None
        # Extra degradation on top of the REAL noisy input -- train split only,
        # and never on validation, which must stay the fixed yardstick.
        self.aug_prob = aug_prob if split == "train" else 0.0
        if not self.aug_prob:
            self._augment = None
        elif aug_mode == "wide":
            # a diverse family, deliberately NOT degrade.py's chain -- see augment.py
            self._augment = WideAugment(p=self.aug_prob, strength=aug_strength, seed=seed)
        else:
            self._augment = InputAugment(p=self.aug_prob, max_strength=aug_strength, seed=seed)
        self._rng_ready = False

        self.cache = cache
        if cache:
            # Shapes are read from the .npy headers, not by loading the arrays,
            # so checking all of them costs nothing. The old code sized the cache
            # from file 0 alone and then wrote every other file into it, which
            # raises a broadcast error the moment a directory mixes resolutions
            # -- exactly what the incoming 512->256 pairs do.
            gs = [_npy_shape(os.path.join(g, f)) for g, f in zip(self._gtd, self.fnames)]
            ns = [_npy_shape(os.path.join(n, f)) for n, f in zip(self._nzd, self.fnames)]
            for f, g, n in zip(self.fnames, gs, ns):
                if tuple(g) != tuple(x * SCALE for x in n):
                    raise RuntimeError(
                        f"expected GT = {SCALE}x NoisyLR, got {g} vs {n} for {f}")

            self.shapes = ns                      # LR shape per index; validate() buckets on it
            self.uniform = len(set(gs)) == 1
            if self.uniform:
                # One contiguous block: the fast path, byte-for-byte what this
                # class did before, and what every single-resolution run uses.
                self._gt = np.empty((len(self.fnames), *gs[0]), np.float16)
                self._noisy = np.empty((len(self.fnames), *ns[0]), np.float16)
                for i, f in enumerate(self.fnames):
                    self._gt[i] = np.load(os.path.join(self._gtd[i], f))
                    self._noisy[i] = np.load(os.path.join(self._nzd[i], f))
                mb = (self._gt.nbytes + self._noisy.nbytes) / 1e6
            else:
                # Mixed resolutions: a list of per-image arrays. self._gt[idx]
                # still yields one image, so __getitem__ is unchanged; random
                # crops are the same size whatever the source resolution.
                self._gt = [np.load(os.path.join(g, f)).astype(np.float16)
                            for g, f in zip(self._gtd, self.fnames)]
                self._noisy = [np.load(os.path.join(n, f)).astype(np.float16)
                               for n, f in zip(self._nzd, self.fnames)]
                mb = sum(a.nbytes for a in self._gt) / 1e6 + \
                     sum(a.nbytes for a in self._noisy) / 1e6
                sizes = {}
                for n in ns:
                    sizes[tuple(n)] = sizes.get(tuple(n), 0) + 1
                print(f"[{split}] mixed resolutions: "
                      + ", ".join(f"{h}x{w} x{c}" for (h, w), c in sorted(sizes.items())),
                      flush=True)
            print(f"[{split}] {len(self.fnames)} pairs cached, {mb:.0f} MB")

    def set_crop(self, crop):
        self.crop = crop

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        if self.cache:
            noisy = self._noisy[idx].astype(np.float32)
            gt = self._gt[idx].astype(np.float32)
        else:
            f = self.fnames[idx]
            noisy = np.load(os.path.join(self._nzd[idx], f)).astype(np.float32)
            gt = np.load(os.path.join(self._gtd[idx], f)).astype(np.float32)

        c = self.crop
        if c and self.split == "train" and c < noisy.shape[-1]:
            y = random.randint(0, noisy.shape[-2] - c)
            x = random.randint(0, noisy.shape[-1] - c)
            noisy = noisy[y:y + c, x:x + c]
            gt = gt[y * SCALE:(y + c) * SCALE, x * SCALE:(x + c) * SCALE]

        if (self._degrader is not None or self._augment is not None) and not self._rng_ready:
            # workers fork this object; reseed per worker or every worker draws
            # the identical noise stream
            wseed = torch.initial_seed() % 2 ** 32
            if self._degrader is not None:
                self._degrader.rng = np.random.default_rng(wseed)
            if self._augment is not None:
                self._augment.rng = np.random.default_rng(wseed ^ 0x5EED)
            self._rng_ready = True

        if self._degrader is not None and random.random() < self.synth_prob:
            noisy = self._degrader(gt)

        # InputAugment makes its own keep/skip draw from its own numpy RNG.
        # Deliberately NOT `random.random() < aug_prob`: that would consume from
        # the `random` stream the rot90/flip below draw from, so turning
        # augmentation on would silently change the geometric augmentation too
        # and make AUG-vs-control a two-change comparison.
        if self._augment is not None:
            noisy = self._augment(noisy)

        noisy = torch.from_numpy(np.ascontiguousarray(noisy)).unsqueeze(0)
        gt = torch.from_numpy(np.ascontiguousarray(gt)).unsqueeze(0)

        if self.augment:
            k = random.randint(0, 3)
            if k:
                noisy, gt = torch.rot90(noisy, k, (-2, -1)), torch.rot90(gt, k, (-2, -1))
            if random.random() < 0.5:
                noisy, gt = torch.flip(noisy, (-1,)), torch.flip(gt, (-1,))
            if random.random() < 0.5:
                noisy, gt = torch.flip(noisy, (-2,)), torch.flip(gt, (-2,))

        return noisy, gt


class TestDataset(Dataset):
    def __init__(self, noisy_dir):
        self.noisy_dir = noisy_dir
        self.fnames = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".npy"))
        if not self.fnames:
            raise RuntimeError(f"no .npy files in {noisy_dir}")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        f = self.fnames[idx]
        arr = np.load(os.path.join(self.noisy_dir, f)).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0), f