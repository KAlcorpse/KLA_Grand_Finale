#!/usr/bin/env python3
"""Smoke test for run.py's TensorRT path and its fallbacks.

    python smoke_test.py [--input-dir DIR] [--n 16] [--with-build]

Every scenario runs run.py as a subprocess and checks (a) it succeeds, (b) which
backend actually processed the images (parsed from run.py's own summary), and
(c) the output agrees with the plain PyTorch path. Scenarios that break things
run in a scratch copy of this folder (symlinks for the big files), so the real
engines are never touched.

  torch        --backend torch                      reference output
  trt          --backend trt                        engine used, matches torch
  auto-small   default flags, 16 images            below TRT_MIN_IMAGES -> .pth, no warning
  auto-large   default flags, 200 images           at/above TRT_MIN_IMAGES -> engine
  selfcheck    tolerance forced to ~0              engine disagrees with .pth -> falls back
  compat-only  only the ampere_plus engine present  loads it, matches torch
  corrupt      truncated engine file                falls back to the .pth
  no-engine    engine missing, build not allowed    falls back to the .pth
  no-tensorrt  `import tensorrt` raises             falls back to the .pth
  mixed-sizes  128 + 256 + 100 px in one directory   128 -> engine, 256 and 100 -> .pth
  build        (--with-build) no engine, --trt-build auto, 128 + 100 px inputs
                                                   builds the 128 engine, then uses it
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN_TOL, MAX_TOL = 3e-3, 3e-2      # on [0,1] images; fp16 engine vs fp16 checkpoint measures ~2e-4 / ~3e-3


def make_inputs(dst, src_dir, n, extra_100=0, extra_256=0):
    os.makedirs(dst, exist_ok=True)
    if src_dir:
        names = sorted(f for f in os.listdir(src_dir) if f.endswith(".npy"))[:n]
        for f in names:
            shutil.copy(os.path.join(src_dir, f), dst)
    else:
        rng = np.random.default_rng(0)
        for i in range(n):
            base = np.clip(rng.gamma(2.0, 0.12, (128, 128)), 0, 1.5)
            np.save(os.path.join(dst, f"{i:04d}.npy"), base.astype(np.float32))
    rng = np.random.default_rng(1)
    for i in range(extra_256):
        np.save(os.path.join(dst, f"big_{i:03d}.npy"), np.clip(rng.gamma(2.0, 0.12, (256, 256)), 0, 1.5).astype(np.float32))
    for i in range(extra_100):
        np.save(os.path.join(dst, f"odd_{i:03d}.npy"), np.clip(rng.gamma(2.0, 0.12, (100, 100)), 0, 1.5).astype(np.float32))


def scratch_copy(root, engines="link"):
    """A private copy of this folder: symlinks to run.py/trt_backend.py/weights/ONNX,
    engines linked, corrupted or omitted."""
    os.makedirs(os.path.join(root, "models"))
    for f in ("run.py", "trt_backend.py"):
        os.symlink(os.path.join(HERE, f), os.path.join(root, f))
    for f in os.listdir(os.path.join(HERE, "models")):
        src = os.path.join(HERE, "models", f)
        if f.endswith(".engine"):
            if engines == "link":
                os.symlink(src, os.path.join(root, "models", f))
            elif engines == "compat" and "ampere_plus" in f:
                os.symlink(src, os.path.join(root, "models", f))
            elif engines == "corrupt":
                with open(src, "rb") as a, open(os.path.join(root, "models", f), "wb") as b:
                    b.write(a.read(1 << 20))
        elif os.path.isfile(src):
            os.symlink(src, os.path.join(root, "models", f))
    return root


def run(folder, inp, out, *flags, env=None):
    e = dict(os.environ, **(env or {}))
    p = subprocess.run([sys.executable, os.path.join(folder, "run.py"), inp, out, *flags],
                       capture_output=True, text=True, env=e)
    m = re.search(r"backend\s+TensorRT (\d+) img, PyTorch (\d+) img", p.stderr)
    return p.returncode, (int(m.group(1)), int(m.group(2))) if m else None, p.stderr


def compare(out_a, out_b):
    worst_mean = worst_max = 0.0
    names = sorted(os.listdir(out_a))
    assert names == sorted(os.listdir(out_b)), "different file sets"
    for f in names:
        a, b = np.load(os.path.join(out_a, f)), np.load(os.path.join(out_b, f))
        assert a.shape == b.shape, f"{f}: shape {a.shape} vs {b.shape}"
        d = np.abs(a - b)
        worst_mean, worst_max = max(worst_mean, d.mean()), max(worst_max, d.max())
    return worst_mean, worst_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="real .npy inputs (default: synthetic 128x128)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--with-build", action="store_true", help="also test an on-the-fly engine build (~1-3 min)")
    a = ap.parse_args()

    have_engine = bool(glob.glob(os.path.join(HERE, "models", "model_fp16_128_*.engine")))
    results = []

    def record(name, ok, detail):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:12s} {detail}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        inp, inp_mixed = os.path.join(tmp, "in"), os.path.join(tmp, "in_mixed")
        make_inputs(inp, a.input_dir, a.n)
        make_inputs(inp_mixed, a.input_dir, 6, extra_100=2, extra_256=2)

        ref = os.path.join(tmp, "out_torch")
        rc, used, err = run(HERE, inp, ref, "--backend", "torch")
        record("torch", rc == 0 and used == (0, a.n), f"rc={rc} backend TRT/torch imgs={used}")
        if rc != 0:
            print(err)
            sys.exit(1)

        def check(name, folder, flags, expect, inputs=inp, ref_dir=ref, env=None, same=False, contains=None):
            out = os.path.join(tmp, f"out_{name}")
            rc, used, err = run(folder, inputs, out, *flags, env=env)
            if rc != 0 or used is None:
                return record(name, False, f"rc={rc}\n{err[-800:]}")
            mean, mx = compare(out, ref_dir)
            tol_ok = (mean == 0 and mx == 0) if same else (mean <= MEAN_TOL and mx <= MAX_TOL)
            warned = any(l.startswith("[trt]") and "TRT_MIN_IMAGES" not in l and
                         ("PyTorch" in l or "falling back" in l) for l in err.splitlines())
            found = contains is None or contains in err
            record(name, tol_ok and found and expect(used, warned),
                   f"backend TRT/torch imgs={used}  fallback warning={'yes' if warned else 'no'}  "
                   f"{'' if contains is None else ('[' + contains + ' found] ' if found else '[MISSING ' + contains + '] ')}"
                   f"vs torch: mean|d|={mean:.2e} max|d|={mx:.2e}")

        if have_engine:
            n = a.n
            check("trt", HERE, ["--backend", "trt"], lambda u, w: u == (n, 0) and not w)
            check("auto-small", HERE, [], lambda u, w: u == (0, n) and not w, same=True)
            big, big_ref = os.path.join(tmp, "in_big"), os.path.join(tmp, "out_torch_big")
            make_inputs(big, a.input_dir, 200)
            if len(os.listdir(big)) < 200:                      # fewer real files than 200: repeat them
                src = sorted(os.listdir(big))
                for i in range(200 - len(src)):
                    shutil.copy(os.path.join(big, src[i % len(src)]), os.path.join(big, f"rep_{i:04d}.npy"))
            run(HERE, big, big_ref, "--backend", "torch", "--quiet")
            check("auto-large", HERE, [], lambda u, w: u == (200, 0) and not w, inputs=big, ref_dir=big_ref)
            s = os.path.join(tmp, "s_selfcheck")
            scratch_copy(s, engines="link")
            os.remove(os.path.join(s, "run.py"))                # private copy with an impossible tolerance
            shutil.copy(os.path.join(HERE, "run.py"), os.path.join(s, "run.py"))
            txt = open(os.path.join(s, "run.py")).read().replace("TRT_SELFCHECK_TOL = 3e-3", "TRT_SELFCHECK_TOL = 1e-9")
            open(os.path.join(s, "run.py"), "w").write(txt)
            check("selfcheck", s, ["--backend", "trt"], lambda u, w: u[0] == 0 and u[1] == n and w)
            s = scratch_copy(os.path.join(tmp, "s_compat"), engines="compat")
            check("compat-only", s, ["--backend", "trt"], lambda u, w: u == (n, 0) and not w,
                  contains="loaded ampere_plus engine")
        else:
            record("trt", False, "no models/model_fp16_128_*.engine here: build one with "
                                 "`python trt_backend.py build --size 128` first")

        s = scratch_copy(os.path.join(tmp, "s_corrupt"), engines="corrupt")
        check("corrupt", s, ["--backend", "trt"], lambda u, w: u == (0, a.n) and w, same=True)

        s = scratch_copy(os.path.join(tmp, "s_none"), engines="none")
        check("no-engine", s, ["--backend", "trt"], lambda u, w: u == (0, a.n) and w, same=True)

        fake = os.path.join(tmp, "fake_trt")
        os.makedirs(fake)
        open(os.path.join(fake, "tensorrt.py"), "w").write("raise ImportError('simulated: tensorrt not installed')\n")
        check("no-tensorrt", HERE, ["--backend", "trt"], lambda u, w: u == (0, a.n) and w, same=True,
              env={"PYTHONPATH": fake})

        if have_engine:
            ref_m = os.path.join(tmp, "out_torch_mixed")
            run(HERE, inp_mixed, ref_m, "--backend", "torch")
            check("mixed-sizes", HERE, ["--backend", "trt"], lambda u, w: u == (6, 4), inputs=inp_mixed, ref_dir=ref_m)

        if a.with_build:
            inp_b, ref_b = os.path.join(tmp, "in_build"), os.path.join(tmp, "out_torch_build")
            make_inputs(inp_b, a.input_dir, 8, extra_100=4)
            run(HERE, inp_b, ref_b, "--backend", "torch", "--quiet")
            s = scratch_copy(os.path.join(tmp, "s_build"), engines="none")
            check("build", s, ["--backend", "trt", "--trt-build", "auto"], lambda u, w: u == (8, 4),
                  inputs=inp_b, ref_dir=ref_b)

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} scenarios passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
