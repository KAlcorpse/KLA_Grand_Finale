#!/usr/bin/env python3
"""Optional TensorRT backend for run.py.

run.py imports this lazily, only when it is about to try TensorRT, so the
default torch-only path keeps its startup cost. Everything here is allowed to
fail: run.py catches any exception and falls back to the PyTorch checkpoint.

An engine is NOT portable across GPU architectures or TensorRT versions, so the
file name carries both (`model_fp16_128_sm86_trt11.3.0.99.engine`). The portable
artifact is the ONNX (`models/model_fp16_<size>.onnx`); engines are built from it
on the machine that will run them.

Engine lookup order for one input size (see `load_runner`):
  1. native engine for this exact GPU + TensorRT version   (fastest)
  2. `ampere_plus` engine, built with TensorRT hardware compatibility so it may
     load on any Ampere-or-newer GPU (slightly slower, cannot be verified until
     it is actually loaded on that GPU)
  3. build a native engine from the ONNX now, only when allowed (~1-3 min)

Command line (run from this directory):
    python trt_backend.py export --size 128         # best.pth -> models/model_fp16_128.onnx
    python trt_backend.py build  --size 128         # ONNX -> native engine for THIS GPU
    python trt_backend.py build  --size 128 --compat  # ONNX -> portable ampere_plus engine
    python trt_backend.py info
"""
import argparse
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "kla_trt_engines")

# Only 128x128 (the validated 128->256 scale) has an engine. A 256x256 engine built
# the same way disagreed with the checkpoint by ~4e-3 mean abs error (about 40x the
# 128 engine's ~3e-4 against fp32) on real and synthetic inputs, so run.py's
# self-check rejected it; the cause was not investigated and 256x256 inputs run on
# the .pth. Add 256 back here only after that is understood.
SIZES = (128,)
MAX_BATCH = {128: 32}              # upper bound of the engine's batch profile (fit_max_batch may lower it)
OPT_BATCH = {128: 32}            # run.py's fixed batch, so the kernels are tuned for the common case


def onnx_path(size):
    return os.path.join(MODELS, f"model_fp16_{size}.onnx")


def arch_tag():
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def trt_version():
    import tensorrt as trt
    return trt.__version__


def engine_name(size, kind, arch=None, ver=None):
    tag = "ampere_plus" if kind == "ampere_plus" else (arch or arch_tag())
    return f"model_fp16_{size}_{tag}_trt{ver or trt_version()}.engine"


def has_candidates(size):
    """True if an engine file for this GPU (native or ampere_plus) exists. Cheap:
    no TensorRT import, so a machine without engines pays nothing at startup."""
    import glob
    arch = arch_tag()
    pats = (f"model_fp16_{size}_{arch}_trt*.engine", f"model_fp16_{size}_ampere_plus_trt*.engine")
    return any(glob.glob(os.path.join(d, p)) for d in (MODELS, CACHE) for p in pats)


def writable_dir():
    for d in (MODELS, CACHE):
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".write_probe")
            open(probe, "w").close()
            os.remove(probe)
            return d
        except OSError:
            continue
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# ONNX: export the checkpoint, then rewrite it as an explicitly typed fp16 graph
# ---------------------------------------------------------------------------
def convert_fp16(model):
    """TensorRT 11 has no FP16 builder flag: precision is whatever the ONNX says.

    Conv/MatMul run in fp16; ReduceMean, ReduceL2, Softmax, Sqrt, Reciprocal
    (LayerNorm statistics, F.normalize, attention softmax) and the op producing
    the graph output stay fp32; every other op follows its inputs (promoting to
    fp32 when mixed); initializers adopt the dtype of their consumer. This is the
    same split torch.autocast makes, so the engine matches the checkpoint's own
    fp16 path (measured mean |diff| ~2e-4 on [0,1] images).
    """
    import collections

    import numpy as np
    import onnx
    from onnx import TensorProto as TP
    from onnx import helper, numpy_helper, shape_inference

    low_ops, fp32_ops = {"Conv", "MatMul", "Gemm"}, {"ReduceMean", "ReduceL2", "Softmax", "Sqrt", "Reciprocal"}
    inferred = shape_inference.infer_shapes(model)
    elem = {v.name: v.type.tensor_type.elem_type
            for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    elem.update({n: t.data_type for n, t in inits.items()})
    is_float = lambda n: elem.get(n) == TP.FLOAT
    cur = {i.name: TP.FLOAT for i in g.input if is_float(i.name)}
    outs = {o.name for o in g.output}
    nodes, new_inits, casts, stats = [], {}, {}, collections.Counter()

    def init_as(name, tp):
        if tp == TP.FLOAT:
            return name
        if name not in new_inits:
            new_inits[name] = numpy_helper.from_array(
                numpy_helper.to_array(inits[name]).astype(np.float16), f"{name}__fp16")
        return new_inits[name].name

    def act_as(name, tp):
        if cur[name] == tp:
            return name
        if (name, tp) not in casts:
            out = f"{name}__to_{'fp16' if tp == TP.FLOAT16 else 'fp32'}"
            nodes.append(helper.make_node("Cast", [name], [out], to=tp, name=f"Cast_{out}"))
            casts[(name, tp)], cur[out] = out, tp
            stats["casts"] += 1
        return casts[(name, tp)]

    for node in g.node:
        float_outs = [o for o in node.output if o and is_float(o)]
        if not float_outs:
            nodes.append(node)
            continue
        float_ins = [i for i in node.input if i and is_float(i)]
        acts = [i for i in float_ins if i not in inits]
        if node.op_type in low_ops:
            target = TP.FLOAT16
        elif node.op_type in fp32_ops or any(o in outs for o in node.output) or not acts:
            target = TP.FLOAT
        else:
            target = TP.FLOAT if any(cur[a] == TP.FLOAT for a in acts) else TP.FLOAT16
        new = onnx.NodeProto()
        new.CopyFrom(node)
        for k, i in enumerate(node.input):
            if i in float_ins:
                new.input[k] = init_as(i, target) if i in inits else act_as(i, target)
        nodes.append(new)
        for o in float_outs:
            cur[o] = target
        stats["fp16" if target == TP.FLOAT16 else "fp32"] += 1

    used = {i for n in nodes for i in n.input}
    keep = [t for t in g.initializer if t.name in used] + list(new_inits.values())
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.value_info[:]
    return stats


def export_onnx(size, out_path=None):
    import onnx
    import torch
    sys.path.insert(0, HERE)
    from models.minirestormer import MiniRestormer, config_from_state_dict, load_state_dict_compat

    out_path = out_path or onnx_path(size)
    ck = torch.load(os.path.join(MODELS, "best.pth"), map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = MiniRestormer(**config_from_state_dict(sd, ck if isinstance(ck, dict) else None))
    load_state_dict_compat(model, sd)
    model.eval()
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = os.path.join(tmp, "model.onnx")
        # dummy batch of 2, not 1: torch.export specialises a size-1 batch to a constant, which
        # bakes batch=1 into the Reshapes and breaks every batch > 1
        torch.onnx.export(model, torch.randn(2, 1, size, size), fp32,
                          input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                          opset_version=18, do_constant_folding=True)
        m = onnx.load(fp32)          # pulls in the external weight file the exporter wrote
        stats = convert_fp16(m)
        onnx.checker.check_model(m)
        onnx.save(m, out_path)       # single self-contained file
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB): "
          f"{stats['fp16']} fp16 ops, {stats['fp32']} fp32 ops, {stats['casts']} casts")


# ---------------------------------------------------------------------------
# Engine build
# ---------------------------------------------------------------------------
def fit_max_batch(size, cap):
    """Largest power-of-two batch whose engine activation memory stays under ~30% of the
    GPU's free memory (measured: ~36 MiB per 128x128 image for the fp16 engine, scaling
    with pixels). The 4 GiB RTX 3050 gets 16 at 128x128; an H100 gets the cap."""
    import torch
    free = torch.cuda.mem_get_info()[0]
    fit = int(0.3 * free // ((36 << 20) * (size / 128) ** 2))
    b = 1
    while b * 2 <= min(cap, max(fit, 1)):
        b *= 2
    return b


def build_engine(size, compat=False, max_batch=None, opt_batch=None, log=print):
    import tensorrt as trt
    import torch

    src = onnx_path(size)
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    max_batch = max_batch or fit_max_batch(size, MAX_BATCH[size])
    opt_batch = min(opt_batch or OPT_BATCH[size], max_batch)

    class _Log(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            if severity <= trt.ILogger.Severity.WARNING or os.environ.get("KLA_TRT_VERBOSE"):
                print(f"[TRT {severity.name}] {msg}", file=sys.stderr)

    lg = _Log()
    builder = trt.Builder(lg)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, lg)
    if not parser.parse_from_file(src):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 1 << 30
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(min(2 << 30, free // 2)))
    if compat:
        cfg.hardware_compatibility_level = trt.HardwareCompatibilityLevel.AMPERE_PLUS
    prof = builder.create_optimization_profile()
    prof.set_shape("input", (1, 1, size, size), (opt_batch, 1, size, size), (max_batch, 1, size, size))
    cfg.add_optimization_profile(prof)
    kind = "ampere_plus" if compat else "native"
    log(f"[trt] building {kind} engine {size}x{size}, batch 1..{max_batch} (opt {opt_batch}) from {src} ...")
    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    if plan is None:
        raise RuntimeError("engine build failed")
    out = os.path.join(writable_dir(), engine_name(size, kind))
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(plan)
    os.replace(tmp, out)
    log(f"[trt] built {out} ({os.path.getsize(out) / 1e6:.0f} MB) in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class Runner:
    """One deserialized engine for one square input size, batch 1..max_batch."""

    def __init__(self, path, size, kind):
        import tensorrt as trt
        import torch
        self.torch, self.size, self.kind, self.path = torch, size, kind, path
        with open(path, "rb") as f:
            data = f.read()
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine returned None (wrong GPU/TensorRT version or corrupt file)")
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("could not create an execution context")
        lo, _, hi = self.engine.get_tensor_profile_shape("input", 0)
        self.max_batch = int(hi[0])
        if tuple(lo[1:]) != (1, size, size) or tuple(hi[1:]) != (1, size, size):
            raise RuntimeError(f"engine profile {tuple(lo)}..{tuple(hi)} does not match {size}x{size}")

    def __call__(self, x):
        """x: CUDA float32 (B,1,S,S), B <= max_batch. Returns CUDA float32 (B,1,2S,2S)."""
        torch = self.torch
        x = x.float().contiguous()
        b = x.shape[0]
        y = torch.empty((b, 1, 2 * self.size, 2 * self.size), dtype=torch.float32, device=x.device)
        self.ctx.set_input_shape("input", tuple(x.shape))
        self.ctx.set_tensor_address("input", x.data_ptr())
        self.ctx.set_tensor_address("output", y.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        return y


def load_runner(size, build=False, log=print):
    """Return a Runner for `size`, trying native, then ampere_plus, then (if
    `build`) a fresh native build. Raises if nothing works."""
    ver, arch = trt_version(), arch_tag()
    errors = []
    for kind in ("native", "ampere_plus"):
        for d in (MODELS, CACHE):
            path = os.path.join(d, engine_name(size, kind, arch, ver))
            if not os.path.isfile(path):
                continue
            try:
                r = Runner(path, size, kind)
                log(f"[trt] {size}x{size}: loaded {kind} engine {os.path.relpath(path, HERE)} "
                    f"(TensorRT {ver}, {arch}, batch<= {r.max_batch})")
                return r
            except Exception as e:
                errors.append(f"{kind} engine {path}: {e}")
                log(f"[trt] {size}x{size}: could not use {kind} engine {os.path.basename(path)}: {e}")
    if build:
        path = build_engine(size, compat=False, log=log)
        return Runner(path, size, "native")
    hint = ("no matching engine found" if not errors else "; ".join(errors))
    raise RuntimeError(f"{hint}; build one with `python trt_backend.py build --size {size}` "
                       f"or pass --trt-build auto")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("export", "build", "info"))
    ap.add_argument("--size", type=int, action="append", choices=SIZES)
    ap.add_argument("--compat", action="store_true", help="build the portable ampere_plus engine")
    ap.add_argument("--max-batch", type=int, default=None)
    a = ap.parse_args()
    sizes = a.size or [128]
    if a.cmd == "export":
        for s in sizes:
            export_onnx(s)
    elif a.cmd == "build":
        for s in sizes:
            build_engine(s, compat=a.compat, max_batch=a.max_batch)
    else:
        import torch
        print(f"TensorRT {trt_version()}, GPU {torch.cuda.get_device_name(0)} ({arch_tag()})")
        for s in sizes:
            for kind in ("native", "ampere_plus"):
                p = os.path.join(MODELS, engine_name(s, kind))
                print(f"  {s}x{s} {kind:12s} {'present' if os.path.isfile(p) else 'missing '} {os.path.basename(p)}")


if __name__ == "__main__":
    main()
