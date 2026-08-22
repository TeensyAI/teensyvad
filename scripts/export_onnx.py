"""Export the teensyvad MLP to ONNX (+ dynamic-int8 variant) and benchmark.

    .venv/bin/python scripts/export_onnx.py --model models/teensy-v2.npz

Why: numpy has no int8 kernels (float32 BLAS actually wins — see
quant.py).  ONNX Runtime does.  This script

1. exports the model to a 4-op ONNX graph (Gemm→Relu→Gemm→Relu→Gemm),
   folding the input standardisation (mean/std) into the first Gemm so
   the graph is raw-features in, logit out;
2. produces a dynamic-int8 quantized copy with ORT's quantize_dynamic;
3. verifies both against the numpy model (max |Δp| on val frames);
4. benchmarks numpy-f32 vs ONNX-f32 vs ONNX-int8, single-frame and
   batched — the honest speed table.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teensyvad.model import load_model  # noqa: E402
from scripts_utils import context_windows  # noqa: E402


def fold_normalization(m):
    """Return (W1', b1', W2, b2, W3, b3) with mean/std folded into layer 1."""
    W1, b1 = m.p["W1"].astype(np.float64), m.p["b1"].astype(np.float64)
    W1f = W1 / m.in_std[:, None]                # column scale by 1/std
    b1f = b1 - (m.in_mean / m.in_std) @ W1
    return (W1f.astype(np.float32), b1f.astype(np.float32),
            m.p["W2"], m.p["b2"], m.p["W3"], m.p["b3"])


def build_onnx(W1, b1, W2, b2, W3, b3, in_dim):
    import onnx
    from onnx import TensorProto, helper

    def tensor(name, arr):
        return helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.ravel().tolist())

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, in_dim])
    y = helper.make_tensor_value_info("logit", TensorProto.FLOAT, [None, 1])
    init = [tensor(n, a) for n, a in
            [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2), ("W3", W3), ("b3", b3)]]
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["h1"], alpha=1.0, beta=1.0),
        helper.make_node("Relu", ["h1"], ["r1"]),
        helper.make_node("Gemm", ["r1", "W2", "b2"], ["h2"]),
        helper.make_node("Relu", ["h2"], ["r2"]),
        helper.make_node("Gemm", ["r2", "W3", "b3"], ["logit"]),
    ]
    graph = helper.make_graph(nodes, "teensyvad_mlp", [x], [y], init)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def bench(fn, n=2000, warmup=200):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/teensy-v2.npz"))
    ap.add_argument("--data", type=Path, default=Path("data/prepared/val.distill.npz"))
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort

    m = load_model(args.model)
    K = int(m.meta["context"])
    onnx_path = args.model.with_suffix(".onnx")
    int8_path = args.model.with_name(args.model.stem + "-int8.onnx")

    # 1) export ------------------------------------------------------------
    W1, b1, W2, b2, W3, b3 = fold_normalization(m)
    onnx_model = build_onnx(W1, b1, W2, b2, W3, b3, m.sizes[0])
    onnx.save(onnx_model, onnx_path)
    print(f"exported → {onnx_path} ({onnx_path.stat().st_size/1024:.1f} KB)")

    # 2) dynamic int8 --------------------------------------------------------
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(onnx_path), str(int8_path),
                     weight_type=QuantType.QInt8)
    print(f"quantized → {int8_path} ({int8_path.stat().st_size/1024:.1f} KB)")

    # 3) numeric verification -------------------------------------------------
    z = np.load(args.data)
    X, _ = context_windows(z["F"], z["F"][:, 0], K)
    Xv = X[:20000].astype(np.float32)
    p_np = m.probs(Xv)

    so = ort.SessionOptions()
    sess_f = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(str(int8_path), so, providers=["CPUExecutionProvider"])

    def probs_ort(sess, x):
        logit = sess.run(["logit"], {"x": x})[0]
        return 1.0 / (1.0 + np.exp(-logit.ravel()))

    p_f32 = probs_ort(sess_f, Xv)
    p_i8 = probs_ort(sess_q, Xv)
    sig = lambda a, b: float(np.mean((a >= .5) == (b >= .5)))
    print(f"\nmax |Δp| onnx-f32 vs numpy : {np.abs(p_f32-p_np).max():.6f}  "
          f"agree {sig(p_f32, p_np)*100:.3f}%")
    print(f"max |Δp| onnx-int8 vs numpy: {np.abs(p_i8-p_np).max():.6f}  "
          f"agree {sig(p_i8, p_np)*100:.3f}%")

    # 4) speed -----------------------------------------------------------------
    X1 = Xv[:1]
    print("\nspeed (median):")
    print(f"  numpy float32      : {bench(lambda: m.probs(X1))*1e6:7.2f} µs/frame single | "
          f"{bench(lambda: m.probs(Xv), 60, 10)/len(Xv)*1e6:6.3f} µs/frame batched")
    for tag, s in [("onnx float32    ", sess_f), ("onnx int8       ", sess_q)]:
        t1 = bench(lambda: probs_ort(s, X1)) * 1e6
        tb = bench(lambda: probs_ort(s, Xv), 60, 10) / len(Xv) * 1e6
        print(f"  {tag}: {t1:7.2f} µs/frame single | {tb:6.3f} µs/frame batched")


if __name__ == "__main__":
    main()
