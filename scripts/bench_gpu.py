"""Measure what the local AMD GPU actually delivers for this workload.

Run this before committing to a training configuration: it establishes the
achievable tokens/second and confirms which attention kernels and dtypes exist
on this box, rather than assuming the CUDA playbook transfers to ROCm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.device import describe_memory, device_report, pick_device, tune_rocm  # noqa: E402


def timeit(fn, *, warmup: int = 3, iters: int = 10) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def bench_matmul(dtype, sizes=(4096,)) -> dict:
    import torch

    out = {}
    for n in sizes:
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        secs = timeit(lambda: a @ b, warmup=2, iters=5)
        tflops = 2 * n**3 / secs / 1e12
        out[f"{n}x{n}"] = {"seconds": secs, "tflops": round(tflops, 2)}
        del a, b
    return out


def bench_sdpa(dtype, batch=8, heads=8, seq=512, head_dim=64) -> dict:
    import torch
    import torch.nn.functional as F

    q, k, v = (
        torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype) for _ in range(3)
    )
    try:
        secs = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), warmup=2, iters=5)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"seconds": secs, "seq": seq, "heads": heads, "head_dim": head_dim}


def bench_training_step(cfg_name: str, *, dtype_name: str, context: int, batch: int, steps: int = 6) -> dict:
    """End-to-end throughput: forward + backward + optimizer, tokens/second."""
    import torch

    from flybrain.connectome import build_synthetic_pathway
    from flybrain.model import FlyBrainLM, ModelConfig

    pathway = build_synthetic_pathway(240, 2400, 34, seed=0)
    cfg = ModelConfig(
        vocab_size=16384,
        d_model=512,
        n_layers=8,
        n_heads=8,
        context=context,
        ffn_mode="mb" if cfg_name == "mb" else "swiglu",
        mb_steps=2,
    )
    torch.manual_seed(0)
    model = FlyBrainLM(cfg, None if cfg_name == "swiglu" else pathway).to("cuda")
    model.train()
    params = model.param_count()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True if torch.cuda.is_available() else False)
    x = torch.randint(0, cfg.vocab_size, (batch, context), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (batch, context), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    step()  # compile-free warmup
    secs = timeit(step, warmup=2, iters=steps)
    tokens = batch * context
    return {
        "ffn_mode": cfg_name,
        "dtype": dtype_name,
        "params": params,
        "seconds_per_step": round(secs, 4),
        "tokens_per_second": round(tokens / secs, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/gpu_bench.json")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    tune_rocm(verbose=True)
    device = pick_device()
    report = device_report()
    print("=== device ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if device.type != "cuda":
        print("no accelerator available; aborting benchmark")
        return 1

    import torch

    results: dict = {"device": report, "matmul": {}, "sdpa": {}, "train": []}
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        try:
            results["matmul"][str(dtype).replace("torch.", "")] = bench_matmul(dtype)
        except Exception as exc:
            results["matmul"][str(dtype).replace("torch.", "")] = {"error": str(exc)}
    for dtype in (torch.bfloat16, torch.float16):
        results["sdpa"][str(dtype).replace("torch.", "")] = bench_sdpa(dtype)
    for cfg_name in ("mb", "swiglu"):
        for dtype_name in ("bf16",):
            try:
                results["train"].append(
                    bench_training_step(cfg_name, dtype_name=dtype_name, context=args.context, batch=args.batch)
                )
                torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                results["train"].append({"ffn_mode": cfg_name, "error": f"{type(exc).__name__}: {exc}"})
    results["memory_after"] = describe_memory()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("=== results ===")
    print(json.dumps({k: v for k, v in results.items() if k != "device"}, indent=2, ensure_ascii=False))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
