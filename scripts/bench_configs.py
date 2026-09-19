"""Measure training throughput for candidate configurations on the real pathway.

Model size should be chosen from measurement, not from the CUDA playbook: on
this machine the connectome module is memory-bound rather than matmul-bound, so
its cost does not follow from FLOPs alone.

    python scripts/bench_configs.py --pathway data/pathway_malecns.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import Pathway, build_synthetic_pathway, shuffle_pathway  # noqa: E402
from flybrain.device import tune_rocm  # noqa: E402
from flybrain.model import FlyBrainLM, ModelConfig  # noqa: E402


def measure(cfg: ModelConfig, pathway, *, context: int, batch: int, steps: int, dtype) -> dict:
    torch.manual_seed(0)
    model = FlyBrainLM(cfg, pathway).to("cuda")
    model.train()
    params = model.param_count()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, cfg.vocab_size, (batch, context), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (batch, context), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    step(), step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    secs = (time.perf_counter() - started) / steps
    tokens = batch * context
    return {
        "params": params,
        "seconds_per_step": round(secs, 4),
        "tokens_per_second": round(tokens / secs, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway", default="data/pathway_malecns.npz")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--out", default="reports/config_bench.json")
    args = parser.parse_args()

    tune_rocm()
    dtype = torch.bfloat16
    if os.path.exists(args.pathway):
        real = Pathway.load(args.pathway)
        print(f"real pathway: PN={real.n_pn} KC={real.n_kc} MBON={real.n_mbon} (nnz {real.meta['pn_to_kc_nnz']})")
    else:
        real = build_synthetic_pathway(692, 4064, 97, seed=0)
        print("pathway file missing; using synthetic wiring of the real shape")

    variants = [
        ("mb", real, {"mb_steps": 1, "mb_kc_sparsity": 0.05}),
        ("mb", real, {"mb_steps": 2, "mb_kc_sparsity": 0.05}),
        ("mb", real, {"mb_steps": 1, "mb_kc_sparsity": 0.10}),
        ("mb-shuffled", shuffle_pathway(real, seed=1), {"mb_steps": 1, "mb_kc_sparsity": 0.05}),
        ("swiglu", None, {"ffn_hidden": 1024}),
        ("swiglu", None, {"ffn_hidden": 512}),
    ]

    results = []
    for mode, pathway, extra in variants:
        cfg = ModelConfig(
            vocab_size=16384,
            d_model=args.d_model,
            n_layers=args.layers,
            n_heads=8,
            context=args.context,
            ffn_mode=mode,
            **extra,
        )
        torch.cuda.reset_peak_memory_stats()
        try:
            stats = measure(cfg, pathway, context=args.context, batch=args.batch, steps=args.steps, dtype=dtype)
        except Exception as exc:
            stats = {"error": f"{type(exc).__name__}: {exc}"}
        tokens_per_step = args.batch * args.context
        entry = {
            "ffn_mode": mode,
            "config": extra,
            **stats,
            "minutes_per_100m_tokens": round(1e8 / stats["tokens_per_second"] / 60, 2)
            if "tokens_per_second" in stats
            else None,
            "tokens_per_optimizer_step_at_accum4": tokens_per_step * 4,
        }
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
