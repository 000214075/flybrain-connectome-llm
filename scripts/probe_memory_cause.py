"""Isolate which part of the step drives the allocator's reserved memory up.

Runs the same step with the whole brain off, on with a large edge chunk, and on
with a small one, printing reserved memory after each step. Reserved memory far
above live memory is fragmentation, and this finds its source.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import Pathway  # noqa: E402
from flybrain.device import tune_rocm  # noqa: E402
from flybrain.model import FlyBrainLM, ModelConfig  # noqa: E402


def run_case(
    label: str,
    *,
    brain: bool,
    edge_chunk: int,
    impl: str = "csr",
    batch: int,
    context: int,
    steps: int,
    d_model: int = 512,
) -> dict:
    Path = "data/wholebrain.npz"
    cfg = ModelConfig(
        vocab_size=16384,
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        context=context,
        ffn_mode="mb",
        mb_steps=1,
        brain_path=Path if brain else "",
        brain_chunks=4,
        brain_iters=1,
        brain_edge_chunk=edge_chunk,
        brain_impl=impl,
    )
    brain_module = None
    if brain:
        from flybrain.wholebrain import BrainPathway, load_whole_brain

        circuit, _, _ = load_whole_brain(Path, device="cuda", edge_chunk=edge_chunk, impl=impl)
        brain_module = BrainPathway(circuit, d_model, rank=64, chunks=4, iters=1)
    model = FlyBrainLM(cfg, Pathway.load("data/pathway_malecns.npz"), brain_module).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, cfg.vocab_size, (batch, context), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (batch, context), device="cuda")
    model.train()

    timings = []
    for step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if step == 0 or step == steps - 1:
            print(
                f"  {label:<34} step {step}: live {torch.cuda.memory_allocated() / 1024**3:5.2f} GiB  "
                f"reserved {torch.cuda.memory_reserved() / 1024**3:5.2f} GiB  "
                f"free {torch.cuda.mem_get_info()[0] / 1024**3:5.2f} GiB  "
                f"{batch * context / timings[-1]:,.0f} tok/s"
            )
    best = min(timings[1:]) if len(timings) > 1 else timings[0]
    result = {
        "case": label,
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        "tokens_per_second": round(batch * context / best, 1),
    }
    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    tune_rocm()
    results = []
    cases = [
        ("no brain", dict(brain=False, edge_chunk=4_000_000, impl="csr")),
        ("brain csr, chunk 4M", dict(brain=True, edge_chunk=4_000_000, impl="csr")),
        ("brain index_add, chunk 4M", dict(brain=True, edge_chunk=4_000_000, impl="index_add")),
        ("brain index_add, chunk 1M", dict(brain=True, edge_chunk=1_000_000, impl="index_add")),
    ]
    for label, kwargs in cases:
        torch.cuda.reset_peak_memory_stats()
        results.append(run_case(label, batch=args.batch, context=args.context, steps=args.steps, **kwargs))

    print("\n=== summary ===", flush=True)
    for row in results:
        print(f"  {row['case']:<30} reserved {row['reserved_gb']:5.2f} GiB   {row['tokens_per_second']:>9,.0f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
