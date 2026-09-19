"""Does torch.compile help the connectome modules on this ROCm build?

The brain and mushroom body modules are full of small pointwise ops (LIF steps,
k-WTA, layer norms, means, softplus) that are memory-bound and launch-bound.
Inductor can fuse those, so it is worth measuring rather than assuming -- but it
also has to survive on ROCm, where several inductor paths are immature.

    python scripts/bench_compile.py --batch 8 --steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import Pathway  # noqa: E402
from flybrain.device import tune_rocm  # noqa: E402
from flybrain.model import FlyBrainLM, ModelConfig  # noqa: E402


def build(*, brain: bool, d_model: int = 512) -> FlyBrainLM:
    cfg = ModelConfig(
        vocab_size=16384,
        d_model=d_model,
        n_layers=8,
        n_heads=8,
        context=512,
        ffn_mode="mb",
        mb_steps=1,
        brain_path="data/wholebrain.npz" if brain else "",
        brain_chunks=2,
        brain_iters=1,
        brain_impl="index_add",
    )
    module = None
    if brain:
        from flybrain.wholebrain import BrainPathway, load_whole_brain

        circuit, _, _ = load_whole_brain("data/wholebrain.npz", device="cuda", impl="index_add")
        module = BrainPathway(circuit, d_model, rank=64, chunks=2, iters=1)
    return FlyBrainLM(cfg, Pathway.load("data/pathway_malecns.npz"), module).cuda()


def measure(model: FlyBrainLM, *, batch: int, steps: int) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, model.cfg.vocab_size, (batch, 512), device="cuda")
    y = torch.randint(0, model.cfg.vocab_size, (batch, 512), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    step()  # includes any compilation
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    secs = (time.perf_counter() - started) / steps
    return {
        "seconds_per_step": round(secs, 4),
        "tokens_per_second": round(batch * 512 / secs, 1),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", default="reports/compile_bench.json")
    args = parser.parse_args()

    tune_rocm()
    results = []
    for brain in (False, True):
        for compiled in (False, True):
            label = f"{'brain' if brain else 'no brain'}, compile={compiled}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model = build(brain=brain)
                if compiled:
                    model = torch.compile(model)
                stats = measure(model, batch=args.batch, steps=args.steps)
            except Exception as exc:
                print(f"{label:<28} FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                results.append({"case": label, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                continue
            entry = {"case": label, **stats}
            results.append(entry)
            print(
                f"{label:<28} {stats['tokens_per_second']:>9,.0f} tok/s  "
                f"{stats['seconds_per_step']:.3f} s/step  reserved {stats['reserved_gb']:.1f} GB",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
