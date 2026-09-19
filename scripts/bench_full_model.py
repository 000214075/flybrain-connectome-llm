"""Choose the training configuration from measurement, including GPU utilisation.

Sweeps batch size and the whole-brain settings, reporting tokens/s, VRAM use and
the real GPU engine busy fraction sampled from Windows performance counters while
the workload runs. Utilisation is measured rather than inferred from throughput,
because a memory-bound module can saturate nothing while still taking time.

    python scripts/bench_full_model.py --steps 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flybrain.connectome import Pathway  # noqa: E402
from flybrain.device import tune_rocm  # noqa: E402
from flybrain.model import FlyBrainLM, ModelConfig  # noqa: E402
from gpu_util import pick_gpu_engines, sample_engines  # noqa: E402


class UtilisationSampler:
    """Samples the GPU engine counters in the background while work runs."""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.readings: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.readings.append(sample_engines())
            self._stop.wait(self.interval)

    def __enter__(self) -> "UtilisationSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=70)

    def summary(self) -> dict:
        series = pick_gpu_engines(self.readings)
        out = {}
        for engine, values in series.items():
            out[engine] = {
                "mean_pct": round(statistics.fmean(values), 2),
                "p50_pct": round(statistics.median(values), 2),
                "max_pct": round(max(values), 2),
            }
        return out


def build_model(
    *, d_model: int, n_layers: int, context: int, ffn_mode: str, brain: bool, brain_chunks: int, brain_iters: int
) -> FlyBrainLM:
    pathway = Pathway.load("data/pathway_malecns.npz")
    cfg = ModelConfig(
        vocab_size=16384,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=8,
        context=context,
        ffn_mode=ffn_mode,
        mb_steps=1,
        brain_path="data/wholebrain.npz" if brain else "",
        brain_chunks=brain_chunks,
        brain_iters=brain_iters,
        brain_impl="csr",
    )
    brain_module = None
    if brain:
        from flybrain.wholebrain import BrainPathway, load_whole_brain

        circuit, meta, _ = load_whole_brain("data/wholebrain.npz", impl="csr")
        brain_module = BrainPathway(
            circuit, d_model, rank=64, chunks=brain_chunks, iters=brain_iters, spiking=True
        )
        print(f"  brain: {meta.n_neurons:,} neurons / {meta.nnz:,} synapses")
    model = FlyBrainLM(cfg, None if ffn_mode == "swiglu" else pathway, brain_module)
    return model


def measure(model: FlyBrainLM, *, batch: int, context: int, steps: int, dtype=torch.bfloat16) -> dict:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    x = torch.randint(0, model.cfg.vocab_size, (batch, context), device="cuda")
    y = torch.randint(0, model.cfg.vocab_size, (batch, context), device="cuda")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    for _ in range(2):  # warm-up, also builds the CSR structures
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with UtilisationSampler(interval=1.5) as sampler:
        started = time.perf_counter()
        for _ in range(steps):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    tokens = batch * context * steps
    return {
        "params": model.param_count(),
        "seconds": round(elapsed, 3),
        "seconds_per_step": round(elapsed / steps, 4),
        "tokens_per_second": round(tokens / elapsed, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "utilisation": sampler.summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--out", default="reports/full_model_bench.json")
    args = parser.parse_args()

    tune_rocm(verbose=True)
    results: list[dict] = []

    print("=== whole-brain inclusion and batch scaling (d_model 512, 8 layers) ===")
    cases = [
        ("mb, no brain", dict(ffn_mode="mb", brain=False, brain_chunks=2, brain_iters=2), [16, 32, 64]),
        ("mb + whole brain (2 chunks x 2 sweeps)", dict(ffn_mode="mb", brain=True, brain_chunks=2, brain_iters=2), [16, 32, 64]),
        ("mb + whole brain (2 chunks x 1 sweep)", dict(ffn_mode="mb", brain=True, brain_chunks=2, brain_iters=1), [32]),
    ]
    for label, kwargs, batches in cases:
        print(f"\n--- {label} ---")
        for batch in batches:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model = build_model(d_model=512, n_layers=8, context=args.context, **kwargs).cuda()
                stats = measure(model, batch=batch, context=args.context, steps=args.steps)
            except RuntimeError as exc:
                print(f"  batch {batch}: FAILED {type(exc).__name__}: {str(exc)[:160]}")
                results.append({"case": label, "batch": batch, "error": str(exc)[:300]})
                continue
            entry = {"case": label, "batch": batch, **stats}
            results.append(entry)
            util = stats["utilisation"]
            compute = util.get("Compute", {}).get("mean_pct", 0.0)
            print(
                f"  batch {batch:>3}: {stats['tokens_per_second']:>9,.0f} tok/s  "
                f"{stats['seconds_per_step']:.3f} s/step  {stats['peak_vram_gb']:.1f} GB  "
                f"GPU compute busy {compute:.1f}%"
            )
            for engine, values in util.items():
                print(f"           {engine:<14} mean {values['mean_pct']:6.2f}%  peak {values['max_pct']:6.2f}%")
            del model
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
