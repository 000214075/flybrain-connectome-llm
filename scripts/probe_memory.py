"""Find what exhausts the VRAM during a whole-brain training step.

`torch.cuda.memory_allocated()` stays near 2.5 GiB while `mem_get_info()` reports
zero free, so most of the card is held by something other than live tensors. This
separates allocator cache from live allocation and from non-torch buffers (the
sparse library's workspaces) so the fix targets the right thing.
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
from flybrain.wholebrain import BrainPathway, load_whole_brain  # noqa: E402


def report(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    stats = torch.cuda.memory_stats()
    print(
        f"{tag:<28} free {free / 1024**3:6.2f} GiB | torch live "
        f"{torch.cuda.memory_allocated() / 1024**3:6.2f} | torch reserved "
        f"{torch.cuda.memory_reserved() / 1024**3:6.2f} | peak reserved "
        f"{stats.get('reserved_bytes.all.peak', 0) / 1024**3:6.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--edge-chunk", type=int, default=4_000_000)
    args = parser.parse_args()

    tune_rocm()
    print("allocator env:", {k: v for k, v in os.environ.items() if "ALLOC" in k.upper()})
    report("start")

    pathway = Pathway.load("data/pathway_malecns.npz")
    circuit, meta, _ = load_whole_brain(
        "data/wholebrain.npz", device="cuda", edge_chunk=args.edge_chunk, impl="csr"
    )
    brain = BrainPathway(circuit, 512, rank=64, chunks=4, iters=1, spiking=True)
    report("after circuit")

    cfg = ModelConfig(
        vocab_size=16384,
        d_model=512,
        n_layers=8,
        n_heads=8,
        context=args.context,
        ffn_mode="mb",
        mb_steps=1,
    )
    model = FlyBrainLM(cfg, pathway, brain).cuda()
    report("after model")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    report("after optimizer")

    x = torch.randint(0, cfg.vocab_size, (args.batch, args.context), device="cuda")
    y = torch.randint(0, cfg.vocab_size, (args.batch, args.context), device="cuda")

    model.train()
    for step in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = args.batch * args.context
        report(f"step {step} ({tokens / elapsed:,.0f} tok/s)")
    print()
    print("largest device segments by active bytes:")
    segments = torch.cuda.memory_snapshot()
    big = sorted(
        (s for s in segments if s.get("active_size", 0) > 0),
        key=lambda s: -s["active_size"],
    )[:6]
    for segment in big:
        print(
            f"  active {segment['active_size'] / 1024**2:8.1f} MiB  "
            f"reserved {segment['reserved_size'] / 1024**2:8.1f} MiB  "
            f"type {segment.get('segment_type', '?')}  blocks {len(segment.get('blocks', []))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
