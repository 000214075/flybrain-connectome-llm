"""Break down where the mushroom body module spends its time.

The module is ~4x slower than a SwiGLU feed-forward with more parameters, which
means it is bound by memory traffic and non-matmul ops rather than FLOPs. This
measures each stage separately so the optimisations target the real cost.
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.connectome import Pathway  # noqa: E402
from flybrain.device import tune_rocm  # noqa: E402
from flybrain.model import MushroomBody, ModelConfig  # noqa: E402
from flybrain.neurons import k_wta  # noqa: E402


def bench(fn, *, warmup: int = 5, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters * 1e3  # milliseconds


def main() -> int:
    tune_rocm()
    pathway = Pathway.load("data/pathway_malecns.npz")
    tokens, d_model = 8192, 512  # batch 16 x context 512
    cfg = ModelConfig(d_model=d_model, n_layers=1, n_heads=8, mb_steps=1)
    layer = MushroomBody(cfg, pathway).cuda()
    x = torch.randn(tokens, d_model, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pn = layer.pn_proj(x)
        kc_in = pn @ layer.pn_to_kc
        normed = layer.norm(kc_in)
        mean_activity = normed.mean(dim=-1, keepdim=True)
        inhibited = normed - 1.0 * mean_activity * layer.apl_to_kc
        spikes = layer.lif(inhibited, spiking=True)
        code = k_wta(spikes, layer.k)
        mbon = code @ layer.kc_to_mbon
        out = layer.mbon_proj(mbon)

    results = {
        "pn_proj_(512->692)": bench(lambda: layer.pn_proj(x)),
        "pn_to_kc_(692->4064)": bench(lambda: pn @ layer.pn_to_kc),
        "rmsnorm_over_4064": bench(lambda: layer.norm(kc_in)),
        "apl_mean_reduce": bench(lambda: normed.mean(dim=-1, keepdim=True)),
        "lif_1_step": bench(lambda: layer.lif(inhibited, spiking=True)),
        "lif_4_steps": bench(lambda: layer.lif(inhibited, spiking=True, steps=4)),
        "k_wta_topk(5%)": bench(lambda: k_wta(spikes, layer.k)),
        "k_wta_threshold_alt": bench(
            lambda: spikes * (spikes >= torch.quantile(spikes, 0.95, dim=-1, keepdim=True))
        ),
        "kc_to_mbon_(4064->97)": bench(lambda: code @ layer.kc_to_mbon),
        "mbon_proj_(97->512)": bench(lambda: layer.mbon_proj(mbon)),
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        results["FULL_MODULE"] = bench(lambda: layer(x))

    total = sum(v for k, v in results.items() if k != "FULL_MODULE")
    print(f"{'stage':<26} {'ms':>8} {'share':>7}")
    for name, ms in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"{name:<26} {ms:8.3f} {100 * ms / total:6.1f}%")
    print(f"\nsum of stages {total:.3f} ms vs full module {results['FULL_MODULE']:.3f} ms")

    # What a standard feed-forward costs at the same token count for reference.
    gate = torch.nn.Linear(d_model, 1024, bias=False).cuda()
    up = torch.nn.Linear(d_model, 1024, bias=False).cuda()
    down = torch.nn.Linear(1024, d_model, bias=False).cuda()

    def swiglu():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return down(F.silu(gate(x)) * up(x))

    print(f"{'swiglu_hidden=1024':<26} {bench(swiglu):8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
