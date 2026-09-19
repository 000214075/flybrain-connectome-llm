"""Pinpoint the allocation inside the whole-brain sweep.

The module adds ~15 GiB of reserved memory while holding only 2.45 GiB live, so
something allocates far more than the tensors it returns. This measures reserved
memory after the CSR build, after a no-grad matmul, after a matmul with grad, and
after its backward, so the offender is named rather than guessed at.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.device import tune_rocm  # noqa: E402
from flybrain.wholebrain import load_whole_brain  # noqa: E402


def show(tag: str) -> None:
    free, _ = torch.cuda.mem_get_info()
    print(
        f"{tag:<44} live {torch.cuda.memory_allocated() / 1024**3:6.2f}  "
        f"reserved {torch.cuda.memory_reserved() / 1024**3:6.2f}  free {free / 1024**3:6.2f}"
    )


def main() -> int:
    tune_rocm()
    streams = 16
    circuit, meta, _ = load_whole_brain("data/wholebrain.npz", device="cuda", impl="csr")
    show("after load")

    circuit._ensure_csr()
    show("after CSR structure")

    activity = torch.rand(streams, circuit.n_neurons, device="cuda")
    values = circuit.prepare_synapses()
    show("after prepare_synapses")

    print("\n--- forward, no grad ---")
    with torch.no_grad():
        out = circuit.synaptic_current_csr(activity, values)
    torch.cuda.synchronize()
    show("after no-grad CSR matmul")
    del out
    torch.cuda.empty_cache()
    show("after empty_cache")

    print("\n--- forward with grad ---")
    act_g = activity.clone().requires_grad_(True)
    val_g = values.clone().requires_grad_(True)
    out = circuit.synaptic_current_csr(act_g, val_g)
    torch.cuda.synchronize()
    show("after grad-enabled CSR matmul")

    print("\n--- backward ---")
    torch.cuda.reset_peak_memory_stats()
    out.sum().backward()
    torch.cuda.synchronize()
    show("after backward")
    print(
        f"   peak reserved during fwd+bwd: "
        f"{torch.cuda.memory_stats().get('reserved_bytes.all.peak', 0) / 1024**3:.2f} GiB"
    )

    print("\n--- is it the transpose matmul only? ---")
    torch.cuda.empty_cache()
    grad_out = torch.rand(streams, circuit.n_neurons, device="cuda")
    sp_t = torch.sparse_csr_tensor(
        circuit.csr_crow_t,
        circuit.csr_col_t,
        values[circuit.order_t],
        size=(circuit.n_neurons, circuit.n_neurons),
        check_invariants=False,
    )
    show("after building transpose CSR")
    with torch.no_grad():
        r = torch.sparse.mm(sp_t, grad_out.t().contiguous()).t()
    torch.cuda.synchronize()
    show("after transpose matmul (no grad)")
    del r, sp_t, grad_out
    torch.cuda.empty_cache()
    show("after cleanup")

    print("\n--- dense equivalent for scale ---")
    a = torch.rand(streams, 512, device="cuda")
    b = torch.rand(512, 4096, device="cuda")
    with torch.no_grad():
        _ = a @ b
    show("after a small dense matmul")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
