"""Separate the cost of building a CSR tensor from the cost of the sparse matmul.

The first CSR attempt was slower than gather/scatter end to end, which only makes
sense if the per-call construction dominates. This measures each piece so the
overhead can be removed rather than guessed at.
"""

from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from flybrain.device import tune_rocm  # noqa: E402
from flybrain.wholebrain import load_whole_brain  # noqa: E402


def bench(fn, *, warmup: int = 3, iters: int = 15) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iters * 1e3


def main() -> int:
    tune_rocm()
    circuit, meta, _ = load_whole_brain("data/wholebrain.npz", device="cuda")
    n = circuit.n_neurons
    nnz = circuit.pre.numel()
    print(f"nnz {nnz:,} neurons {n:,}")

    post = circuit.post.long()
    pre = circuit.pre.long()
    order = torch.argsort(post, stable=True)
    col = pre[order]
    counts = torch.bincount(post[order], minlength=n)
    crow = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
    torch.cumsum(counts, 0, out=crow[1:])
    del counts
    print("CSR structure built")

    for streams in (16, 32):
        print(f"\n=== streams={streams} ===")
        weights_perm = circuit.weight[order]
        dense = torch.rand(n, streams, device="cuda")

        # (a) permute the per-edge values: needed because only values change per step
        ms = bench(lambda: circuit.weight[order] * 1.0)
        print(f"  permute values (nnz gather)          {ms:8.3f} ms")

        # (b) build the sparse tensor from precomputed int64 indices
        values = weights_perm
        ms = bench(
            lambda: torch.sparse_csr_tensor(crow, col, values, size=(n, n), check_invariants=False)
        )
        print(f"  sparse_csr_tensor construction       {ms:8.3f} ms")

        sparse = torch.sparse_csr_tensor(crow, col, values, size=(n, n), check_invariants=False)

        # (c) the matmul alone
        ms = bench(lambda: torch.sparse.mm(sparse, dense))
        print(f"  torch.sparse.mm alone                {ms:8.3f} ms")

        # (d) int32 -> int64 index conversion, the suspected overhead
        crow32, col32 = crow.to(torch.int32), col.to(torch.int64)
        ms = bench(lambda: crow32.long())
        print(f"  crow32.long()                        {ms:8.3f} ms")
        ms = bench(lambda: col32.long())
        print(f"  col32.long()                         {ms:8.3f} ms")

        # (e) building the CSR out of a COO each call (the naive route)
        indices = torch.stack([post, pre])
        ms = bench(lambda: torch.sparse_coo_tensor(indices, circuit.weight, (n, n)).to_sparse_csr())
        print(f"  coo -> csr conversion                {ms:8.3f} ms")

        # (f) end-to-end: what the model would actually pay per sweep
        def full_csr():
            v = circuit.weight[order]
            sp = torch.sparse_csr_tensor(crow, col, v, size=(n, n), check_invariants=False)
            return torch.sparse.mm(sp, dense)

        ms = bench(full_csr)
        print(f"  END-TO-END csr sweep                 {ms:8.3f} ms")

        # (g) end-to-end gather/scatter
        act = dense.t().contiguous()
        syn = circuit.weight

        def full_index_add():
            out = torch.zeros_like(act)
            chunk = 4_000_000
            for start in range(0, nnz, chunk):
                stop = min(start + chunk, nnz)
                out.index_add_(1, post[start:stop], act[:, pre[start:stop]] * syn[start:stop])
            return out

        ms = bench(full_index_add)
        print(f"  END-TO-END index_add sweep           {ms:8.3f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
