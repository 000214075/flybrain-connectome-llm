"""The whole fly brain as a GPU-executable recurrent circuit.

Unlike `MushroomBody`, which wires one sublayer from a ~5,000-neuron pathway,
this module runs *every* neuron and *every* synapse of the MaleCNS connectome
(164,587 cells, ~25.5M neuron-to-neuron connections) inside the language model's
forward pass.

Sign conventions come from the data, not from a guess: each neuron's contribution
is signed by its predicted neurotransmitter, so GABAergic and glutamatergic cells
inhibit their targets. The sign is a fixed buffer -- training cannot learn the
inhibition away -- while the release *magnitude* per neuron is learnable, because
synaptic strength is exactly the unknown biophysics the connectome does not hold.

Performance notes, all measured on the RX 7900 XTX:
  * `torch.sparse.mm` on a CSR matrix runs the sweep at ~368 GB/s against ~112 GB/s
    for an equivalent gather/scatter with `index_add_`, so CSR is used.
  * fp32 is the fast path. bf16 sparse matmul is unoptimised in this build and
    measures *slower* than fp32 (22.9 ms against 6.4 ms at 16 streams).
  * `torch.sparse_csr_tensor` validates the index structure by default, which
    costs ~30 ms per call here -- four times the matmul it guards -- so
    `check_invariants=False` is set.
  * the autograd backward of `sparse.mm` densifies 164587^2 and requests ~101 GiB,
    so the sweep implements its own backward: the input gradient comes from a
    transpose CSR matmul, the value gradient from a chunked per-edge reduction.

Input and output interfaces are factorised through a low-rank bottleneck: a dense
`d_model x 164587` projection would cost 84M parameters and ~1.4 TFLOP per forward
pass on its own.

Causality is preserved by reading out the brain state *before* folding in the
current chunk, so a token never sees its own future. That is the standard chunked
formulation used by state-space models, and at generation time (chunk size 1) it
reduces to a plain token-by-token brain recurrence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .neurons import LIFLayer

# Bound on how many synapses are gathered at once, so the (chunk x streams)
# temporary stays flat regardless of how large the connectome is.
DEFAULT_EDGE_CHUNK = 4_000_000


def effective_rank(matrix: Tensor) -> int:
    """Numerical rank: how many independent directions a matrix really carries.

    The parameter count is not the bandwidth. A (164587, 64) projection that has
    collapsed onto two directions is a 2-dimensional interface no matter how many
    numbers are in it, and only this number shows it.
    """
    if matrix.numel() == 0:
        return 0
    m = matrix.detach().float()
    gram = m @ m.t() if m.shape[0] <= m.shape[1] else m.t() @ m
    eigenvalues = torch.linalg.eigvalsh(gram)
    tolerance = eigenvalues.max() * max(m.shape) * torch.finfo(torch.float32).eps
    return int((eigenvalues > tolerance).sum())


@dataclass
class WholeBrainMeta:
    n_neurons: int
    nnz: int
    excitatory: int
    inhibitory: int
    modulatory: int
    source: str

    @classmethod
    def from_dict(cls, raw: dict) -> "WholeBrainMeta":
        return cls(
            n_neurons=int(raw["n_neurons"]),
            nnz=int(raw["nnz"]),
            excitatory=int(raw.get("excitatory", 0)),
            inhibitory=int(raw.get("inhibitory", 0)),
            modulatory=int(raw.get("modulatory", 0)),
            source=str(raw.get("source", "unknown")),
        )


class _ConnectomeSweep(torch.autograd.Function):
    """out[s, i] = sum over edges (j -> i) of activity[s, j] * value[edge].

    Implemented as a CSR sparse matmul in both directions. The backward avoids
    `sparse.mm`'s own autograd, which densifies the matrix and needs ~101 GiB.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        activity: Tensor,
        values: Tensor,
        crow: Tensor,
        col: Tensor,
        crow_t: Tensor,
        col_t: Tensor,
        order_t: Tensor,
        pre: Tensor,
        post: Tensor,
        edge_chunk: int,
    ) -> Tensor:
        n_neurons = crow.numel() - 1
        sparse = torch.sparse_csr_tensor(
            crow, col, values, size=(n_neurons, n_neurons), check_invariants=False
        )
        out = torch.sparse.mm(sparse, activity.t().contiguous()).t()
        ctx.save_for_backward(activity, values)
        ctx.sweep = (crow_t, col_t, order_t, pre, post, edge_chunk, n_neurons)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        activity, values = ctx.saved_tensors
        crow_t, col_t, order_t, pre, post, edge_chunk, n_neurons = ctx.sweep

        grad_activity = None
        if ctx.needs_input_grad[0]:
            # The transpose of the connectome scatters the output gradient back
            # onto the pre-synaptic neurons: grad_act = W^T @ grad_out.
            sparse_t = torch.sparse_csr_tensor(
                crow_t,
                col_t,
                values[order_t],
                size=(n_neurons, n_neurons),
                check_invariants=False,
            )
            grad_activity = torch.sparse.mm(sparse_t, grad_out.t().contiguous()).t()

        grad_values = None
        if ctx.needs_input_grad[1]:
            # Per-edge gradient: a dot product over streams, reduced contiguously
            # so no atomics are involved.
            grad_values = torch.empty_like(values)
            nnz = pre.numel()
            for start in range(0, nnz, edge_chunk):
                stop = min(start + edge_chunk, nnz)
                grad_values[start:stop] = (
                    activity[:, pre[start:stop]] * grad_out[:, post[start:stop]]
                ).sum(dim=0)

        return grad_activity, grad_values, None, None, None, None, None, None, None, None


class _ConnectomeSweepScaled(torch.autograd.Function):
    """The sweep with the sparse matrix held constant, so neither side rebuilds it.

    An edge transmits `weight * sign[pre] * release_gain[pre]`, and the first two
    factors are frozen buffers, so the only factor training moves is indexed by the
    *source* neuron. Applying it to the activity instead of to the edges,

        out[target, s] = sum_e base[e] * (gain[pre[e]] * activity[pre[e], s])

    is the same sum of the same products in a different order: exact, not an
    approximation. What that buys is that the sparse matrix stops depending on the
    parameters at all, which removes both the 25.5M-element rebuild every sweep and
    the 25.5M-element per-edge gradient in the backward -- the gradient onto
    `raw_gain` becomes a per-neuron reduction instead. Measured on the real
    connectome at 32 streams, one training step costs 54.7 ms here against 27.6 ms
    on this path.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        activity: Tensor,
        gain: Tensor,
        crow: Tensor,
        col: Tensor,
        crow_t: Tensor,
        col_t: Tensor,
        base: Tensor,
        base_t: Tensor,
    ) -> Tensor:
        n_neurons = crow.numel() - 1
        scaled = gain * activity
        sparse = torch.sparse_csr_tensor(
            crow, col, base, size=(n_neurons, n_neurons), check_invariants=False
        )
        out = torch.sparse.mm(sparse, scaled.t().contiguous()).t()
        ctx.save_for_backward(activity, gain, scaled)
        ctx.sweep = (crow_t, col_t, base_t, n_neurons)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        activity, gain, scaled = ctx.saved_tensors
        crow_t, col_t, base_t, n_neurons = ctx.sweep
        sparse_t = torch.sparse_csr_tensor(
            crow_t, col_t, base_t, size=(n_neurons, n_neurons), check_invariants=False
        )
        # Only the constant matrix is left on the sparse pointer, so this one
        # sparse matmul is the whole backward: no per-edge term to reduce.
        grad_scaled = torch.sparse.mm(sparse_t, grad_out.t().contiguous()).t()
        grad_activity = grad_scaled * gain
        grad_gain = (grad_scaled * activity).sum(dim=0)
        return grad_activity, grad_gain, None, None, None, None, None, None


class _ConnectomeScatter(torch.autograd.Function):
    """out[s, post] += activity[s, pre] * value, with a backward that recomputes.

    Same arithmetic as the gather/scatter sweep, but it does not let autograd keep
    the (batch, edges) gather. That tensor is the reason eight sweeps did not fit:
    autograd saves one per edge chunk per sweep, 16 streams x 25,563,096 edges x 4 B
    = 1.6 GiB per sweep, ~13 GiB across eight -- measured as `HIP out of memory` on
    a 23.98 GiB card. Here only the population state (batch x 164,587, ~10 MiB) is
    saved and the two gathers are done again in the backward, for one extra pass
    over the edge list.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        activity: Tensor,
        values: Tensor,
        pre: Tensor,
        post: Tensor,
        n_neurons: int,
        edge_chunk: int,
    ) -> Tensor:
        out = activity.new_zeros(activity.shape[0], n_neurons)
        nnz = pre.numel()
        for start in range(0, nnz, edge_chunk):
            stop = min(start + edge_chunk, nnz)
            out.index_add_(1, post[start:stop].long(), activity[:, pre[start:stop].long()] * values[start:stop])
        ctx.save_for_backward(activity, values, pre, post)
        ctx.edge_chunk = edge_chunk
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        activity, values, pre, post = ctx.saved_tensors
        grad_activity = torch.zeros_like(activity) if ctx.needs_input_grad[0] else None
        grad_values = torch.empty_like(values) if ctx.needs_input_grad[1] else None
        nnz = pre.numel()
        for start in range(0, nnz, ctx.edge_chunk):
            stop = min(start + ctx.edge_chunk, nnz)
            sources = pre[start:stop].long()
            targets = post[start:stop].long()
            if grad_values is not None:
                grad_values[start:stop] = (activity[:, sources] * grad_out[:, targets]).sum(dim=0)
            if grad_activity is not None:
                grad_activity.index_add_(1, sources, grad_out[:, targets] * values[start:stop])
        return grad_activity, grad_values, None, None, None, None


class WholeBrainCircuit(nn.Module):
    """One synaptic step across the entire connectome."""

    def __init__(
        self,
        pre: Tensor,
        post: Tensor,
        weight: Tensor,
        sign: Tensor,
        *,
        tau_m: float = 2.0,
        v_th: float = 0.5,
        edge_chunk: int = DEFAULT_EDGE_CHUNK,
        impl: str = "csr",
    ) -> None:
        super().__init__()
        self.n_neurons = int(sign.numel())
        self.edge_chunk = edge_chunk
        self.impl = impl
        # Sub-threshold population state from the last sweep, set by `step`.
        self.last_analog: Tensor | None = None
        # Linear pre-activation (the synaptic current reaching each cell, before the
        # threshold) from the last sweep. Kept because the spike and the rate both
        # throw information away: measured on this connectome, a single sweep leaves
        # only ~0.3% of the population in the responsive band around the threshold,
        # so a difference between two inputs survives in ~500 of 164,587 neurons.
        # The current is what a downstream cell actually integrates, so reading it is
        # the same circuit, read without the saturation.
        self.last_pre: Tensor | None = None
        self.register_buffer("pre", pre.to(torch.int32), persistent=True)
        self.register_buffer("post", post.to(torch.int32), persistent=True)
        self.register_buffer("weight", weight.to(torch.float32), persistent=True)
        # sign is fixed: it encodes real inhibition and must not be trained away.
        self.register_buffer("sign", sign.to(torch.float32), persistent=True)
        # Release magnitude per neuron, initialised so the effective gain starts at
        # 1 for every cell: the analogue of learning unknown synaptic biophysics.
        # The softplus is offset so the magnitude cannot be driven to zero -- a
        # circuit whose synapses have all been scaled away is not running, and
        # keeping the brain alive is a hard requirement of this project.
        self.raw_gain = nn.Parameter(torch.full((self.n_neurons,), 0.0))
        self.raw_step = nn.Parameter(torch.tensor(0.0))
        # Floors on the three factors that scale the circuit: synaptic release, the
        # membrane integration rate, and the token drive. Each factor is its floor
        # plus a trainable part, so training can raise them but can never multiply
        # the circuit down to silence. The floors are set at the level the circuit
        # needs to cross the firing threshold -- a factor of one on the drive and on
        # the integration rate still fires, anything below does not.
        #
        # How much the model *uses* the brain is a separate question, left to the
        # read-out projection, which is free to shrink. Running the brain and using
        # the brain are deliberately different decisions.
        self.gain_floor = 0.5
        self.step_floor = 1.0
        # A short membrane time constant keeps the population responsive within the
        # few sweeps a forward pass can afford. With the usual 20 ms constant a
        # neuron would need ~20 steps to reach threshold.
        self.lif = LIFLayer(self.n_neurons, steps=1, tau_m=tau_m, v_th=v_th)
        self._csr_ready = False

    @property
    def step_size(self) -> Tensor:
        """Current-to-membrane scaling, floored so integration cannot be halted."""
        return F.softplus(self.raw_step) + self.step_floor

    @property
    def release_gain(self) -> Tensor:
        """Per-neuron release magnitude, floored so the circuit cannot be muted."""
        return F.softplus(self.raw_gain) + self.gain_floor

    # -- synapse values -----------------------------------------------------

    def effective_synapses(self) -> Tensor:
        """weight * sign[pre] * release_gain[pre], one value per edge."""
        pre = self.pre.long()
        return self.weight * self.sign[pre] * self.release_gain[pre]

    def out_strength(self) -> Tensor:
        """Total synaptic strength leaving each neuron, magnitudes only.

        This is the per-neuron vector that turns "which cells fired" into "what
        fraction of the connectome's synapses transmitted", without an O(edges)
        gather: sum_s w_s * a_pre(s) == sum_j a_j * out_strength_j.
        """
        cached = getattr(self, "_out_strength_buf", None)
        if cached is None:
            buf = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.weight.device)
            buf.index_add_(0, self.pre.long(), self.weight)
            self.register_buffer("_out_strength_buf", buf, persistent=False)
            cached = buf
        return cached

    def in_strength(self) -> Tensor:
        """Total synaptic strength arriving at each neuron, magnitudes only."""
        cached = getattr(self, "_in_strength_buf", None)
        if cached is None:
            buf = torch.zeros(self.n_neurons, dtype=torch.float32, device=self.weight.device)
            buf.index_add_(0, self.post.long(), self.weight)
            self.register_buffer("_in_strength_buf", buf, persistent=False)
            cached = buf
        return cached

    def prepare_synapses(self) -> Tensor:
        """Per-edge values in the layout the active implementation wants.

        Computed once per forward pass: the tensor covers every edge, so redoing it
        for each of several sweeps would add traffic without changing anything.
        """
        if self.impl not in ("csr", "csr-scaled"):
            return self.effective_synapses()
        self._ensure_csr()
        return self.weight_perm * self.sign[self.pre_perm] * self.release_gain[self.pre_perm]

    @property
    def needs_edge_values(self) -> bool:
        """Whether the active implementation consumes `prepare_synapses`.

        The scaled sweep keeps its own constant copy of the untrainable part of
        every edge, so building the 25.5M-element values tensor would be work whose
        result is thrown away.
        """
        return self.impl != "csr-scaled"

    # -- connectome structure ----------------------------------------------

    def _ensure_csr(self) -> None:
        """Build the CSR structures for the connectome and its transpose.

        Edges are sorted by post-synaptic neuron so the scatter becomes CSR rows.
        Only the values change between steps, so both index structures and the
        pre-permuted weights are built once and reused.

        The backward feeds `values[order_t]` to the transpose, and `values` is in
        the order permutation, so the transpose has to be built in that same
        permutation: `argsort` of the permuted `pre`, not of the raw `pre`. Building
        it from the raw array pairs each edge's value with a different edge -- a
        silent wrong gradient rather than an error, only visible when the incoming
        population state carries gradient (every token after the first).
        """
        if self._csr_ready:
            return
        device = self.post.device
        post = self.post.long()
        pre = self.pre.long()

        order = torch.argsort(post, stable=True)
        crow = torch.zeros(self.n_neurons + 1, dtype=torch.int64, device=device)
        torch.cumsum(torch.bincount(post[order], minlength=self.n_neurons), 0, out=crow[1:])
        pre_perm = pre[order]
        post_perm = post[order]

        order_t = torch.argsort(pre_perm, stable=True)
        crow_t = torch.zeros(self.n_neurons + 1, dtype=torch.int64, device=device)
        torch.cumsum(
            torch.bincount(pre_perm[order_t], minlength=self.n_neurons), 0, out=crow_t[1:]
        )

        self.register_buffer("csr_crow", crow, persistent=False)
        self.register_buffer("csr_col", pre_perm, persistent=False)
        self.register_buffer("csr_crow_t", crow_t, persistent=False)
        self.register_buffer("csr_col_t", post_perm[order_t], persistent=False)
        self.register_buffer("order_t", order_t, persistent=False)
        self.register_buffer("weight_perm", self.weight[order], persistent=False)
        self.register_buffer("pre_perm", pre_perm, persistent=False)
        self.register_buffer("post_perm", post_perm, persistent=False)
        # The part of each edge's value that training cannot move. Held in the order
        # permutation so the scaled sweep can build its constant matrix without a
        # gather, and transposed so its backward needs no gather either.
        base = self.weight[order] * self.sign[pre_perm]
        self.register_buffer("csr_base", base, persistent=False)
        self.register_buffer("csr_base_t", base[order_t].contiguous(), persistent=False)
        self._csr_ready = True

    # -- implementations ----------------------------------------------------

    def synaptic_current_csr(self, activity: Tensor, synapses: Tensor) -> Tensor:
        """Connectome sweep as CSR sparse matmuls; `activity` is (streams, n_neurons)."""
        self._ensure_csr()
        return _ConnectomeSweep.apply(
            activity,
            synapses,
            self.csr_crow,
            self.csr_col,
            self.csr_crow_t,
            self.csr_col_t,
            self.order_t,
            self.pre_perm,
            self.post_perm,
            self.edge_chunk,
        )

    def synaptic_current_index_add(self, activity: Tensor, synapses: Tensor) -> Tensor:
        """Sweep as gather + scatter-add, chunked to bound memory. The fallback path.

        Runs through `_ConnectomeScatter` so the backward recomputes instead of
        keeping a (batch, edges) gather per chunk.
        """
        return _ConnectomeScatter.apply(
            activity, synapses, self.pre, self.post, self.n_neurons, self.edge_chunk
        )

    def synaptic_current_scaled(self, activity: Tensor) -> Tensor:
        """Sweep with a constant sparse matrix and the release gain on the activity.

        `synapses` is not an argument because this path never needs one: the only
        trainable factor per edge is the release gain of its source neuron, and that
        is applied to `activity` before the matmul.
        """
        self._ensure_csr()
        return _ConnectomeSweepScaled.apply(
            activity,
            self.release_gain,
            self.csr_crow,
            self.csr_col,
            self.csr_crow_t,
            self.csr_col_t,
            self.csr_base,
            self.csr_base_t,
        )

    def synaptic_current(
        self, activity: Tensor, synapses: Tensor | None = None, *, impl: str | None = None
    ) -> Tensor:
        """One sweep across every synapse in the connectome.

        `synapses` may be omitted for the implementations that do not consume it.
        Falls back to gather/scatter if the sparse kernel is unavailable for the
        active dtype or layout, rather than failing the whole training run.
        """
        choice = impl or self.impl
        if choice == "csr-scaled":
            try:
                return self.synaptic_current_scaled(activity)
            except RuntimeError:
                self.impl = "index_add"
        elif choice == "csr":
            try:
                return self.synaptic_current_csr(activity, synapses)
            except RuntimeError:
                self.impl = "index_add"
        if synapses is None:
            synapses = self.prepare_synapses()
        return self.synaptic_current_index_add(activity, synapses)

    # -- neuron dynamics ----------------------------------------------------

    def step(
        self, state: Tensor, drive: Tensor, synapses: Tensor | None, *, spiking: bool = True
    ) -> Tensor:
        """Advance every neuron by one time step of LIF dynamics.

        The sweep runs with autocast *off*, in fp32. Two measured reasons, both on
        the connectome's own scale:

        * this ROCm build raises `hipsparseSpMM not implemented` for bf16 sparse
          matmul, so the fp32 path is the only one that exists on the GPU;
        * leaving it inside autocast is not merely imprecise, it corrupts the
          backward. Measured on CPU bf16 autocast, the per-edge gradient buffer came
          back with a gradient of 4.9e29 for `raw_gain` on 3 of 12 identical steps,
          and one run died with `INDICES element is out of DATA bounds`. The same
          loop with autocast disabled was bit-identical on all 12. The population
          state is a few megabytes, so running it in fp32 costs nothing.
        """
        dtype = state.dtype
        with torch.autocast(device_type=state.device.type, enabled=False):
            current = self.synaptic_current(
                state.float(), None if synapses is None else synapses.float()
            ) + drive.float()
            pre_spike = current * self.step_size
            out = self.lif(pre_spike, spiking=spiking)
        # Sub-threshold population state, for the read-out only: a spike is a
        # threshold crossing, so it discards how far each neuron was from firing.
        # Measured on this project's model, that quantisation made the population
        # state bit-identical across a batch of eight different documents.
        self.last_analog = torch.sigmoid(4.0 * (pre_spike - self.lif.v_th)).to(dtype)
        self.last_pre = pre_spike
        return out.to(dtype)

    def forward(
        self, state: Tensor, drive: Tensor, synapses: Tensor | None, *, spiking: bool = True
    ) -> Tensor:
        return self.step(state, drive, synapses, spiking=spiking)

    def initial_state(self, batch: int, device, dtype) -> Tensor:
        return torch.zeros(batch, self.n_neurons, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        return (
            f"n_neurons={self.n_neurons:,} nnz={self.pre.numel():,} impl={self.impl}"
        )


def load_port_mask(path: str, n_neurons: int) -> Tensor:
    """Load a boolean port mask built by `scripts/build_port_masks.py`.

    The mask is a 0/1 float vector over the connectome's own neuron order, so the
    model never has to know anything about cell-type annotations.
    """
    import numpy as np  # local: this module is otherwise pure torch

    with np.load(path) as handle:
        mask = handle["mask"]
    if mask.shape != (n_neurons,):
        raise ValueError(f"{path}: mask has shape {mask.shape}, expected ({n_neurons},)")
    return torch.from_numpy(mask.astype(np.float32))


class BrainPathway(nn.Module):
    """The whole brain as a layer in a language model.

    Attach after the transformer stack. It keeps a state over every neuron in the
    connectome, folds in a projection of each chunk of tokens, iterates the
    connectome a configured number of times, and projects the state back to the
    model width.

    `input_mask_path` / `output_mask_path` optionally restrict which neurons the
    tokens may drive and which the read-out may see. Left as None (the default) the
    model is the one that already exists, untouched.
    """

    def __init__(
        self,
        circuit: WholeBrainCircuit,
        d_model: int,
        *,
        rank: int = 64,
        chunks: int = 2,
        iters: int = 2,
        spiking: bool = True,
        dropout: float = 0.0,
        analog_readout: bool = True,
        analog_mode: str = "rate",
        input_mask_path: str | None = None,
        output_mask_path: str | None = None,
    ) -> None:
        super().__init__()
        if chunks < 1:
            raise ValueError("chunks must be >= 1")
        if analog_mode not in ("spike", "rate", "current"):
            raise ValueError("analog_mode must be 'spike', 'rate' or 'current'")
        n_neurons = circuit.n_neurons
        # Registered only when a mask is actually wanted, and `persistent=False` so a
        # checkpoint from an unmasked run still loads. A buffer of ones would be
        # simpler but is not the same model: multiplying by 1.0 is exact in IEEE yet
        # still an extra op in the graph, and the `all` arm of a port comparison has
        # to be bit-identical to the model that already exists. `_ports()` below
        # therefore reads the buffer if present and skips the multiply if not.
        if input_mask_path:
            self.register_buffer(
                "input_mask", load_port_mask(input_mask_path, n_neurons), persistent=False
            )
        if output_mask_path:
            self.register_buffer(
                "output_mask", load_port_mask(output_mask_path, n_neurons), persistent=False
            )
        self.circuit = circuit
        self.chunks = chunks
        self.iters = iters
        self.spiking = spiking
        n_neurons = circuit.n_neurons
        # Factorised interfaces: d_model -> rank -> n_neurons and back.
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, n_neurons, bias=False)
        self.read_down = nn.Linear(n_neurons, rank, bias=False)
        self.read_up = nn.Linear(rank, d_model, bias=False)
        self.norm_in = nn.LayerNorm(d_model)
        # Population-level normalisation of the input current, with no learnable
        # affine terms. Without it the drive lands orders of magnitude below the
        # firing threshold, every neuron stays silent, the surrogate gradient is
        # never non-zero and the circuit cannot learn anything at all. Homeostatic
        # scaling of this kind is a real property of neural circuits.
        # eps is deliberately small: `up` is initialised so its output already has
        # a healthy variance, and a default eps would dominate the normaliser.
        self.norm_drive = nn.LayerNorm(n_neurons, eps=1e-6, elementwise_affine=False)
        # Floored, so the drive cannot be scaled away to silence the circuit.
        self.raw_drive_scale = nn.Parameter(torch.tensor(0.0))
        self.drive_floor = 1.0
        # Reading the brain out through the sub-threshold current as well as the
        # spike train (see `analog_read`). Starts neutral so an existing checkpoint
        # loads unchanged, and is learnable because how much of the analogue state is
        # useful is an empirical question.
        self.analog_readout = analog_readout
        self.analog_mode = analog_mode
        self.raw_analog = nn.Parameter(torch.full((), -1.0))
        # Spontaneous activity: the connectome starts from a learned resting state
        # rather than from silence. A real fly brain is never at exactly zero
        # activity, and the alternative -- starting from zeros -- makes the first
        # chunk's read-out a constant zero, so those tokens get nothing at all from
        # the brain. The parameter is squashed into [0, 1] because it represents a
        # firing probability per neuron.
        self.raw_rest = nn.Parameter(torch.full((n_neurons,), -2.944))  # sigmoid ~ 0.05
        self.dropout = nn.Dropout(dropout)
        # Firing fraction of the last forward pass, kept as a side channel so the
        # trainer can regularise it. Without a homeostatic term the optimiser
        # silences the brain early on -- treating its contribution as noise is
        # cheaper than learning to use it -- and the whole connectome stops
        # participating.
        self.last_activity: Tensor | None = None
        # Most recent population state, kept detached from the graph so reporting
        # utilisation does not extend the backward pass.
        self.last_state: Tensor | None = None
        # Number of connectome sweeps and their summed mean activity, both reset at
        # the start of every forward pass so the trainer can report real utilisation
        # rather than the last sweep it happened to run.
        self.sweeps = 0
        self._activity_total: Tensor | None = None
        # Last token drive and last read-out, graphs intact: the participation
        # penalty is computed from them, so they must not be detached.
        self._last_drive: Tensor | None = None
        self._last_readout: Tensor | None = None

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.read_down.weight, std=0.02)
        # `up` must land the current at the firing threshold: too small and the
        # circuit never spikes, too large and every neuron fires every step.
        nn.init.normal_(self.up.weight, std=1.0 / (rank**0.5))
        nn.init.normal_(self.read_up.weight, std=0.02 / (rank**0.5))

    @property
    def drive_scale(self) -> Tensor:
        return F.softplus(self.raw_drive_scale) + self.drive_floor

    @property
    def analog_read(self) -> Tensor:
        """Weight on the analogue population signal when reading the brain out.

        The spike train is binary, and that is what makes the read-out constant: a
        neuron's drive has to move by a whole threshold for its spike to change, so
        small per-token differences vanish. Measured on this project's model, the
        population state after a forward pass was **bit-identical across all eight
        streams of a batch** (`(state - state[0]).abs().max() == 0.0`) even though
        their drives spanned 3.8 independent directions. Reading out from the
        pre-spike current as well keeps those differences: it is the same neurons,
        the same connectome, but the read-out is no longer quantised.
        """
        if not self.analog_readout or self.analog_mode == "spike":
            return torch.zeros((), device=self.raw_analog.device)
        return F.softplus(self.raw_analog)

    def read_state(self, state: Tensor) -> Tensor:
        """The population state as the language model sees it.

        Three ways to read the same circuit, all of them the connectome:

        * `spike`   -- the thresholded population state alone;
        * `rate`    -- that state plus the sub-threshold current, sharpened by a
          sigmoid around the threshold;
        * `current` -- that state plus the linear pre-activation, normalised across
          the population per stream.

        The last one exists because a rate is a saturating function of the current,
        and this connectome drives most of its cells far from the threshold: a sweep
        leaves only ~0.3% of the population in the responsive band, so a difference
        between two inputs survives in a few hundred neurons out of 164,587. Taking
        the sigmoid of a saturated current cannot recover it, but the current itself
        still holds it, and the current is what a downstream cell integrates.
        """
        if self.analog_mode == "spike" or not self.analog_readout:
            return state
        if self.analog_mode == "current":
            pre = self.circuit.last_pre
            if pre is None:
                return state
            # Normalised across the population, per stream, with no trainable affine
            # terms -- the same parameter-free normalisation `norm_drive` applies to
            # the input, and for the same reason: it fixes the scale of a signal
            # whose absolute magnitude is a property of the wiring, not of the
            # input. Normalising per stream cannot mix streams, so it cannot leak
            # information between positions.
            analog = F.layer_norm(pre.float(), (self.circuit.n_neurons,), eps=1e-6)
            return state + self.analog_read * analog.to(state.dtype)
        analog = self.circuit.last_analog
        if analog is None:
            return state
        return state + self.analog_read * analog.to(state.dtype)

    @property
    def resting_state(self) -> Tensor:
        """Spontaneous firing probability per neuron, in [0, 1]."""
        return torch.sigmoid(self.raw_rest)

    def initial_state(self, batch: int, device=None, dtype=None) -> Tensor:
        """The brain starts from its learned spontaneous activity, tiled per stream."""
        rest = self.resting_state.to(device=device, dtype=dtype)
        return rest.unsqueeze(0).expand(batch, -1).contiguous()

    def _input_ports(self, drive: Tensor) -> Tensor:
        """Restrict the token drive to the input port neurons, if ports are set.

        Applied after `norm_drive`, not before: the normaliser is a population
        statistic, so masking first would change the scale of the drive rather than
        only who receives it -- a different experiment.
        """
        mask = self._buffers.get("input_mask")
        if mask is None:
            return drive
        return drive * mask.to(dtype=drive.dtype, device=drive.device)

    def _output_ports(self, state: Tensor) -> Tensor:
        """Restrict what the read-out may see, if output ports are set."""
        mask = self._buffers.get("output_mask")
        if mask is None:
            return state
        return state * mask.to(dtype=state.dtype, device=state.device)

    def _drive(self, x: Tensor) -> Tensor:
        """Project a chunk of tokens (B, c, d) onto every neuron: (B, n_neurons)."""
        pooled = self.norm_in(x).mean(dim=1)
        current = self.up(self.down(pooled))
        drive = self._input_ports(self.norm_drive(current) * self.drive_scale)
        self._last_drive = drive
        return drive

    def _read(self, state: Tensor) -> Tensor:
        """Project a brain state (B, n_neurons) back to (B, d_model)."""
        return self.read_up(self.read_down(self._output_ports(state)))

    def read(self, state: Tensor, *, dtype: torch.dtype | None = None) -> Tensor:
        """The read half of the brain interface: state (B, n_neurons) -> (B, d_model)."""
        out = self._read(self.read_state(state.to(dtype) if dtype is not None else state))
        self._last_readout = out
        return out

    def advance(self, state: Tensor, features: Tensor, *, iters: int | None = None) -> Tensor:
        """The write half: fold a chunk of hidden states in and sweep the connectome.

        `read` then `advance` is the causal unit of the brain -- reading first means
        a chunk can never see its own influence through the connectome.
        """
        return self.advance_drive(state, self._drive(features), iters=iters)

    def advance_drive(self, state: Tensor, drive: Tensor, *, iters: int | None = None) -> Tensor:
        """Fold an already-computed per-neuron drive in and sweep the connectome.

        Used by the connectome-only language model, where the drive for every
        position is projected up front and only the sweeps have to run in sequence.
        """
        synapses = self.circuit.prepare_synapses() if self.circuit.needs_edge_values else None
        for _ in range(max(1, iters or self.iters)):
            state = self.circuit(state, drive, synapses, spiking=self.spiking)
            self._record(state)
        return state

    def read_latent(self, state: Tensor) -> Tensor:
        """The population state as a low-rank vector: (B, n_neurons) -> (B, rank)."""
        out = self.read_down(self._output_ports(self.read_state(state)))
        # Recorded so `readout_participation` works for the connectome-only model too,
        # where the read-out is this vector rather than `read_up(read_down(...))`.
        self._last_readout = out
        return out

    def drive_from_tokens(self, embeddings: Tensor) -> Tensor:
        """Project per-token vectors (B, T, d) onto every neuron: (B, T, n_neurons).

        The whole drive for a sequence in one matmul: the connectome's sweeps have to
        be serial, but nothing else here does.
        """
        b, t, _ = embeddings.shape
        pooled = self.norm_in(embeddings.reshape(b * t, -1))
        current = self.up(self.down(pooled))
        drive = self._input_ports(self.norm_drive(current) * self.drive_scale)
        drive = drive.reshape(b, t, -1)
        # The batch's drive is the average over its positions, so the participation
        # ratio asks the question that matters: do different documents drive the
        # population to different places.
        self._last_drive = drive.mean(dim=1)
        return drive

    def _record(self, state: Tensor) -> None:
        """Track firing so the trainer can regularise it and report utilisation."""
        mean = state.mean()
        self.sweeps += 1
        self.last_activity = mean
        self.last_state = state.detach()
        self._activity_total = mean if self._activity_total is None else self._activity_total + mean

    @property
    def mean_activity(self) -> Tensor | None:
        """Firing fraction averaged over every sweep of the current forward pass."""
        if self._activity_total is None:
            return None
        return self._activity_total / max(1, self.sweeps)

    def synapse_use(self, state: Tensor) -> float:
        """Fraction of the connectome's synapses that carried a signal this sweep.

        Weighted by synaptic strength and by the state, because for a rate state the
        state *is* the probability that the cell transmits: summing `state` gives
        the expected share of the 25.5M synapses that carried something. For a binary
        spike state this is exactly the old "fraction whose source fired", since the
        state is 0 or 1. It is the number that says whether the whole brain is
        *computing* or merely being stepped: at a 10% firing rate only about a tenth
        of the synapses transmit, and the rest of the matrix is read for nothing.

        Computed through a per-neuron outgoing-strength vector, so it costs
        O(neurons) instead of materialising a 25.5M-edge gather per batch stream.
        """
        with torch.no_grad():
            strength = self.circuit.out_strength()
            active = state.detach().float().clamp(min=0.0).to(strength.dtype)
            total = strength.sum().clamp_min(1e-12)
            return float(((active * strength).sum(dim=-1) / total).mean())

    def drive_orthogonality(self) -> Tensor:
        """How far the token drive has collapsed onto one direction.

        The drive is the brain's input bandwidth: `up(down(x))` can hand the
        population any pattern in a rank-dimensional subspace. Left alone the
        optimiser collapses it. Measured on this project's own checkpoint after
        1,500 steps, `down` stayed full rank (64) while `up` fell to **rank 2**, so
        the composed map was rank 1 -- all 164,587 neurons were being driven by a
        single scalar, and the connectome could only be used as a gain knob.

        The penalty is the mean squared deviation of the column Gram matrix from the
        identity, on columns scaled to unit length so it measures pure collapse and
        not magnitude. It is ~0 for independent directions and ~1 when every
        direction is the same.
        """
        weight = self.up.weight
        if weight.shape[1] < 2:
            return torch.zeros((), device=weight.device)
        normalised = weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        gram = normalised.t() @ normalised
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).pow(2).mean()

    @staticmethod
    def participation_ratio(vectors: Tensor) -> Tensor:
        """How many independent directions a batch of vectors actually spans.

        `(sum l)^2 / sum l^2` over the eigenvalues of the vectors' Gram matrix:
        1.0 when they all point the same way, `batch` when they are independent.
        Only the variation *across* the batch counts, so the per-feature mean is
        removed first.

        Computed in fp32 outside autocast: this ROCm build has no bf16 `eigvalsh`,
        and a `(batch, batch)` Gram matrix is cheap enough that precision is free.
        """
        work = vectors.float()
        with torch.autocast(device_type=work.device.type, enabled=False):
            centred = work - work.mean(dim=0, keepdim=True)
            gram = centred @ centred.t()
            # Normalised by its own trace: the ratio is already scale-invariant, but
            # this keeps its *gradient* scale-invariant too. Without it the penalty
            # pushes on the weights in proportion to their magnitude, and a first
            # attempt with weight 1.0 drove the gradient norm to 789 and the loss
            # from 1.93 up to 3.02 in forty steps.
            gram = gram / gram.trace().clamp_min(1e-30)
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            total = eigenvalues.sum()
            ratio = total.pow(2) / eigenvalues.pow(2).sum().clamp_min(1e-30)
        return ratio.clamp(1.0, float(vectors.shape[0]))

    def drive_participation(self) -> Tensor | None:
        """Independent drive patterns per batch -- the composed map, not its factors.

        This is the measurement the factor ranks miss. After 500 fine-tuning steps
        `up` had been restored to rank 64, yet the composed map was still judged
        rank 1: the two factors were healthy and their product was not. The reason
        is that decorrelating `up`'s columns says nothing about whether *the tokens
        the model actually sees* land on different directions. This does.
        """
        if self._last_drive is None:
            return None
        return self.participation_ratio(self._last_drive)

    def readout_participation(self) -> Tensor | None:
        """Independent read-out vectors per batch: does the brain answer each token differently."""
        if self._last_readout is None:
            return None
        return self.participation_ratio(self._last_readout)

    def participation_penalty(self) -> Tensor:
        """Push both interfaces away from acting as a single shared scalar.

        Centring the batch removes one direction, so `batch - 1` independent
        directions is the most a batch of `batch` vectors can span; the penalty is
        measured against that. It is 0 when the batch spans everything left and ~0.8
        when every stream gets the same vector.
        """
        penalty = torch.zeros((), device=self.up.weight.device)
        for vectors in (self._last_drive, self._last_readout):
            if vectors is None:
                continue
            ratio = self.participation_ratio(vectors)
            span = max(1.0, float(vectors.shape[0]) - 1.0)
            penalty = penalty + (1.0 - ratio / span).clamp_min(0.0)
        return penalty

    def drive_bandwidth(self) -> tuple[int, int]:
        """Effective rank of the two drive factors, for the training log."""
        with torch.no_grad():
            return effective_rank(self.down.weight), effective_rank(self.up.weight.t())

    def run(self, x: Tensor, **kwargs) -> Tensor:
        """The brain pass over a chunk of tokens, for use inside the model."""
        out, _ = self.forward_sequence(x, **kwargs)
        return out

    def forward_sequence(
        self,
        x: Tensor,
        *,
        state: Tensor | None = None,
        chunk_size: int | None = None,
        iters: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns the modulated tokens and the final brain state.

        Pass the returned state back in with the next tokens to continue the brain
        across calls, which is what generation does.
        """
        b, t, _ = x.shape
        chunk_size = chunk_size or max(1, t // self.chunks)
        self.reset_stats()
        if state is None:
            # A fresh sequence: no sub-threshold trace to inherit. Continuing from a
            # passed-in state keeps it, which is what makes the incremental path
            # agree with the full pass chunk for chunk.
            self.start_sequence()
            state = self.initial_state(b, x.device, x.dtype)

        # Read-out and sweep are interleaved per chunk: a chunk is read from the
        # state left by the chunks before it and only then folded in, so the brain
        # cannot leak a token into its own read-out.
        outputs: list[Tensor] = []
        for start in range(0, t, chunk_size):
            chunk = x[:, start : start + chunk_size]
            outputs.append(self._read(self.read_state(state.to(x.dtype))))
            state = self.advance(state, chunk, iters=iters)
        readout = torch.stack(outputs, dim=1)  # (B, n_chunks, d_model)
        expanded = readout.repeat_interleave(chunk_size, dim=1)[:, :t]
        return x + self.dropout(expanded), state

    def reset_stats(self) -> None:
        """Start a fresh utilisation count for the forward pass about to run."""
        self.sweeps = 0
        self._activity_total = None
        self._last_drive = None
        self._last_readout = None

    def start_sequence(self) -> None:
        """Forget the sub-threshold trace left over from a previous sequence.

        The trace is handed from one chunk to the next inside a sequence, which is
        what keeps incremental decoding consistent with a full pass, so it must not
        be cleared per call. It must not leak *across* sequences either: a stale
        trace makes the same tokens score differently depending on what was run
        before them.
        """
        self.circuit.last_analog = None
        self.circuit.last_pre = None

    def activity_stats(self, state: Tensor) -> dict:
        """How much of the population is active, the check that the brain is not dead.

        What counts as active depends on the neuron model. A spike is a binary event,
        so `state > 0` is the firing fraction. A rate is a number in (0, 1] for every
        neuron including the silent ones, so `state > 0` reads 100% no matter what
        the circuit is doing and says nothing; there the fraction above half
        activation is the equivalent number. Both are reported alongside the mean, so
        a log never shows a saturated-looking figure that is only an artefact of the
        neuron model.
        """
        stats = {
            "mean_state": round(float(state.mean().item()), 4),
            "max_state": round(float(state.max().item()), 4),
            # A rate state is never exactly 0 for a silent cell, so the share of the
            # population that is effectively switched off has to be read off a
            # tolerance. It is the closest thing to "how many neurons are online":
            # `half_active_fraction` counts the loud ones, this counts the dead ones.
            "silent_fraction": round(float((state <= 1e-3).float().mean().item()), 4),
        }
        if self.spiking:
            stats["firing_fraction"] = round(float((state > 0).float().mean().item()), 4)
        else:
            stats["half_active_fraction"] = round(float((state > 0.5).float().mean().item()), 4)
        return stats


def circuit_from_state_dict(
    state: dict,
    *,
    prefix: str = "brain.circuit.",
    edge_chunk: int = DEFAULT_EDGE_CHUNK,
    impl: str = "csr",
) -> WholeBrainCircuit:
    """Rebuild a circuit from a checkpoint's buffers.

    The wiring is saved as persistent buffers, so a trained checkpoint reloads
    without needing data/wholebrain.npz again.
    """
    required = ("pre", "post", "weight", "sign")
    missing = [name for name in required if f"{prefix}{name}" not in state]
    if missing:
        raise KeyError(f"checkpoint is missing connectome buffers: {missing}")
    return WholeBrainCircuit(
        state[f"{prefix}pre"],
        state[f"{prefix}post"],
        state[f"{prefix}weight"],
        state[f"{prefix}sign"],
        edge_chunk=edge_chunk,
        impl=impl,
    )


def expand_state_dict_rank(
    state: dict[str, Tensor],
    model: nn.Module,
    *,
    prefix: str = "brain.",
    std_scale: float = 0.1,
) -> dict[str, Tensor]:
    """Widen the factorised brain projections in a checkpoint to the model's rank.

    The read-out and the token drive are the brain's bandwidth: a rank of 64 says
    at most 64 independent directions of a 164,587-neuron state can reach the
    language model, and at most 64 directions of token context can reach the brain.
    Raising the rank on an already trained checkpoint is what this makes cheap --
    the trained block keeps its values and the new slice is drawn small.

    Small, not zero: a zero slice has a zero gradient, so those units would never
    receive an update and the added bandwidth would stay dead for the whole run.
    `std_scale` sets the new slice against the reference scale -- at 1.0 the added
    units would be as loud as the trained ones and the checkpoint would count for
    nothing, at 0.0 they would never learn. A tenth keeps the function within a
    percent of what was trained while still giving the new units a live gradient.
    """
    target = model.state_dict()
    out = dict(state)
    for key, ref in target.items():
        if not key.startswith(prefix) or key not in out:
            continue
        old = out[key]
        if old.ndim != 2 or old.shape == ref.shape:
            continue
        differing = [axis for axis in (0, 1) if old.shape[axis] != ref.shape[axis]]
        if len(differing) != 1:
            continue  # not a rank change, e.g. a different neuron count
        axis = differing[0]
        new = torch.empty_like(ref)
        nn.init.normal_(new, std=std_scale * float(ref.std()) if ref.numel() else 0.0)
        keep = min(old.shape[axis], ref.shape[axis])
        if axis == 0:
            new[:keep] = old[:keep]
        else:
            new[:, :keep] = old[:, :keep]
        out[key] = new
    return out


def load_whole_brain(
    path: str,
    *,
    device: str | torch.device = "cpu",
    edge_chunk: int = DEFAULT_EDGE_CHUNK,
    impl: str = "csr",
    tau_m: float = 2.0,
    v_th: float = 0.5,
) -> tuple[WholeBrainCircuit, WholeBrainMeta, dict]:
    """Build a `WholeBrainCircuit` from the npz written by build_wholebrain.py."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        pre = torch.from_numpy(data["pre"].astype(np.int64))
        post = torch.from_numpy(data["post"].astype(np.int64))
        weight = torch.from_numpy(data["weight"].astype(np.float32))
        sign = torch.from_numpy(data["sign"].astype(np.float32))
        meta = WholeBrainMeta.from_dict(json.loads(str(data["meta"])))
        neuron_ids = torch.from_numpy(data["neuron_ids"].astype(np.int64))
    circuit = WholeBrainCircuit(
        pre, post, weight, sign, tau_m=tau_m, v_th=v_th, edge_chunk=edge_chunk, impl=impl
    ).to(device)
    return circuit, meta, {"neuron_ids": neuron_ids, "path": path}


def shuffled_edges(
    pre,
    post,
    n_neurons: int,
    *,
    seed: int = 0,
    rounds: int = 20,
    proposals: int = 2_000_000,
):
    """Rewire an edge list while keeping every neuron's in- and out-degree.

    This is the scientific control for the whole-brain circuit: same neurons, same
    synapse count, same degrees, random partners. Swapping the postsynaptic
    endpoint of two random edges leaves the multiset of presynaptic neurons
    untouched, so every cell keeps its transmitter (its sign) *and* its
    out-degree, and every target keeps its in-degree -- which is also the
    quantity `build_wholebrain.py` normalised the weights by. Synapse counts can
    therefore be reused unchanged, and the only difference between the two arms
    is *which* neurons are joined.

    Swaps that would create a self-loop or a second copy of an existing edge are
    rejected, so the result is a simple graph of exactly the same size.
    """
    import numpy as np

    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64).copy()
    nnz = post.size
    if nnz < 2:
        return post
    rng = np.random.default_rng(seed)
    keys = np.sort(pre * n_neurons + post)

    def _present(candidates) -> np.ndarray:
        pos = np.searchsorted(keys, candidates)
        inside = pos < keys.size
        return inside & (keys[np.minimum(pos, keys.size - 1)] == candidates)

    for _ in range(rounds):
        i = rng.integers(0, nnz, size=proposals)
        j = rng.integers(0, nnz, size=proposals)
        # Every edge index may take part in at most one swap per round, otherwise
        # the vectorised write below is a last-write-wins race instead of a
        # transposition (the multiset of posts would survive, but a duplicate
        # edge could slip past the check).
        both = np.concatenate([i, j])
        order = np.argsort(both, kind="stable")
        ranked = both[order]
        first = np.zeros(both.size, dtype=bool)
        first[0] = True
        np.not_equal(ranked[1:], ranked[:-1], out=first[1:])
        keep = np.zeros(both.size, dtype=bool)
        keep[order[first]] = True
        keep = keep[:proposals] & keep[proposals:]
        i, j = i[keep], j[keep]
        if i.size == 0:
            continue
        new_i, new_j = post[j], post[i]
        ok = (
            ~_present(pre[i] * n_neurons + new_i)
            & ~_present(pre[j] * n_neurons + new_j)
            & (pre[i] != new_i)
            & (pre[j] != new_j)
        )
        i, j, new_i, new_j = i[ok], j[ok], new_i[ok], new_j[ok]
        if i.size == 0:
            continue
        # Two swaps in the same round can propose the same new edge, and neither
        # of them sees the other. Keep one proposal per distinct key so the graph
        # stays simple.
        proposed = np.concatenate([pre[i] * n_neurons + new_i, pre[j] * n_neurons + new_j])
        order = np.argsort(proposed, kind="stable")
        ranked = proposed[order]
        first = np.zeros(proposed.size, dtype=bool)
        first[0] = True
        np.not_equal(ranked[1:], ranked[:-1], out=first[1:])
        once = np.zeros(proposed.size, dtype=bool)
        once[order[first]] = True
        once = once[: i.size] & once[i.size :]
        i, j, new_i, new_j = i[once], j[once], new_i[once], new_j[once]
        if i.size == 0:
            continue
        removed = np.sort(np.concatenate([pre[i] * n_neurons + post[i], pre[j] * n_neurons + post[j]]))
        post[i], post[j] = new_i, new_j
        cut = np.searchsorted(removed, keys)
        gone = removed[np.minimum(cut, removed.size - 1)] == keys
        keys = np.sort(np.concatenate([keys[~gone], pre[i] * n_neurons + new_i, pre[j] * n_neurons + new_j]))
    return post
