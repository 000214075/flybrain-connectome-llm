"""The FlyBrain language model.

A decoder-only transformer in which the feed-forward sublayer is replaced by a
mushroom-body circuit wired from real connectome data:

    x -> PN (learned projection)     [projection neurons, antennal lobe output]
      -> KC (fixed sparse matrix)    [Kenyon cells, ~10x sparse expansion]
      -> APL (global inhibition)     [single bilateral inhibitory pair]
      -> LIF spikes + k-WTA          [sparse Kenyon cell code]
      -> MBON (fixed sparse matrix)  [mushroom body output neurons]
      -> x (learned projection)

Only the input and output projections, the inhibitory gain and the LIF time
constant are learned; the wiring itself is the fly's. Three feed-forward modes
are provided so the connectome contribution can be measured rather than assumed:
`mb` (real wiring), `mb-shuffled` (same degree sequence, random partners) and
`swiglu` (no connectome at all).

Departures from the biology that exist for trainability are marked in the code.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .connectome import Pathway
from .neurons import LIFLayer, k_wta

# Above this many elements a dense wiring matrix is not worth the memory, so the
# module switches to a sparse matmul (relevant only for whole-brain subgraphs).
DENSE_ELEMENT_LIMIT = 50_000_000


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None  # grouped-query attention; None means == n_heads
    context: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10_000.0
    # feed-forward sublayer
    ffn_mode: str = "mb"  # "mb" | "mb-shuffled" | "swiglu"
    ffn_hidden: int = 0  # 0 = derive from the mushroom body width
    swiglu_mult: float = 4.0
    # mushroom body circuit
    path: str = "connectome/pathway.npz"
    mb_copies: int = 1
    mb_kc_sparsity: float = 0.05
    mb_steps: int = 2
    mb_spiking: bool = True
    mb_norm: bool = True
    mb_min_weight: float = 1.0
    mb_max_pn: int | None = None
    mb_max_kc: int | None = None
    mb_max_mbon: int | None = None
    # whole brain circuit (all 164,587 neurons of the MaleCNS connectome)
    brain_path: str = ""
    brain_rank: int = 64
    brain_chunks: int = 2
    brain_iters: int = 2
    brain_spiking: bool = True
    brain_edge_chunk: int = 4_000_000
    # "csr-scaled" (constant sparse matrix, measured 1.98x faster than "csr")
    # | "csr" | "index_add"
    brain_impl: str = "csr-scaled"
    # Read the brain out through the sub-threshold current as well as the spike
    # train. Measured on the previous model: with spikes alone the population state
    # was bit-identical across a batch of eight different documents, so the language
    # model received the same vector from the whole brain for every token.
    brain_analog_readout: bool = True
    # Which analogue signal the read-out adds on top of the thresholded population
    # state: 'rate' (the sub-threshold current through a sigmoid), 'current' (the
    # linear pre-activation, normalised across the population) or 'spike' (nothing).
    # See `BrainPathway.read_state`. The connectome-only model needs 'current',
    # because with 'rate' the input difference between two streams survives in only
    # ~0.3% of the population.
    brain_analog_mode: str = "rate"
    # Which neurons the token embedding may drive, and which the read-out may see.
    # Empty means all of them (the model as it has always been). A path is a mask
    # built by `scripts/build_port_masks.py`: the real input/output populations of
    # the connectome, or an equal-count random control. Not decoration -- the control
    # is what separates "these ports" from "fewer ports", and FlyToLLM measured the
    # port choice to matter more than the wiring itself.
    brain_input_mask: str = ""
    brain_output_mask: str = ""
    # Interleave the connectome with the stack instead of appending it after the
    # last block. Appended, no transformer layer ever sees the brain -- its output
    # only reaches the language-model head, so "running the whole brain" and "the
    # whole brain being used by the network" are different things. Interleaved, the
    # brain read-out modulates every block on every chunk, and the drive into the
    # brain comes from the stack's own output.
    brain_in_layers: bool = False
    # Which sequence model to build. "transformer" is the hybrid: attention does the
    # sequence modelling and the fly circuits ride along inside it. "connectome" is
    # the pure one in `connectome_lm.py`, where the connectome's own recurrence is
    # the only thing that mixes information across positions.
    arch: str = "transformer"
    # Unused since the connectome-only model started driving the population through
    # the brain's own `drive_from_tokens` path, with an embedding `d_model` wide.
    # Kept so model configs saved before that change still load.
    sensory_dim: int = 256
    # training switches recorded here so a checkpoint is self-describing
    init_std: float = 0.02

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


@dataclass
class ModelCache:
    """Incremental decoding state: one attention cache per block, plus the brain."""

    layers: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    brain: Tensor | None = None

    def truncated(self, context: int) -> "ModelCache":
        return ModelCache(
            layers=[(k[:, :, -context:], v[:, :, -context:]) for k, v in self.layers],
            brain=self.brain,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)


def build_rope_cache(context: int, head_dim: int, base: float, device, dtype) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(context, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    """Rotate query/key pairs. `x` is (B, H, T, Dh) with Dh even.

    `offset` is the position of the first token in `x`, which is what makes
    incremental decoding with a KV cache match a single full-sequence pass.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[offset : offset + x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + x.shape[-2]].unsqueeze(0).unsqueeze(0)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        if cfg.d_model % cfg.n_heads:
            raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
        self.head_dim = cfg.d_model // cfg.n_heads
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        b, t, _ = x.shape
        offset = past[0].shape[-2] if past is not None else 0
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin, offset), apply_rope(k, cos, sin, offset)

        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v) if use_cache else None

        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if past is None:
            # Prefill: the causal mask is implicit and kernel-fused.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        elif t == 1:
            # Decoding one token against the cache: every cached key is visible.
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            q_len, k_len = q.shape[-2], k.shape[-2]
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(
                diagonal=k_len - q_len
            )
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), present


class MushroomBody(nn.Module):
    """The connectome-wired feed-forward sublayer."""

    def __init__(self, cfg: ModelConfig, pathway: Pathway) -> None:
        super().__init__()
        self.n_copies = max(1, cfg.mb_copies)
        self.n_pn = pathway.n_pn
        self.n_kc = pathway.n_kc
        self.n_mbon = pathway.n_mbon
        self.spiking = cfg.mb_spiking
        self.norm = RMSNorm(self.n_kc) if cfg.mb_norm else None

        # Fixed wiring, replicated block-diagonally when several copies are used.
        pn_to_kc = torch.from_numpy(pathway.pn_to_kc).float()
        kc_to_mbon = torch.from_numpy(pathway.kc_to_mbon).float()
        if self.n_copies > 1:
            pn_to_kc = torch.block_diag(*([pn_to_kc] * self.n_copies))
            kc_to_mbon = torch.block_diag(*([kc_to_mbon] * self.n_copies))
        self.register_buffer("pn_to_kc", pn_to_kc, persistent=True)
        self.register_buffer("kc_to_mbon", kc_to_mbon, persistent=True)
        self.register_buffer(
            "apl_to_kc",
            torch.from_numpy(pathway.apl_to_kc).float().repeat(self.n_copies),
            persistent=True,
        )

        d_model = cfg.d_model
        self.pn_proj = nn.Linear(d_model, self.n_pn * self.n_copies, bias=False)
        self.mbon_proj = nn.Linear(self.n_mbon * self.n_copies, d_model, bias=False)
        # APL gain is a single learned scalar: the real circuit has one inhibitory
        # pair acting globally, so it should not have per-cell freedom.
        self.raw_apl_gain = nn.Parameter(torch.tensor(0.0))
        self.lif = LIFLayer(self.n_kc * self.n_copies, steps=cfg.mb_steps)
        self.k = max(1, int(round(cfg.mb_kc_sparsity * self.n_kc * self.n_copies)))

        std = cfg.init_std
        nn.init.normal_(self.pn_proj.weight, std=std)
        nn.init.normal_(self.mbon_proj.weight, std=std)
        self._sparse_cache: dict[str, Tensor] = {}

    @property
    def apl_gain(self) -> Tensor:
        """Positive, and starts near zero so the circuit begins as a linear readout."""
        return F.softplus(self.raw_apl_gain)

    def _wiring(self, name: str) -> Tensor:
        """The wiring matrix, as a cached sparse tensor when too large to be dense."""
        dense = getattr(self, name)
        if dense.numel() <= DENSE_ELEMENT_LIMIT:
            return dense
        if name not in self._sparse_cache:
            self._sparse_cache[name] = dense.to_sparse()
        return self._sparse_cache[name]

    def _matmul(self, mat: Tensor, x: Tensor) -> Tensor:
        """`x` is (n_tokens, in_features); returns (n_tokens, out_features)."""
        if mat.layout == torch.strided:
            return x @ mat
        return torch.sparse.mm(mat, x.t()).t()

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        n_tok = b * t
        pn = self.pn_proj(x).reshape(n_tok, self.n_pn * self.n_copies)
        kc_in = self._matmul(self._wiring("pn_to_kc"), pn)

        if self.norm is not None:
            # Not biological: normalising the KC input keeps the fixed-fan-in
            # circuit trainable at depth, where raw synapse counts drift in scale.
            kc_in = self.norm(kc_in)

        # APL: one inhibitory neuron pair subtracting the population mean from
        # every Kenyon cell, weighted by the real APL->KC synapse counts.
        mean_activity = kc_in.mean(dim=-1, keepdim=True)
        kc_in = kc_in - self.apl_gain * mean_activity * self.apl_to_kc

        spikes = self.lif(kc_in, spiking=self.spiking)
        sparse_code = k_wta(spikes, self.k)
        mbon = self._matmul(self._wiring("kc_to_mbon"), sparse_code)
        return self.mbon_proj(mbon).reshape(b, t, -1)


class SwiGLU(nn.Module):
    """Standard transformer feed-forward, used as the no-connectome baseline."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden or int(cfg.swiglu_mult * cfg.d_model * 2 / 3)
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, pathway: Pathway | None) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        mode = cfg.ffn_mode
        if mode in ("mb", "mb-shuffled"):
            if pathway is None:
                raise ValueError(f"ffn_mode={mode!r} needs a connectome pathway")
            self.ffn: nn.Module = MushroomBody(cfg, pathway)
        elif mode == "swiglu":
            self.ffn = SwiGLU(cfg)
        else:
            raise ValueError(f"unknown ffn_mode: {mode}")

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        bias: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        # `bias` is the whole-brain read-out for this chunk of tokens. It enters
        # through the sublayer normalisations rather than the residual stream, so
        # every layer is modulated by the connectome while the residual itself
        # stays a pure sum of sublayer outputs -- which is what keeps a chunked
        # pass numerically identical to one full-sequence pass.
        attn_in = self.norm1(x + bias) if bias is not None else self.norm1(x)
        attn_out, present = self.attn(attn_in, cos, sin, past, use_cache)
        x = x + attn_out
        ffn_in = self.norm2(x + bias) if bias is not None else self.norm2(x)
        x = x + self.ffn(ffn_in)
        return x, present


class FlyBrainLM(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        pathway: Pathway | None = None,
        brain: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, pathway) for _ in range(cfg.n_layers)])
        self.brain = brain
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.register_buffer(
            "rope_cos", torch.empty(0), persistent=False
        )
        self.register_buffer("rope_sin", torch.empty(0), persistent=False)
        self._build_rope(cfg.context)
        self.apply(self._init_weights)
        # Scale residual-branch output projections by depth (GPT-2 convention).
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "mbon_proj.weight", "down.weight")):
                nn.init.normal_(param, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _build_rope(self, context: int) -> None:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = build_rope_cache(context, head_dim, self.cfg.rope_base, self.rope_cos.device, torch.float32)
        self.rope_cos = cos
        self.rope_sin = sin

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.cfg.init_std)

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        cache: ModelCache | None = None,
        use_cache: bool = False,
    ):
        """One forward pass.

        Returns `logits` when nothing else is asked for, `(logits, loss)` with
        targets, or `(logits, ModelCache)` when `use_cache` is set. The cache
        carries both the per-block attention state and the whole-brain state, so
        generation continues the brain rather than restarting it.
        """
        if idx.shape[1] > self.cfg.context:
            raise ValueError(f"sequence length {idx.shape[1]} exceeds context {self.cfg.context}")
        if self.rope_cos.numel() == 0 or self.rope_cos.shape[0] < idx.shape[1]:
            self._build_rope(max(idx.shape[1], self.cfg.context))

        if self.brain is not None and self.cfg.brain_in_layers:
            x, brain_state, presents = self._forward_with_brain(idx, cache)
        else:
            x = self.embed(idx)
            presents = []
            for i, block in enumerate(self.blocks):
                layer_past = (
                    cache.layers[i] if cache is not None and i < len(cache.layers) else None
                )
                x, present = block(x, self.rope_cos, self.rope_sin, layer_past, use_cache)
                if present is not None:
                    presents.append(present)
            if self.brain is not None:
                # Incremental decoding feeds one token at a time, so each token is
                # its own chunk; a prefill pass uses the configured chunking.
                chunk_size = 1 if cache is not None else None
                x, brain_state = self.brain.forward_sequence(
                    x, state=cache.brain if cache is not None else None, chunk_size=chunk_size
                )
        new_cache = ModelCache(layers=presents) if use_cache else None
        if new_cache is not None and self.brain is not None:
            new_cache.brain = brain_state

        x = self.norm_f(x)
        logits = self.lm_head(x)
        if use_cache:
            return logits, new_cache
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    def _forward_with_brain(
        self, idx: Tensor, cache: ModelCache | None
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """Run the stack and the connectome interleaved, one chunk at a time.

        Each chunk is attended over the previous chunks through a threaded key/value
        state, which makes this exactly equivalent to a single full-sequence pass --
        attention is causal, so splitting the sequence cannot move information
        backwards. What it buys is the ordering the brain needs: read the current
        neuron state, let *every* block consume that read-out, then fold the chunk's
        own output into the brain. By the time a token is scored, every block has
        been modulated by all 164,587 neurons, and no token can see itself through
        the connectome.
        """
        b, t = idx.shape
        chunk = 1 if cache is not None else max(1, t // max(1, self.cfg.brain_chunks))
        brain = self.brain
        assert brain is not None
        state = cache.brain if cache is not None else None
        dtype = self.embed.weight.dtype
        if state is None:
            state = brain.initial_state(b, idx.device, dtype)
        pasts = [
            cache.layers[i] if cache is not None and i < len(cache.layers) else None
            for i in range(len(self.blocks))
        ]
        brain.reset_stats()
        if cache is None:
            brain.start_sequence()
        chunks: list[Tensor] = []
        for start in range(0, t, chunk):
            window = idx[:, start : start + chunk]
            # (B, 1, d_model): broadcasts over the chunk's positions.
            bias = brain.read(state, dtype=dtype).unsqueeze(1)
            h = self.embed(window)
            for i, block in enumerate(self.blocks):
                h, present = block(h, self.rope_cos, self.rope_sin, pasts[i], True, bias=bias)
                pasts[i] = present
            chunks.append(h)
            state = brain.advance(state, h)
        return torch.cat(chunks, dim=1), state, pasts

    def param_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def layer_breakdown(self) -> dict:
        out: dict[str, int] = {}
        for name, param in self.named_parameters():
            key = name.split(".")[0] if not name.startswith("blocks") else ".".join(name.split(".")[:3])
            out[key] = out.get(key, 0) + param.numel()
        return out
