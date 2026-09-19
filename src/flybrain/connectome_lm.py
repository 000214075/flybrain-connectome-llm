"""A language model whose entire sequence model is the fly connectome.

The hybrid model in `model.py` is a transformer whose feed-forward block has been
replaced by a mushroom body, with the whole brain interleaved as a modulation. That
is still a transformer: multi-head attention with rotary positions, RMSNorm, a
residual stream, a key/value cache and a transformer block layout are all doing the
sequence modelling, and the fly circuits are decoration on top of them.

This module has none of that. A token is projected into a drive on the population,
the population is swept once per token through all 25,563,096 real synapses, and the
next token is read out of the population state:

    token id -> sensory projection -> drive (B, 164587)
    for each position t:
        state    = LIF(W_connectome @ state + drive_t)   # fold token t in
        logits_t = decoder(read(state))                  # then predict t+1

There is no attention, no positional encoding, no key/value cache, no residual
stream and no transformer block. Sequence structure -- word order, the fact that
the previous token matters -- can only come from the connectome's own recurrence,
which is how a brain does it. The mushroom body is not a separate module here
either: its Kenyon cells and MBONs are cells inside the same connectome.

What remains of ordinary language-model machinery is the interface, and only the
interface: a token embedding and a linear read-out to the vocabulary. Text has to
get in and out of any model somehow, biological or not; neither one mixes
information across positions, so neither one is the sequence model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flybrain.model import ModelCache
from flybrain.wholebrain import BrainPathway


class ConnectomeLM(nn.Module):
    """Next-token prediction with the connectome as the only sequence mechanism."""

    def __init__(self, cfg, brain: BrainPathway) -> None:
        super().__init__()
        self.cfg = cfg
        self.brain = brain
        self.n_neurons = brain.circuit.n_neurons
        self.rank = brain.read_down.weight.shape[0]

        # The interface: text in, text out. Neither of these mixes positions. The
        # embedding is `d_model` wide because the drive into the population is the
        # brain's own `drive_from_tokens` path (`norm_in -> down -> up -> norm_drive`),
        # the same one the hybrid model uses. An earlier version of this model added
        # its own `sensory` projection instead and left the brain's `up` unused,
        # which was 42M parameters of dead weight in a 135M-parameter model.
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.head = nn.Linear(self.rank, cfg.vocab_size, bias=False)

        # The read-out is the one place where this model can be badly conditioned at
        # initialisation, because it sums 164,587 inputs into 256. Measured on the
        # assembled model with the shared 0.02 init, the read-out came out with a
        # standard deviation of ~5 while the logits it fed had a standard deviation
        # of ~3.5, so the untrained model was confidently wrong: loss 13.3 to 16.4
        # against 9.70 for a uniform guess, purely from logit variance. Scaling the
        # init by sqrt(n_neurons) makes the read-out preserve its input's scale.
        nn.init.normal_(self.brain.read_down.weight, std=1.0 / (self.n_neurons**0.5))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _drives(self, idx: Tensor) -> Tensor:
        """(B, T) token ids -> (B, T, n_neurons) drives, projected in one shot.

        The whole drive for a sequence in one matmul; only the sweeps have to be
        serial. The population-level normalisation the connectome needs -- measured
        earlier in this project, a drive 25x too small took the firing rate from
        10.7% to exactly zero -- is the brain's own `norm_drive`, applied inside
        `drive_from_tokens`.
        """
        return self.brain.drive_from_tokens(self.embed(idx))

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        cache: object = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        b, t = idx.shape
        if cache is None and t > self.cfg.context:
            raise ValueError(f"sequence length {t} exceeds context {self.cfg.context}")

        # Clear the utilisation counters *before* projecting the drives, since that is
        # where `_last_drive` is recorded and `reset_stats` would otherwise wipe it.
        self.brain.reset_stats()
        drives = self._drives(idx)
        if cache is None:
            self.brain.start_sequence()
            state = self.brain.initial_state(b, idx.device, drives.dtype)
        else:
            # Incremental decoding. The state handed over is the population vector
            # itself -- there is no key/value cache to carry, because nothing here
            # remembers anything except the activity of 164,587 neurons. The
            # pre-activation trace is deliberately *not* cleared: it belongs to the
            # sweep that produced the incoming state, and reading it is how the first
            # token of this call sees the previous one.
            state = cache.brain
            if state is None:
                state = self.brain.initial_state(b, idx.device, drives.dtype)

        logits: list[Tensor] = []
        for position in range(t):
            # Fold the token in first, then read. The read therefore sees positions
            # `0..position` and predicts the token after them, which is exactly the
            # next-token task the dataset defines. Reading *before* folding -- the
            # first version of this loop -- left the model unable to see the token
            # immediately before the one it was predicting, and disagreed with
            # `generate`, which folds the prompt and only then reads. Causality is
            # still ordering and nothing else: the target has not been folded when
            # it is predicted, so no token can see itself.
            state = self.brain.advance_drive(state, drives[:, position])
            logits.append(self.head(self.brain.read_latent(state)))

        logits = torch.stack(logits, dim=1)
        if use_cache:
            return logits, ModelCache(brain=state)
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-1
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, temperature: float = 1.0) -> Tensor:
        """Sample continuations, carrying the population state across tokens.

        The whole model state between steps is the population vector, which is what
        a recurrent network would call its hidden state and what a brain would call
        its activity. Nothing is cached or recomputed.
        """
        b = idx.shape[0]
        self.brain.reset_stats()
        self.brain.start_sequence()
        state = self.brain.initial_state(b, idx.device, self.embed.weight.dtype)
        for position in range(idx.shape[1]):
            drive = self._drives(idx[:, position : position + 1])[:, 0]
            state = self.brain.advance_drive(state, drive)
        out = idx
        for _ in range(max_new_tokens):
            logits = self.head(self.brain.read_latent(state))[:, -1] / max(temperature, 1e-6)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            out = torch.cat([out, nxt], dim=1)
            drive = self._drives(nxt)[:, 0]
            state = self.brain.advance_drive(state, drive)
        return out


def build_lm(cfg, pathway, brain: BrainPathway | None):
    """The model the config asks for: pure connectome, or the transformer hybrid."""
    from flybrain.model import FlyBrainLM

    if getattr(cfg, "arch", "transformer") == "connectome":
        if brain is None:
            raise ValueError("arch='connectome' needs a brain; set model.brain_path")
        return ConnectomeLM(cfg, brain)
    return FlyBrainLM(cfg, pathway, brain)
