"""Sampling and chat formatting for the trained model.

Decoding is incremental: each step feeds one token and reuses the KV cache, so
generation cost per token stays roughly flat as the context grows. Streaming
decodes the whole generated prefix each step and emits the delta, which keeps
multi-byte Chinese text from being split mid-character.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from flybrain.connectome_lm import build_lm
from flybrain.model import FlyBrainLM, ModelConfig

# Roles the chat template can emit; `<system>` is handled separately by
# `format_chat` because it has no dedicated token.
CHAT_ROLES = ("user", "assistant")


@dataclass
class SamplingConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.92
    repetition_penalty: float = 1.08
    seed: int | None = None

    @classmethod
    def from_json(cls, path: str) -> "SamplingConfig":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def special_id(tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer has no {token} token")
    return int(token_id)


def format_chat(
    messages: Sequence[dict],
    tokenizer,
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Render a conversation into token ids using this project's chat tokens.

    Special tokens are inserted by id rather than by encoding their text, so the
    template does not depend on the tokenizer's special-token matching rules.

    There is no `<system>` token in the vocabulary: the tokenizer was trained with
    `<pad> <bos> <eos> <user> <assistant>` only, and inventing an id that never
    appeared in training would produce an embedding that means nothing. A system
    prompt is therefore folded into the first user turn.
    """
    system_text = " ".join(
        str(m.get("content", "")) for m in messages if m.get("role") == "system"
    ).strip()
    turns = [m for m in messages if m.get("role") != "system"]

    ids = [special_id(tokenizer, "<bos>")]
    if system_text:
        # A system preamble becomes an opening user turn carrying the instruction.
        ids.append(special_id(tokenizer, "<user>"))
        ids.extend(tokenizer.encode(system_text).ids)
        ids.append(special_id(tokenizer, "<eos>"))
    for message in turns:
        role = message.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        ids.append(special_id(tokenizer, f"<{role}>"))
        ids.extend(tokenizer.encode(str(message.get("content", ""))).ids)
        ids.append(special_id(tokenizer, "<eos>"))
    if add_generation_prompt:
        ids.append(special_id(tokenizer, "<assistant>"))
    return ids


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """Divide (or multiply) the logits of tokens already present."""
    if penalty == 1.0 or generated.numel() == 0:
        return logits
    unique = torch.unique(generated)
    selected = logits[..., unique]
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    return logits.index_copy(-1, unique, selected)


def _filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Zero the probability mass outside the nucleus."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_index, sorted_logits)
    return logits


@torch.no_grad()
def generate_stream(
    model: FlyBrainLM,
    tokenizer,
    prompt_ids: Sequence[int],
    *,
    cfg: SamplingConfig | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    stop_ids: Sequence[int] = (),
) -> Iterator[str]:
    """Yield generated text incrementally, stopping at EOS or the token budget."""
    cfg = cfg or SamplingConfig()
    device = torch.device(device)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    model.eval()

    context = model.cfg.context
    ids = list(prompt_ids)[-context:]
    idx = torch.tensor([ids], device=device)
    generated: list[int] = []
    cache = None
    emitted = ""

    for _ in range(cfg.max_new_tokens):
        window = idx if cache is None else idx[:, -1:]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            logits, cache = model(window, cache=cache, use_cache=True)
        logits = logits[:, -1, :].float()

        if cfg.repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, idx[0], cfg.repetition_penalty)
        logits = logits / max(cfg.temperature, 1e-5)
        logits = _filter_top_k_top_p(logits, cfg.top_k, cfg.top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())

        if next_id in stop_ids:
            break
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=-1)
        generated.append(next_id)

        # Keep the window inside the context; the brain state keeps evolving.
        if idx.shape[1] > context:
            idx = idx[:, -context:]
            cache = cache.truncated(context)

        text = tokenizer.decode(generated, skip_special_tokens=True)
        if len(text) > len(emitted):
            yield text[len(emitted) :]
            emitted = text

    if len(emitted) == 0 and generated:
        yield tokenizer.decode(generated, skip_special_tokens=True)


def generate(
    model: FlyBrainLM,
    tokenizer,
    prompt_ids: Sequence[int],
    *,
    cfg: SamplingConfig | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    stop_ids: Sequence[int] = (),
) -> str:
    """Blocking version of `generate_stream`."""
    return "".join(
        generate_stream(
            model, tokenizer, prompt_ids, cfg=cfg, device=device, dtype=dtype, stop_ids=stop_ids
        )
    )


def chat_reply(
    model: FlyBrainLM,
    tokenizer,
    history: Sequence[dict],
    *,
    cfg: SamplingConfig | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[str]:
    """Stream the assistant's next turn given a conversation so far."""
    prompt_ids = format_chat(history, tokenizer, add_generation_prompt=True)
    stop = [special_id(tokenizer, "<eos>"), special_id(tokenizer, "<user>"), special_id(tokenizer, "<bos>")]
    yield from generate_stream(
        model, tokenizer, prompt_ids, cfg=cfg, device=device, dtype=dtype, stop_ids=stop
    )


def load_model(checkpoint: str, *, device: str = "cuda") -> tuple[FlyBrainLM, ModelConfig]:
    """Rebuild the model from a checkpoint, using the config saved inside it.

    Both the connectome wiring and the whole-brain circuit are stored as buffers
    in the checkpoint, so a saved model reloads without needing the source data
    files again.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**state["model_config"])
    weights = state["model"]

    pathway = None
    if getattr(model_cfg, "arch", "transformer") != "connectome" and model_cfg.ffn_mode in ("mb", "mb-shuffled"):
        from flybrain.connectome import build_synthetic_pathway

        wiring = weights.get("blocks.0.ffn.pn_to_kc")
        if wiring is None:
            raise KeyError("checkpoint has no connectome wiring buffer")
        n_kc = int(wiring.shape[-1])
        n_pn = int(wiring.shape[0])
        mbon = weights.get("blocks.0.ffn.kc_to_mbon")
        n_mbon = int(mbon.shape[-1]) if mbon is not None else 34
        pathway = build_synthetic_pathway(n_pn, n_kc, n_mbon, seed=0)

    brain = None
    if any(key.startswith("brain.circuit.") for key in weights):
        from flybrain.wholebrain import BrainPathway, circuit_from_state_dict

        circuit = circuit_from_state_dict(weights, edge_chunk=model_cfg.brain_edge_chunk, impl=model_cfg.brain_impl)
        # Take the bottleneck width from the saved weights rather than from the
        # config: the two can disagree, and the shapes in the checkpoint are the
        # only authoritative record of how the model was actually built.
        down = weights.get("brain.down.weight")
        read_down = weights.get("brain.read_down.weight")
        if read_down is not None:
            rank = int(read_down.shape[0])
        elif down is not None:
            rank = int(down.shape[0])
        else:
            rank = model_cfg.brain_rank
        brain = BrainPathway(
            circuit,
            model_cfg.d_model,
            rank=rank,
            chunks=model_cfg.brain_chunks,
            iters=model_cfg.brain_iters,
            spiking=model_cfg.brain_spiking,
            analog_readout=model_cfg.brain_analog_readout,
            analog_mode=getattr(model_cfg, "brain_analog_mode", "rate"),
            input_mask_path=getattr(model_cfg, "brain_input_mask", "") or None,
            output_mask_path=getattr(model_cfg, "brain_output_mask", "") or None,
        )

    model = build_lm(model_cfg, pathway, brain)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model, model_cfg
