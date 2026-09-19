"""Corpus preparation and the packed token dataset.

Text is tokenised once into flat `uint16`/`uint32` binary shards; training then
reads random windows straight out of the memory-mapped file, so the corpus can be
much larger than RAM.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, Iterator

# The domain is the fly brain and the connectome work around it. Fetched pages are
# full of side content -- site navigation, unrelated news, other people's
# biomarkers -- and a teacher told to ground itself in a passage will happily write
# a confident question about whatever it was handed. Measured: without this filter
# the grounded corpus contained question/answer pairs about human biomarkers and
# cable theory for a model that is supposed to know about Kenyon cells.
FLY_BRAIN_TERMS = re.compile(
    r"果蝇|苍蝇|Drosophila|drosophila|FlyWire|flywire|MaleCNS|hemibrain|蘑菇体|"
    r"mushroom body|Kenyon|连接组|connectome|突触|synapse|嗅觉|olfactory|"
    r"投射神经元|projection neuron|天线叶|antennal lobe|脑|brain|神经元|neuron",
    re.IGNORECASE,
)

# Terms that only make sense in this domain; one of these is enough on its own.
FLY_BRAIN_STRONG_TERMS = re.compile(
    r"果蝇|苍蝇|Drosophila|FlyWire|MaleCNS|hemibrain|蘑菇体|mushroom body|Kenyon|"
    r"连接组|connectome|嗅觉|olfactory|突触|synapse",
    re.IGNORECASE,
)


# The neuron-model literature this project implements (LIF dynamics, surrogate
# gradients, spike trains) is in-domain even when it never mentions a fly, so it
# gets its own rule rather than being lost to the two-term threshold.
NEURON_MODEL_TERMS = re.compile(
    r"spik|脉冲|放电|synap|突触|threshold|阈值|membrane|膜电位|integrate-and-fire|"
    r"surrogate gradient|代理梯度|LIF",
    re.IGNORECASE,
)


def is_domain_text(text: str, *, min_terms: int = 2) -> bool:
    """Whether a passage is about the fly brain rather than a page's side content.

    One strong term (果蝇, connectome, mushroom body) is enough; otherwise two
    weaker ones (brain, neuron), so a paragraph merely mentioning "brain" in
    passing does not qualify.
    """
    if FLY_BRAIN_STRONG_TERMS.search(text):
        return True
    if re.search(r"neuron|neuronal|neural|神经元|神经细胞", text, re.IGNORECASE) and NEURON_MODEL_TERMS.search(
        text
    ):
        return True
    return len(set(m.group(0).lower() for m in FLY_BRAIN_TERMS.finditer(text))) >= min_terms

import numpy as np
import torch


def iter_jsonl_text(path: str, fields: tuple[str, ...] = ("text",)) -> Iterator[str]:
    """Yield the text of each JSONL record, tolerating chat-style records."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in fields:
                value = obj.get(field)
                if isinstance(value, str) and value.strip():
                    yield value
                    break
            else:
                messages = obj.get("messages")
                if isinstance(messages, list):
                    parts = [
                        f"<{m.get('role', 'user')}>{m.get('content', '')}<eos>"
                        for m in messages
                        if isinstance(m, dict)
                    ]
                    if parts:
                        yield "".join(parts)


def corpus_files(root: str, patterns: tuple[str, ...] = (".txt", ".jsonl", ".parquet")) -> list[str]:
    """Every corpus file under `root`, sorted for reproducibility."""
    found: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.endswith(patterns):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def _iter_text(path: str, max_chars: int) -> Iterator[str]:
    """Stream a corpus file as text chunks, whatever its on-disk format."""
    if path.endswith(".jsonl"):
        yield from iter_jsonl_text(path)
        return
    if path.endswith(".parquet"):
        yield from _iter_parquet_text(path)
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            chunk = fh.read(max_chars)
            if not chunk:
                return
            yield chunk


def _iter_parquet_text(path: str) -> Iterator[str]:
    """Yield one text/document per row of a parquet corpus, row group by row group."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    text_column = None
    for field in parquet_file.schema_arrow:
        if field.name.lower() in ("text", "content", "document", "body"):
            text_column = field.name
            break
    if text_column is None:
        raise ValueError(f"no text column in {path}; columns are {parquet_file.schema_arrow.names}")
    for batch in parquet_file.iter_batches(batch_size=512, columns=[text_column]):
        for value in batch.column(0).to_pylist():
            if value:
                yield str(value)


def tokenize_corpus(
    files: Iterable[str],
    tokenizer,
    out_dir: str,
    *,
    val_fraction: float = 0.01,
    token_dtype: str = "uint16",
    max_chars_per_read: int = 1 << 20,
    log_every: int = 200,
) -> dict:
    """Tokenise files into train/val shards and return a summary.

    Each file is appended to the train shard; a validation shard is carved out by
    holding back every `1/val_fraction`-th chunk, which keeps the two splits from
    sharing whole documents.
    """
    os.makedirs(out_dir, exist_ok=True)
    dtype = np.dtype(token_dtype)
    # Reserve the first ids for specials added by the trainer.
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer has no <eos> token")

    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    counts = {"train": 0, "val": 0}
    files = list(files)
    stride = max(2, int(round(1.0 / max(val_fraction, 1e-9)))) if val_fraction > 0 else 0

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        chunk_index = 0
        for path in files:
            for text in _iter_text(path, max_chars_per_read):
                ids = tokenizer.encode(text).ids
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=dtype)
                target = f_val if stride and chunk_index % stride == stride - 1 else f_train
                target.write(arr.tobytes())
                counts["val" if target is f_val else "train"] += arr.size
                chunk_index += 1
                if chunk_index % log_every == 0:
                    print(
                        f"  tokenized {chunk_index} chunks "
                        f"train={counts['train'] / 1e6:.1f}M val={counts['val'] / 1e6:.2f}M",
                        flush=True,
                    )

    summary = {
        "files": files,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "token_dtype": token_dtype,
        "val_fraction": val_fraction,
        "vocab_size": tokenizer.get_vocab_size(),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


@dataclass
class PackedBin:
    path: str
    dtype: np.dtype
    n_tokens: int

    @classmethod
    def open(cls, path: str, token_dtype: str = "uint16") -> "PackedBin":
        size = os.path.getsize(path)
        dtype = np.dtype(token_dtype)
        return cls(path, dtype, size // dtype.itemsize)

    def read_slice(self, start: int, stop: int) -> np.ndarray:
        with open(self.path, "rb") as fh:
            fh.seek(start * self.dtype.itemsize)
            raw = fh.read((stop - start) * self.dtype.itemsize)
        return np.frombuffer(raw, dtype=self.dtype, count=stop - start)


class TokenWindowDataset(torch.utils.data.Dataset):
    """Random fixed-length windows from a packed token shard.

    Consecutive windows are reconstructed deterministically from `(epoch, index)`
    so that resuming a run reproduces the same data order.
    """

    def __init__(self, path: str, context: int, *, token_dtype: str = "uint16", seed: int = 0, epoch: int = 0):
        self.bin = PackedBin.open(path, token_dtype)
        self.context = context
        if self.bin.n_tokens < context + 1:
            raise ValueError(f"{path} holds {self.bin.n_tokens} tokens, need at least {context + 1}")
        self.n_windows = self.bin.n_tokens // (context + 1)
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_windows

    def _start(self, index: int) -> int:
        rng = np.random.default_rng((self.seed, self.epoch, index))
        lo = 0
        hi = max(0, self.bin.n_tokens - self.context - 1)
        return int(rng.integers(lo, hi + 1)) if hi > 0 else 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._start(index)
        window = self.bin.read_slice(start, start + self.context + 1).astype(np.int64)
        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:])
        return x, y
