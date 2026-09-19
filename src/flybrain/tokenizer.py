"""Byte-level BPE tokenizer.

Byte-level BPE is used rather than a word-level scheme so that Chinese text and
any other script round-trip losslessly without an unknown token.
"""

from __future__ import annotations

import os
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def train_tokenizer(
    files: Iterable[str],
    out_path: str,
    *,
    vocab_size: int = 16384,
    min_frequency: int = 2,
    sample_texts: Iterable[str] | None = None,
) -> Tokenizer:
    """Train a byte-level BPE model over the given text files."""
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def iterator():
        for path in files:
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.strip():
                        yield line
        if sample_texts:
            for text in sample_texts:
                yield text

    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tokenizer.save(out_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)
