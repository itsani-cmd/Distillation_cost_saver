"""OPUS-100 en-de data loading, tokenization, and a seq2seq collator.

The student and teacher share the T5 tokenizer, so a single tokenization of the
batch feeds both models. That is what makes word-level KD on soft logits align
without any vocabulary mapping.

Public surface:
    - :func:`load_raw_dataset` -> raw HF splits
    - :func:`build_tokenizer` -> the shared T5 tokenizer
    - :func:`tokenize_example` -> one example -> model inputs (pure, testable)
    - :class:`Seq2SeqCollator` -> pad a list of tokenized examples into a batch
    - :func:`build_dataloaders` -> ready-to-train train/eval dataloaders
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from src.config import Config

if TYPE_CHECKING:  # import heavy deps only for type checkers, not at runtime
    from transformers import PreTrainedTokenizerBase

# Token id used by HF losses to mean "ignore this position".
IGNORE_INDEX = -100


def build_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Load the shared T5 tokenizer used for both student and teacher.

    Args:
        model_name_or_path: HF id or local path of the student model.

    Returns:
        A fast T5 tokenizer.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name_or_path)


def _maybe_subset(split: Any, n: int | None) -> Any:
    """Return the first ``n`` rows of a HF split, or the whole split if ``n`` is None."""
    if n is None:
        return split
    return split.select(range(min(n, len(split))))


def load_raw_dataset(cfg: Config) -> tuple[Any, Any]:
    """Load (and optionally subset) the OPUS-100 train and validation splits.

    Args:
        cfg: The run config (reads the ``data`` section).

    Returns:
        A ``(train_split, eval_split)`` tuple of HF datasets with a
        ``translation`` column of ``{source_lang: str, target_lang: str}``.
    """
    # Imported lazily so unit tests that exercise pure functions do not need
    # the (heavy) datasets dependency or network access.
    from datasets import load_dataset

    d = cfg.data
    ds = load_dataset(d.dataset_name, d.config_name)
    eval_key = "validation" if "validation" in ds else "test"
    train = _maybe_subset(ds["train"], d.train_subset)
    eval_ = _maybe_subset(ds[eval_key], d.eval_subset)
    return train, eval_


def tokenize_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
) -> dict[str, list[int]]:
    """Tokenize a single translation example into model inputs.

    The source is prefixed with the T5 task prefix. Target tokens become
    ``labels``; padding positions are later masked to :data:`IGNORE_INDEX` by
    the collator.

    Args:
        example: A row with an ``translation`` dict keyed by language code.
        tokenizer: The shared T5 tokenizer.
        cfg: The run config (reads ``model.prefix`` and ``data`` lengths).

    Returns:
        A dict with ``input_ids``, ``attention_mask``, and ``labels`` lists.
    """
    d = cfg.data
    pair = example["translation"]
    source_text = cfg.model.prefix + pair[d.source_lang]
    target_text = pair[d.target_lang]

    model_inputs = tokenizer(
        source_text,
        max_length=d.max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=target_text,
        max_length=d.max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


@dataclass
class Seq2SeqCollator:
    """Pad a list of tokenized examples into a batched tensor dict.

    Pads ``input_ids``/``attention_mask`` with the tokenizer pad id and
    ``labels`` with :data:`IGNORE_INDEX` so padded label positions do not
    contribute to the loss.

    Attributes:
        tokenizer: The shared T5 tokenizer (provides the pad id).
        label_pad_token_id: Pad id for labels (defaults to ``-100``).
    """

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Collate ``features`` into a padded batch of tensors."""
        labels = [f["labels"] for f in features]
        input_features = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_label_len = max(len(x) for x in labels)
        padded_labels = []
        for seq in labels:
            pad_len = max_label_len - len(seq)
            padded_labels.append(list(seq) + [self.label_pad_token_id] * pad_len)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_dataloaders(
    cfg: Config,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, DataLoader]:
    """Build train and eval dataloaders from the configured dataset.

    Args:
        cfg: The run config.
        tokenizer: The shared T5 tokenizer.

    Returns:
        A ``(train_loader, eval_loader)`` tuple.
    """
    train_raw, eval_raw = load_raw_dataset(cfg)

    def _map(split: Any) -> Any:
        return split.map(
            lambda ex: tokenize_example(ex, tokenizer, cfg),
            remove_columns=split.column_names,
            desc="tokenizing",
        )

    train_tok = _map(train_raw)
    eval_tok = _map(eval_raw)
    train_tok.set_format(type="python")
    eval_tok.set_format(type="python")

    collator = Seq2SeqCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_tok,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
    )
    eval_loader = DataLoader(
        eval_tok,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, eval_loader
