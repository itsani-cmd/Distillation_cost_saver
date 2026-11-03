"""Tests for tokenization and the seq2seq collator.

Uses a tiny fake tokenizer so these run fast on CPU with no downloads. The
transformers-backed path is exercised in the Colab notebook / integration runs.
"""

import torch

from src.config import config_from_dict
from src.data import IGNORE_INDEX, Seq2SeqCollator, tokenize_example


class FakeTokenizer:
    """Minimal char-level stand-in for a HF tokenizer (enough for the collator)."""

    pad_token_id = 0

    def __call__(self, text=None, text_target=None, max_length=None, truncation=False):
        value = text if text is not None else text_target
        ids = [ord(c) % 97 + 1 for c in value]  # avoid the pad id (0)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, features, padding=True, return_tensors="pt"):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn = [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
        }


def _cfg():
    return config_from_dict({"data": {"max_source_length": 32, "max_target_length": 32}})


def test_tokenize_example_shapes_and_prefix():
    tok = FakeTokenizer()
    cfg = _cfg()
    ex = {"translation": {"en": "hi", "de": "hallo"}}

    out = tokenize_example(ex, tok, cfg)

    assert "input_ids" in out and "attention_mask" in out and "labels" in out
    # Source includes the prefix, so it is longer than the raw 2-char source.
    assert len(out["input_ids"]) == len(cfg.model.prefix) + len("hi")
    assert len(out["labels"]) == len("hallo")


def test_tokenize_example_truncation():
    tok = FakeTokenizer()
    cfg = config_from_dict({"data": {"max_source_length": 4, "max_target_length": 3}})
    ex = {"translation": {"en": "a long source", "de": "lange"}}

    out = tokenize_example(ex, tok, cfg)

    assert len(out["input_ids"]) == 4
    assert len(out["labels"]) == 3


def test_collator_pads_inputs_and_masks_labels():
    tok = FakeTokenizer()
    collator = Seq2SeqCollator(tokenizer=tok)
    features = [
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [9, 10]},
        {"input_ids": [5], "attention_mask": [1], "labels": [9, 10, 11, 12]},
    ]

    batch = collator(features)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 4)
    # Shorter input padded with pad id, attention zeroed.
    assert batch["input_ids"][1].tolist() == [5, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1, 0, 0]
    # Shorter label sequence padded with IGNORE_INDEX.
    assert batch["labels"][0].tolist() == [9, 10, IGNORE_INDEX, IGNORE_INDEX]
