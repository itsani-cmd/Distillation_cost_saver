"""Tests for BLEU computation and predictions export (no model downloads)."""

import json

import pytest

from src.evaluate import compute_bleu, save_predictions


def test_compute_bleu_perfect_match_high():
    preds = ["das ist ein test", "hallo welt"]
    refs = ["das ist ein test", "hallo welt"]
    assert compute_bleu(preds, refs) > 99.0


def test_compute_bleu_mismatch_lower():
    preds = ["completely different text here", "another wrong one"]
    refs = ["das ist ein test", "hallo welt"]
    score = compute_bleu(preds, refs)
    assert score < 50.0


def test_compute_bleu_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_bleu(["a"], ["a", "b"])


def test_save_predictions_schema(tmp_path):
    out = tmp_path / "preds.jsonl"
    path = save_predictions(
        out,
        sources=["hi", "bye"],
        references=["hallo", "tschüss"],
        hypotheses=["hallo", "tschuss"],
        bleu=42.0,
    )
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = json.loads(lines[0])
    assert header["type"] == "summary"
    assert header["bleu"] == 42.0
    assert header["n"] == 2

    first = json.loads(lines[1])
    assert set(first) == {"source", "reference", "hypothesis"}
    assert first["source"] == "hi"
