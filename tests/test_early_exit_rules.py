"""Fast tests for the Part 2 stopping rules, threshold and sweep plumbing (no model download)."""

import pytest
import torch

from src.early_exit.confidence import pick_exit_layer
from src.early_exit.thresholds import constant_threshold

VOCAB = 5


def _onehotish(word, sure):
    """Logits whose top word is ``word``; softmax gap ~1 if ``sure`` else ~0.

    Unsure = two words tied at the top (gap exactly 0).
    """
    logits = torch.zeros(VOCAB)
    if sure:
        logits[word] = 20.0
    else:
        logits[word] = 5.0
        logits[(word + 1) % VOCAB] = 5.0
    return logits


def _layers(spec):
    """spec: list of (top_word, sure) per layer, layer 1 first. Records which layers were read."""
    seen = []

    def get_logits(layer):
        seen.append(layer)
        word, sure = spec[layer - 1]
        return _onehotish(word, sure)

    return get_logits, seen


def test_static_stops_at_k_and_reads_only_that_layer():
    get, seen = _layers([(0, False)] * 6)
    layer, _ = pick_exit_layer(get, 6, "static", static_layer=3)
    assert layer == 3
    assert seen == [3]


def test_static_layer_is_clipped():
    get, _ = _layers([(0, False)] * 6)
    assert pick_exit_layer(get, 6, "static", static_layer=99)[0] == 6
    assert pick_exit_layer(get, 6, "static", static_layer=0)[0] == 1


def test_softmax_stops_at_first_confident_layer_lazily():
    get, seen = _layers([(0, False), (1, False), (2, True), (3, True), (4, True), (4, True)])
    layer, logits = pick_exit_layer(get, 6, "softmax", threshold=0.5)
    assert layer == 3
    assert int(logits.argmax()) == 2
    assert seen == [1, 2, 3]  # never computed layers 4-6


def test_softmax_falls_back_to_last_layer():
    get, _ = _layers([(0, False)] * 6)
    assert pick_exit_layer(get, 6, "softmax", threshold=0.5)[0] == 6


def test_softmax_threshold_one_never_exits_early():
    get, _ = _layers([(0, True)] * 6)
    assert pick_exit_layer(get, 6, "softmax", threshold=1.0)[0] == 6


def test_softmax_threshold_zero_always_exits_at_layer_one():
    get, _ = _layers([(0, False)] * 6)
    assert pick_exit_layer(get, 6, "softmax", threshold=0.0)[0] == 1


def test_softmax_min_layer_skips_shallow_layers():
    # Layer 1 is confidently wrong; with min_layer=2 it must not even be scored.
    get, seen = _layers([(0, True), (1, False), (2, True), (2, True), (2, True), (2, True)])
    layer, logits = pick_exit_layer(get, 6, "softmax", threshold=0.5, min_layer=2)
    assert layer == 3
    assert int(logits.argmax()) == 2
    assert seen == [2, 3]
    # Without it, the confidently wrong layer 1 wins.
    get, _ = _layers([(0, True), (1, False), (2, True), (2, True), (2, True), (2, True)])
    assert pick_exit_layer(get, 6, "softmax", threshold=0.5)[0] == 1


def test_softmax_min_layer_is_clipped():
    get, _ = _layers([(0, True)] * 6)
    assert pick_exit_layer(get, 6, "softmax", threshold=0.0, min_layer=99)[0] == 6
    assert pick_exit_layer(get, 6, "softmax", threshold=0.0, min_layer=0)[0] == 1


def test_oracle_stops_at_first_layer_matching_last():
    get, _ = _layers([(0, True), (3, False), (2, True), (2, True), (1, True), (2, True)])
    layer, logits = pick_exit_layer(get, 6, "oracle")
    assert layer == 3
    assert int(logits.argmax()) == 2


def test_oracle_uses_last_layer_when_nothing_matches():
    get, _ = _layers([(0, True)] * 5 + [(4, True)])
    assert pick_exit_layer(get, 6, "oracle")[0] == 6


def test_unknown_rule_rejected():
    get, _ = _layers([(0, True)] * 6)
    with pytest.raises(ValueError):
        pick_exit_layer(get, 6, "saturation")
    with pytest.raises(ValueError):
        pick_exit_layer(get, 0, "static")


def test_constant_threshold():
    assert constant_threshold(0.7) == 0.7
    assert constant_threshold(0.7, position=50) == 0.7
    with pytest.raises(ValueError):
        constant_threshold(1.5)


def test_build_run_specs_sweep():
    from src.early_exit.run_eval import build_run_specs

    specs = build_run_specs(6, thresholds=(0.9, 0.5))
    names = [s.name for s in specs]
    assert names[0] == "full" and specs[0].static_layer == 6
    assert names == [
        "full",
        "static_L1",
        "static_L2",
        "static_L3",
        "static_L4",
        "static_L5",
        "softmax_0.9",
        "softmax_0.5",
        "oracle",
    ]
    assert [s.name for s in build_run_specs(6, (), static_layers=(2,), oracle=False)] == [
        "full",
        "static_L2",
    ]
    mins = build_run_specs(6, thresholds=(0.9,), static_layers=(), oracle=False, min_layers=(1, 3))
    assert [(s.name, s.min_layer) for s in mins] == [
        ("full", 1),
        ("softmax_0.9", 1),
        ("softmax_0.9_min3", 3),
    ]


def test_summarize_and_table():
    pytest.importorskip("sacrebleu")
    from src.early_exit.run_eval import RunSpec, format_table, summarize

    refs = ["das ist ein test", "hallo welt"]
    row = summarize(
        RunSpec("softmax_0.5", "softmax", threshold=0.5), refs, [1, 2, 3, 6], 0.1, refs, refs, 6
    )
    assert row["bleu"] == pytest.approx(100.0)
    assert row["bleu_vs_full"] == pytest.approx(100.0)
    assert row["avg_layers"] == pytest.approx(3.0)
    assert row["layer_reduction"] == pytest.approx(2.0)
    assert row["exit_L1"] == pytest.approx(0.25) and row["exit_L4"] == 0.0
    assert row["static_layer"] is None and row["threshold"] == 0.5
    assert row["min_exit_layer"] == 1
    assert "softmax_0.5" in format_table([row])
