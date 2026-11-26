"""Lightweight tests; run with `pytest -q tests/needle` after setting PYTHONPATH."""
import numpy as np
from needle_impl.attention import relative_position_buckets
from needle_impl.train import shifted_targets


def test_relative_position_buckets():
    buckets = relative_position_buckets(6, 6, bidirectional=False)
    assert buckets.shape == (6, 6)
    assert buckets.min() >= 0
    assert buckets.max() < 32
    assert np.all(np.diag(buckets) == 0)


def test_shifted_labels():
    labels = np.array([[10, 20, 1, -100], [30, 1, -100, -100]])
    safe, decoder, valid = shifted_targets(labels, pad_id=0, decoder_start_id=0)
    assert decoder.tolist() == [[0, 10, 20, 1], [0, 30, 1, 0]]
    assert valid.tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    assert safe[0, 3] == 0
