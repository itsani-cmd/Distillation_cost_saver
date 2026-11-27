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


def test_cross_attention_mask_shape():
    import numpy as np
    import needle as ndl
    from needle_impl.transformer import attention_mask

    batch_size = 2
    source_length = 12
    target_length = 5

    source_valid = np.ones(
        (batch_size, source_length),
        dtype=bool
    )

    source_valid[0, -2:] = False

    cross_mask = attention_mask(
        source_valid,
        query_length=target_length,
        causal=False,
        device=ndl.cpu(),
        dtype="float32"
    )

    assert cross_mask.shape == (
        batch_size, 1, target_length, source_length
    )

    mask_np = cross_mask.numpy()

    # The final two source positions are padding for batch item 0.
    assert np.all(mask_np[0, 0, :, -2:] < -1e8)

    # No positions are masked for batch item 1.
    assert np.all(mask_np[1] == 0)

    print("PASS: Cross-attention mask shape and padding")
