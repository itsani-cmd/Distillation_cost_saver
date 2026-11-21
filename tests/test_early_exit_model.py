"""Early-exit tests that load real t5-small (exit heads, training step, report).

Gated behind RUN_HEAVY_TESTS=1 like the other model tests. Run them with:

    RUN_HEAVY_TESTS=1 pytest tests/test_early_exit_model.py
"""

import os

import pytest

RUN_HEAVY = os.environ.get("RUN_HEAVY_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_HEAVY, reason="set RUN_HEAVY_TESTS=1 to run model download tests"
)


def _cfg(**ee):
    from src.config import config_from_dict

    raw = {
        "run_name": "ee_it",
        "model": {"student_name_or_path": "t5-small"},
        "lora": {"r": 4, "alpha": 8},
        "data": {
            "train_subset": 8,
            "eval_subset": 4,
            "max_source_length": 16,
            "max_target_length": 16,
            "num_workers": 0,
        },
        "training": {"batch_size": 4, "eval_batch_size": 4, "num_epochs": 1, "log_every": 1},
        "early_exit": {"enabled": True, **ee},
    }
    return config_from_dict(raw)


def _toy_batch(tokenizer):
    from src.data import IGNORE_INDEX

    src = tokenizer(
        ["translate English to German: I have a dog.", "translate English to German: Hello."],
        return_tensors="pt",
        padding=True,
    )
    tgt = tokenizer(
        text_target=["Ich habe einen Hund.", "Hallo."], return_tensors="pt", padding=True
    )
    labels = tgt["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = IGNORE_INDEX
    return {
        "input_ids": src["input_ids"],
        "attention_mask": src["attention_mask"],
        "labels": labels,
    }


def test_last_exit_head_matches_model_logits():
    """Recomputing the last layer through the exit-head code == HF's own logits."""
    import torch

    from src.data import build_tokenizer
    from src.early_exit.heads import all_exit_logits, num_decoder_layers
    from src.model import build_student

    cfg = _cfg()
    model = build_student(cfg).eval()
    batch = _toy_batch(build_tokenizer("t5-small"))
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
        per_layer = all_exit_logits(model, out.decoder_hidden_states)  # recompute last

    assert num_decoder_layers(model) == 6
    assert len(per_layer) == 6
    assert torch.allclose(per_layer[-1], out.logits, atol=1e-5)
    for lg in per_layer:
        assert lg.shape == out.logits.shape


def test_exit_heads_compute_in_float32_under_autocast():
    """Regression: exit logits must not be half precision (fp16 overflow -> inf loss)."""
    import torch

    from src.data import build_tokenizer
    from src.early_exit.heads import all_exit_logits
    from src.model import build_student

    model = build_student(_cfg()).eval()
    batch = _toy_batch(build_tokenizer("t5-small"))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(**batch, output_hidden_states=True)
        per_layer = all_exit_logits(model, out.decoder_hidden_states)
    for lg in per_layer:
        assert lg.dtype == torch.float32
        assert torch.isfinite(lg).all()


def test_exit_head_matches_truncated_decoder():
    """Layer-i exit == running a decoder that only has its first i layers.

    This is the strongest check that intermediate exits are wired correctly
    (right hidden-state index, norm applied once, same scaling + head).
    """
    import copy

    import torch

    from src.data import build_tokenizer
    from src.early_exit.heads import all_exit_logits, unwrap_t5
    from src.model import build_student

    model = build_student(_cfg()).eval()
    batch = _toy_batch(build_tokenizer("t5-small"))
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)
        per_layer = all_exit_logits(model, out.decoder_hidden_states, final_logits=out.logits)

        t5 = copy.deepcopy(unwrap_t5(model))
        for i in (1, 3):
            t5.decoder.block = torch.nn.ModuleList(list(unwrap_t5(model).decoder.block)[:i])
            truncated = t5(**batch).logits
            assert torch.allclose(per_layer[i - 1], truncated, atol=1e-4), f"layer {i} mismatch"


def test_train_step_grades_every_layer_and_backprops(tmp_path):
    import torch

    import utils.colab_utils as cu
    from src.train import Trainer

    os.environ["DISTILL_TRANSLATE_HOME"] = str(tmp_path)
    paths = cu.get_project_paths("ee_it", create=True)
    trainer = Trainer(_cfg(self_distill_weight=0.5), paths=paths)
    batch = next(iter(trainer.train_loader))
    out = trainer._train_step(batch)

    assert len(out.per_layer_ce) == 6
    assert torch.isfinite(out.total)
    assert out.self_distill is not None and out.self_distill.item() >= 0.0
    # Shallow layers should start out worse than the last one (pretrained t5-small).
    assert out.per_layer_ce[0].item() > out.per_layer_ce[-1].item()

    trainer.accelerator.backward(out.total)
    grads = [p.grad for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


def test_train_final_norm_trainable_and_saved(tmp_path):
    import torch

    from src.early_exit.heads import unwrap_t5
    from src.model import build_student, load_adapter, save_adapter

    cfg = _cfg(train_final_norm=True)
    model = build_student(cfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("decoder.final_layer_norm" in n for n in trainable)
    assert not any("encoder.final_layer_norm" in n for n in trainable)

    # Change the norm, round-trip the adapter, and check the change survived.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and "decoder.final_layer_norm" in n:
                p.fill_(0.5)
    save_adapter(model, tmp_path / "adapter")
    reloaded = load_adapter(cfg, tmp_path / "adapter").eval()
    x = torch.ones(1, 1, unwrap_t5(reloaded).config.d_model)
    y = unwrap_t5(reloaded).decoder.final_layer_norm(x)
    assert torch.allclose(y, torch.full_like(y, 0.5), atol=1e-4)


def test_layer_report_sane_numbers():
    import torch
    from torch.utils.data import DataLoader

    from src.data import build_tokenizer
    from src.early_exit.analysis import format_report, layer_report
    from src.model import build_student

    tok = build_tokenizer("t5-small")
    batch = _toy_batch(tok)
    model = build_student(_cfg())
    report = layer_report(model, DataLoader([batch], batch_size=None), torch.device("cpu"))

    assert report["num_layers"] == 6
    last = report["layers"][-1]
    assert last["agreement_with_final"] == pytest.approx(1.0)
    for r in report["layers"]:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["agreement_with_final"] <= 1.0
    fractions = report["oracle"]["exit_layer_fraction"].values()
    assert sum(fractions) == pytest.approx(1.0)
    assert 1.0 <= report["oracle"]["avg_exit_layer"] <= 6.0
    assert "oracle" in format_report(report)
