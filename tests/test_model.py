"""Model tests: LoRA trainability, frozen teacher, adapter round-trip.

These load real (small) models and are gated behind RUN_HEAVY_TESTS=1 so the
default CI/local run stays fast and offline. Run them with:

    RUN_HEAVY_TESTS=1 pytest tests/test_model.py
"""

import os

import pytest

RUN_HEAVY = os.environ.get("RUN_HEAVY_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_HEAVY, reason="set RUN_HEAVY_TESTS=1 to run model download tests"
)


def _cfg(teacher=None):
    from src.config import config_from_dict

    raw = {"model": {"student_name_or_path": "t5-small"}, "lora": {"r": 4, "alpha": 8}}
    if teacher is not None:
        raw["model"]["teacher_name_or_path"] = teacher
        raw["distillation"] = {"kd_weight": 0.5}
    return config_from_dict(raw)


def test_only_lora_params_trainable():
    from src.model import build_student, count_parameters

    model = build_student(_cfg())
    trainable, total = count_parameters(model)
    assert 0 < trainable < total  # LoRA adds a small trainable fraction

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert "lora" in name.lower()


def test_teacher_is_frozen():
    from src.model import build_teacher

    teacher = build_teacher(_cfg(teacher="google/flan-t5-small"))
    assert teacher is not None
    assert all(not p.requires_grad for p in teacher.parameters())
    assert teacher.training is False


def test_no_teacher_returns_none():
    from src.model import build_teacher

    assert build_teacher(_cfg()) is None


def test_adapter_save_load_roundtrip(tmp_path):
    import torch

    from src.model import build_student, load_adapter, save_adapter

    cfg = _cfg()
    model = build_student(cfg)
    save_adapter(model, tmp_path / "adapter")
    reloaded = load_adapter(cfg, tmp_path / "adapter")

    orig = dict(model.named_parameters())
    for name, p in reloaded.named_parameters():
        if "lora" in name.lower() and name in orig:
            assert torch.allclose(p, orig[name], atol=1e-5)
