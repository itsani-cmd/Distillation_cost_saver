"""Tests for the config dataclasses and YAML loader/validation."""

import pytest

from src.config import (
    Config,
    config_from_dict,
    config_to_dict,
    load_config,
)

TINY_PATH = "configs/tiny.yaml"


def test_load_tiny_yaml_fields():
    cfg = load_config(TINY_PATH)
    assert cfg.run_name == "tiny"
    assert cfg.model.student_name_or_path == "t5-small"
    assert cfg.model.teacher_name_or_path is None
    assert cfg.lora.target_modules == ["q", "v"]
    assert cfg.data.dataset_name == "Helsinki-NLP/opus-100"
    assert cfg.data.source_lang == "en"
    assert cfg.data.target_lang == "de"
    assert cfg.training.num_epochs == 2
    assert cfg.distillation.kd_mode == "word"


def test_defaults_build():
    cfg = config_from_dict({})
    assert isinstance(cfg, Config)
    assert cfg.distillation.kd_weight == 0.0
    assert cfg.distillation_enabled is False


def test_distillation_enabled_property():
    cfg = config_from_dict(
        {
            "model": {"teacher_name_or_path": "google/flan-t5-base"},
            "distillation": {"kd_weight": 0.5},
        }
    )
    assert cfg.distillation_enabled is True


def test_roundtrip_to_dict():
    cfg = config_from_dict({})
    d = config_to_dict(cfg)
    assert d["model"]["student_name_or_path"] == "t5-small"
    # Rebuilding from the serialized dict should succeed.
    rebuilt = config_from_dict(d)
    assert rebuilt.lora.r == cfg.lora.r


@pytest.mark.parametrize(
    "raw",
    [
        {"distillation": {"kd_weight": 1.5}},
        {"distillation": {"kd_weight": -0.1}},
        {"distillation": {"temperature": 0.0}},
        {"distillation": {"kd_mode": "bogus"}},
        {"training": {"learning_rate": 0.0}},
        {"training": {"num_epochs": 0}},
        {"training": {"warmup_ratio": 1.0}},
        {"training": {"mixed_precision": "int4"}},
        {"lora": {"r": 0}},
        {"lora": {"target_modules": []}},
        # KD weight > 0 without a teacher is a misconfiguration.
        {"distillation": {"kd_weight": 0.5}},
    ],
)
def test_validation_rejects_bad_values(raw):
    with pytest.raises(ValueError):
        config_from_dict(raw)


def test_unknown_keys_rejected():
    with pytest.raises(ValueError):
        config_from_dict({"model": {"not_a_field": 1}})
    with pytest.raises(ValueError):
        config_from_dict({"totally_unknown": 1})


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does_not_exist.yaml")
