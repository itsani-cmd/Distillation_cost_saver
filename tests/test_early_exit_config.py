"""Tests for the early_exit config section."""

import pytest

from src.config import config_from_dict, config_to_dict, load_config


def test_early_exit_off_by_default():
    cfg = config_from_dict({})
    assert cfg.early_exit.enabled is False
    assert cfg.early_exit.self_distill_weight == 0.0
    # Existing configs are unaffected.
    assert load_config("configs/tiny.yaml").early_exit.enabled is False
    assert load_config("configs/base.yaml").early_exit.enabled is False


@pytest.mark.parametrize(
    "name, sd_weight",
    [
        ("ee_tiny_sft", 0.0),
        ("ee_tiny_selfdistill", 0.5),
        ("ee_mid_sft", 0.0),
        ("ee_mid_selfdistill", 0.5),
    ],
)
def test_early_exit_yaml_configs(name, sd_weight):
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.run_name == name
    assert cfg.early_exit.enabled is True
    assert cfg.early_exit.self_distill_weight == sd_weight
    assert cfg.distillation_enabled is False  # Options A/B use no teacher


def test_option_a_and_b_differ_only_in_self_distill():
    for size in ("tiny", "mid"):
        a = config_to_dict(load_config(f"configs/ee_{size}_sft.yaml"))
        b = config_to_dict(load_config(f"configs/ee_{size}_selfdistill.yaml"))
        for d in (a, b):
            d.pop("run_name")
            d["early_exit"].pop("self_distill_weight")
        assert a == b


def test_roundtrip_keeps_early_exit():
    cfg = config_from_dict({"early_exit": {"enabled": True, "self_distill_weight": 0.3}})
    rebuilt = config_from_dict(config_to_dict(cfg))
    assert rebuilt.early_exit.enabled is True
    assert rebuilt.early_exit.self_distill_weight == 0.3


@pytest.mark.parametrize(
    "ee",
    [
        {"enabled": True, "layer_weighting": "cubic"},
        {"enabled": True, "self_distill_weight": -0.1},
        {"enabled": True, "self_distill_temperature": 0.0},
        {"enabled": True, "measure": "vibes"},
        {"enabled": True, "threshold": 1.5},
        {"enabled": True, "static_layer": 0},
        {"enabled": True, "decay_tau": -1.0},
        # Training-only options without enabling early exit are a misconfiguration.
        {"enabled": False, "self_distill_weight": 0.5},
        {"enabled": False, "train_final_norm": True},
        {"not_a_field": 1},
    ],
)
def test_early_exit_validation_rejects_bad_values(ee):
    with pytest.raises(ValueError):
        config_from_dict({"early_exit": ee})
