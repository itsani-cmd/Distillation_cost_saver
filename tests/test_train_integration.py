"""Integration tests for the training loop (SFT + distillation + resume).

Gated behind RUN_HEAVY_TESTS=1 (they download t5-small and run on CPU). Run:

    RUN_HEAVY_TESTS=1 pytest tests/test_train_integration.py
"""

import os

import pytest

RUN_HEAVY = os.environ.get("RUN_HEAVY_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_HEAVY, reason="set RUN_HEAVY_TESTS=1 to run training integration tests"
)


def _tiny_cfg(tmp_path, teacher=None):
    from src.config import config_from_dict

    raw = {
        "run_name": "it",
        "model": {"student_name_or_path": "t5-small"},
        "lora": {"r": 4, "alpha": 8},
        "data": {
            "train_subset": 8,
            "eval_subset": 4,
            "max_source_length": 16,
            "max_target_length": 16,
            "num_workers": 0,
        },
        "training": {"batch_size": 4, "eval_batch_size": 4, "num_epochs": 2, "log_every": 1},
    }
    if teacher is not None:
        raw["model"]["teacher_name_or_path"] = teacher
        raw["distillation"] = {"kd_weight": 0.5}
    return config_from_dict(raw)


def _paths(tmp_path):
    import utils.colab_utils as cu

    os.environ["DISTILL_TRANSLATE_HOME"] = str(tmp_path)
    return cu.get_project_paths("it", create=True)


def test_sft_writes_checkpoint_per_epoch_and_resumes(tmp_path):
    from src.train import Trainer, _epoch_dir

    cfg = _tiny_cfg(tmp_path)
    paths = _paths(tmp_path)

    trainer = Trainer(cfg, paths=paths)
    trainer.train()

    assert _epoch_dir(paths, 0).exists()
    assert _epoch_dir(paths, 1).exists()

    # A fresh trainer should resume and start past the last completed epoch.
    trainer2 = Trainer(cfg, paths=paths)
    trainer2.maybe_resume()
    assert trainer2.start_epoch == 2


def test_distillation_runs_and_teacher_stays_frozen(tmp_path):
    from src.train import Trainer

    cfg = _tiny_cfg(tmp_path, teacher="google/flan-t5-small")
    cfg.training.num_epochs = 1
    paths = _paths(tmp_path)

    trainer = Trainer(cfg, paths=paths)
    assert trainer.teacher is not None
    before = [p.detach().clone() for p in trainer.teacher.parameters()]
    trainer.train()
    after = list(trainer.teacher.parameters())

    import torch

    assert all(torch.equal(a, b) for a, b in zip(before, after))
