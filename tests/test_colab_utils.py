"""Tests for Colab detection and Drive/local path routing."""

from pathlib import Path

import utils.colab_utils as cu


def test_get_project_paths_local_creates_dirs(tmp_path, monkeypatch):
    # Force the "not Colab" branch and a temp artifact home.
    monkeypatch.setattr(cu, "is_colab", lambda: False)
    monkeypatch.setenv("DISTILL_TRANSLATE_HOME", str(tmp_path))

    paths = cu.get_project_paths("run_a", create=True)

    assert paths.root == tmp_path / "run_a"
    for d in (paths.root, paths.checkpoints, paths.logs, paths.outputs):
        assert d.exists() and d.is_dir()


def test_get_project_paths_colab_branch(monkeypatch):
    # Force the Colab branch; do not create (no real Drive mount in tests).
    monkeypatch.setattr(cu, "is_colab", lambda: True)
    paths = cu.get_project_paths("run_b", create=False)

    expected_root = Path(cu.DRIVE_MOUNT_POINT) / cu.DRIVE_PROJECT_SUBDIR / "run_b"
    assert paths.root == expected_root
    assert paths.checkpoints == expected_root / "checkpoints"
    assert paths.outputs == expected_root / "outputs"


def test_mount_drive_off_colab_raises(monkeypatch):
    monkeypatch.setattr(cu, "is_colab", lambda: False)
    try:
        cu.mount_drive()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_setup_colab_off_colab_returns_local_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cu, "is_colab", lambda: False)
    monkeypatch.setenv("DISTILL_TRANSLATE_HOME", str(tmp_path))

    paths = cu.setup_colab("run_c")
    out = capsys.readouterr().out

    assert paths.root == tmp_path / "run_c"
    assert paths.root.exists()
    assert "Not running on Colab" in out
