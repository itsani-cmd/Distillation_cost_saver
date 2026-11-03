"""Colab detection, Drive mounting, and project path routing.

This module is the single source of truth for *where artifacts go*. Training
and evaluation never hard-code paths; they call :func:`get_project_paths` so the
same code writes to Google Drive on Colab and to a local directory otherwise.

Typical Colab usage (first notebook cell)::

    from utils.colab_utils import setup_colab
    paths = setup_colab()          # installs deps, mounts Drive, returns paths

Typical local usage::

    from utils.colab_utils import get_project_paths
    paths = get_project_paths("tiny")
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Where artifacts live on Colab (under the mounted Drive) and locally.
DRIVE_MOUNT_POINT = "/content/drive"
DRIVE_PROJECT_SUBDIR = "MyDrive/distill-translate"


@dataclass
class ProjectPaths:
    """Resolved artifact directories for a run.

    All directories are created on access via :func:`get_project_paths`.

    Attributes:
        root: Project root for artifacts (Drive on Colab, local otherwise).
        checkpoints: Where end-of-epoch checkpoints are written.
        logs: Where training logs are written.
        outputs: Where evaluation outputs (predictions) are written.
    """

    root: Path
    checkpoints: Path
    logs: Path
    outputs: Path


def is_colab() -> bool:
    """Return ``True`` when running inside a Google Colab runtime.

    Detection is import-based (does the ``google.colab`` module exist) so it
    works without importing the module, and can be monkeypatched in tests.

    ``importlib.util.find_spec`` imports the *parent* package to inspect a
    submodule, so looking up ``google.colab`` raises ``ModuleNotFoundError``
    when no ``google`` namespace exists at all (common off-Colab). We treat any
    such lookup failure as "not Colab".
    """
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def mount_drive(mount_point: str = DRIVE_MOUNT_POINT, force_remount: bool = False) -> Path:
    """Mount Google Drive on Colab and return the mount point.

    Args:
        mount_point: Where to mount Drive.
        force_remount: Force a remount even if already mounted.

    Returns:
        The mount point as a :class:`~pathlib.Path`.

    Raises:
        RuntimeError: If called outside Colab.
    """
    if not is_colab():
        raise RuntimeError("mount_drive() can only be called inside Google Colab.")
    # Imported lazily so the module imports fine off-Colab (and in tests).
    from google.colab import drive  # type: ignore

    drive.mount(mount_point, force_remount=force_remount)
    return Path(mount_point)


def _default_local_root() -> Path:
    """Local artifact root: an ``artifacts/`` dir next to the repo root.

    Overridable with the ``DISTILL_TRANSLATE_HOME`` environment variable.
    """
    env = os.environ.get("DISTILL_TRANSLATE_HOME")
    if env:
        return Path(env).expanduser()
    # utils/colab_utils.py -> repo root is one level up from utils/.
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "artifacts"


def get_project_paths(run_name: str = "default", create: bool = True) -> ProjectPaths:
    """Resolve (and optionally create) artifact directories for a run.

    On Colab, artifacts live under the mounted Drive; otherwise under a local
    ``artifacts/`` directory. The ``run_name`` namespaces each run so multiple
    experiments do not collide.

    Args:
        run_name: Name that namespaces this run's artifacts.
        create: If ``True``, create the directories.

    Returns:
        A :class:`ProjectPaths` for the run.
    """
    if is_colab():
        base = Path(DRIVE_MOUNT_POINT) / DRIVE_PROJECT_SUBDIR
    else:
        base = _default_local_root()

    root = base / run_name
    paths = ProjectPaths(
        root=root,
        checkpoints=root / "checkpoints",
        logs=root / "logs",
        outputs=root / "outputs",
    )
    if create:
        for d in (paths.root, paths.checkpoints, paths.logs, paths.outputs):
            d.mkdir(parents=True, exist_ok=True)
    return paths


def _pip_install(requirements_file: str | Path) -> None:
    """Install packages from a requirements file via pip (best effort)."""
    req = Path(requirements_file)
    if not req.exists():
        print(f"[colab] requirements file not found, skipping install: {req}", file=sys.stderr)
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
        check=True,
    )


def _fix_colab_env() -> None:
    """Work around Colab-preinstalled package clashes before importing peft.

    Colab ships an older ``torchao`` (e.g. 0.10.0). Newer ``peft`` eagerly
    imports torchao and requires ``> 0.16.0``, so ``import peft`` raises an
    ``ImportError`` before any of our code runs. We do not use torchao, so the
    simplest robust fix is to uninstall the stale copy. Best effort: failures
    here are non-fatal and printed, not raised.
    """
    try:
        import torchao  # type: ignore  # noqa: F401
    except Exception:
        return  # torchao not present -> nothing to fix.

    print("[colab] Removing Colab's preinstalled torchao (unused, clashes with peft).")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"],
        check=False,
    )


def setup_colab(
    run_name: str = "default",
    requirements_file: str | Path = "requirements.txt",
    install: bool = True,
) -> ProjectPaths:
    """One-call Colab bootstrap: install deps, mount Drive, resolve paths.

    Safe to call off-Colab too: it skips the Drive mount and simply returns
    local paths, which makes notebooks runnable locally for CI-style checks.

    Args:
        run_name: Name that namespaces this run's artifacts.
        requirements_file: Requirements file to install from (Colab only).
        install: Whether to pip-install requirements on Colab.

    Returns:
        A :class:`ProjectPaths` for the run.
    """
    if is_colab():
        if install:
            _pip_install(requirements_file)
            _fix_colab_env()
        mount_drive()
    else:
        print("[colab] Not running on Colab; using local artifact paths.")
    paths = get_project_paths(run_name=run_name, create=True)
    print(f"[colab] Artifacts root: {paths.root}")
    return paths
