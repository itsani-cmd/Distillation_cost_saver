"""Custom accelerate training loop with end-of-epoch checkpoint/resume.

Supports two modes from a single loop:

* **SFT** -- no teacher configured (or ``kd_weight == 0``). Pure label-smoothed
  cross-entropy.
* **Distillation** -- a frozen teacher is loaded and its soft logits are
  combined with CE via :func:`src.losses.combined_loss`.

Checkpointing is **end-of-epoch only** (per the project decision). A checkpoint
captures everything needed to resume exactly: the LoRA adapter, optimizer,
scheduler, epoch counter, global step, and RNG state (via accelerate's
``save_state``). Resuming restores all of it and continues from the next epoch.

Part 2/3 hook note: the per-step body is a single method
(:meth:`Trainer._train_step`) and the model is a plain HF model, so later parts
can attach per-layer forward hooks or early-exit heads without restructuring.

CLI::

    python -m src.train --config configs/tiny.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from src.config import Config, config_to_dict, load_config
from src.data import build_dataloaders, build_tokenizer
from src.evaluate import evaluate_model
from src.losses import LossOutput, combined_loss
from src.model import build_student, build_teacher, save_adapter, teacher_logits
from utils.colab_utils import ProjectPaths, get_project_paths

# File names inside a checkpoint directory.
ADAPTER_SUBDIR = "adapter"
ACCEL_STATE_SUBDIR = "accel_state"
TRAINER_STATE_FILE = "trainer_state.json"


def _epoch_dir(paths: ProjectPaths, epoch: int) -> Path:
    """Return the checkpoint directory for a completed ``epoch`` (0-indexed)."""
    return paths.checkpoints / f"epoch_{epoch:03d}"


def _latest_completed_epoch(paths: ProjectPaths) -> int | None:
    """Return the highest completed-epoch index on disk, or ``None``."""
    if not paths.checkpoints.exists():
        return None
    epochs = []
    for child in paths.checkpoints.iterdir():
        if child.is_dir() and child.name.startswith("epoch_"):
            try:
                epochs.append(int(child.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(epochs) if epochs else None


class Trainer:
    """Drives SFT / distillation training with epoch checkpointing.

    Attributes:
        cfg: The run config.
        paths: Resolved artifact paths (Drive on Colab, local otherwise).
        tokenizer: The shared T5 tokenizer.
    """

    def __init__(self, cfg: Config, paths: ProjectPaths | None = None) -> None:
        """Build models, data, optimizer, scheduler, and the accelerator.

        Args:
            cfg: The run config.
            paths: Artifact paths. If ``None``, resolved from ``cfg.run_name``.
        """
        self.cfg = cfg
        self.paths = paths or get_project_paths(cfg.run_name)
        set_seed(cfg.training.seed)

        self.accelerator = Accelerator(
            mixed_precision=cfg.training.mixed_precision,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        )

        self.tokenizer: PreTrainedTokenizerBase = build_tokenizer(cfg.model.student_name_or_path)
        self.model = build_student(cfg)
        self.teacher = build_teacher(cfg)  # None unless configured

        self.train_loader, self.eval_loader = build_dataloaders(cfg, self.tokenizer)

        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        steps_per_epoch = math.ceil(
            len(self.train_loader) / cfg.training.gradient_accumulation_steps
        )
        self.total_steps = steps_per_epoch * cfg.training.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * cfg.training.warmup_ratio),
            num_training_steps=self.total_steps,
        )

        # accelerate places everything on the right device(s).
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.scheduler,
        )
        if self.teacher is not None:
            self.teacher = self.teacher.to(self.accelerator.device)

        self.global_step = 0
        self.start_epoch = 0

    # ----------------------------------------------------------------- loss
    def _train_step(self, batch: dict[str, torch.Tensor]) -> LossOutput:
        """Forward the student (and teacher if distilling) and compute the loss.

        This is the single extension point for Part 2/3: per-layer hooks or
        early-exit heads slot in around the student forward here.
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        t_logits = None
        if self.teacher is not None and self.cfg.distillation_enabled:
            t_logits = teacher_logits(self.teacher, batch)

        return combined_loss(
            student_logits=outputs.logits,
            labels=batch["labels"],
            teacher_logits=t_logits,
            kd_weight=self.cfg.distillation.kd_weight,
            temperature=self.cfg.distillation.temperature,
            label_smoothing=self.cfg.training.label_smoothing,
            kd_mode=self.cfg.distillation.kd_mode,
        )

    # ----------------------------------------------------------- checkpoint
    def save_checkpoint(self, epoch: int) -> Path:
        """Save a full end-of-epoch checkpoint and return its directory.

        Saves the LoRA adapter, accelerate state (optimizer + scheduler + RNG),
        and a small trainer-state JSON (epoch, global step).
        """
        ckpt_dir = _epoch_dir(self.paths, epoch)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(self.model)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_adapter(unwrapped, ckpt_dir / ADAPTER_SUBDIR)

        # accelerate persists optimizer, scheduler, and RNG state.
        self.accelerator.save_state(str(ckpt_dir / ACCEL_STATE_SUBDIR))

        if self.accelerator.is_main_process:
            state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "config": config_to_dict(self.cfg),
                "python_rng": random.getstate(),
            }
            # random.getstate() is not JSON-serializable as-is; drop it and rely
            # on accelerate's RNG state for the torch/numpy/cuda generators.
            state.pop("python_rng")
            with (ckpt_dir / TRAINER_STATE_FILE).open("w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        self.accelerator.wait_for_everyone()
        return ckpt_dir

    def maybe_resume(self) -> None:
        """Resume from the latest end-of-epoch checkpoint if one exists."""
        last = _latest_completed_epoch(self.paths)
        if last is None:
            return
        ckpt_dir = _epoch_dir(self.paths, last)
        self.accelerator.load_state(str(ckpt_dir / ACCEL_STATE_SUBDIR))

        state_file = ckpt_dir / TRAINER_STATE_FILE
        if state_file.exists():
            with state_file.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
            self.global_step = int(state.get("global_step", 0))
        self.start_epoch = last + 1
        self.accelerator.print(
            f"[train] Resumed from {ckpt_dir} -> starting at epoch {self.start_epoch}"
        )

    # ---------------------------------------------------------------- train
    def train(self) -> dict[str, object]:
        """Run the full training loop and a final evaluation.

        Returns:
            A dict with the final ``bleu`` and the last epoch's mean loss.
        """
        self.maybe_resume()
        log_path = self.paths.logs / f"{self.cfg.run_name}_train.log"
        last_epoch_loss = float("nan")

        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            self.model.train()
            running = 0.0
            seen = 0
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    loss_out = self._train_step(batch)
                    self.accelerator.backward(loss_out.total)
                    if self.accelerator.sync_gradients and self.cfg.training.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.cfg.training.max_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    running += loss_out.total.item()
                    seen += 1
                    if self.global_step % self.cfg.training.log_every == 0:
                        msg = (
                            f"epoch={epoch} step={self.global_step} "
                            f"loss={loss_out.total.item():.4f} "
                            f"ce={loss_out.ce.item():.4f} kd={loss_out.kd.item():.4f}"
                        )
                        self.accelerator.print(f"[train] {msg}")
                        self._append_log(log_path, msg)

            last_epoch_loss = running / max(seen, 1)
            self.accelerator.print(f"[train] epoch {epoch} done mean_loss={last_epoch_loss:.4f}")
            self.save_checkpoint(epoch)

        # Final evaluation.
        pred_path = self.paths.outputs / f"{self.cfg.run_name}_predictions.jsonl"
        eval_out = evaluate_model(
            self.accelerator.unwrap_model(self.model),
            self.eval_loader,
            self.tokenizer,
            self.cfg,
            device=self.accelerator.device,
            output_path=pred_path,
        )
        bleu = eval_out["bleu"]
        self.accelerator.print(f"[train] final BLEU={bleu:.2f} predictions={pred_path}")
        return {"bleu": bleu, "last_epoch_loss": last_epoch_loss}

    def _append_log(self, path: Path, line: str) -> None:
        """Append a log line on the main process only."""
        if not self.accelerator.is_main_process:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _seed_everything(seed: int) -> None:
    """Seed python/numpy/torch for reproducibility outside accelerate too."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    """CLI entry point: load a config and run training."""
    parser = argparse.ArgumentParser(description="Train distill-translate (SFT or distillation).")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _seed_everything(cfg.training.seed)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
