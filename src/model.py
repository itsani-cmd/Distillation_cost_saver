"""Student (LoRA) and optional frozen teacher model construction.

The student is a T5 encoder-decoder wrapped with a peft LoRA adapter so only a
small set of parameters train. The teacher, when configured, is loaded frozen
in eval mode and only ever used for forward passes under ``torch.no_grad``.

Part 2/3 hook note: because this returns plain HF models (with LoRA on the
student), later parts can register forward hooks on individual
``encoder.block[i]`` / ``decoder.block[i]`` layers, or attach early-exit heads,
without any special wrapper to fight.

Public surface:
    - :func:`build_student` -> LoRA-wrapped student + trainable-param summary
    - :func:`build_teacher` -> frozen teacher (or ``None`` if not configured)
    - :func:`count_parameters` -> (trainable, total)
    - :func:`save_adapter` / :func:`load_adapter` -> LoRA adapter round-trip
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM

from src.config import Config

# peft matches modules_to_save by name suffix; this hits only the decoder's
# final norm (not the per-block "layer_norm"s, nor the encoder's final norm).
FINAL_NORM_MODULES = ("decoder.final_layer_norm",)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts for ``model``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_student(cfg: Config) -> PeftModel:
    """Load the student and wrap it with a LoRA adapter.

    Only the LoRA parameters are left trainable; the base weights are frozen by
    peft. Exception: with ``early_exit.train_final_norm`` the decoder's final
    layer norm (shared by every early-exit head) is also trained, via peft's
    ``modules_to_save`` (so it is saved/loaded with the adapter).

    Args:
        cfg: The run config (reads ``model.student_name_or_path``, ``lora`` and
            ``early_exit.train_final_norm``).

    Returns:
        A peft ``PeftModel`` wrapping the T5 student.
    """
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.student_name_or_path)
    ee = cfg.early_exit
    modules_to_save = list(FINAL_NORM_MODULES) if ee.enabled and ee.train_final_norm else None
    peft_cfg = PeftLoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(base, peft_cfg)
    return model


def build_teacher(cfg: Config) -> nn.Module | None:
    """Load the teacher frozen in eval mode, or return ``None``.

    Args:
        cfg: The run config (reads ``model.teacher_name_or_path``).

    Returns:
        A frozen teacher model, or ``None`` when no teacher is configured.
    """
    name = cfg.model.teacher_name_or_path
    if name is None:
        return None
    teacher = AutoModelForSeq2SeqLM.from_pretrained(name)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def save_adapter(model: PeftModel, directory: str | Path) -> None:
    """Save only the LoRA adapter weights to ``directory``.

    Args:
        model: The peft-wrapped student.
        directory: Destination directory (created if needed).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))


def load_adapter(cfg: Config, directory: str | Path) -> PeftModel:
    """Reload a student with LoRA adapter weights from ``directory``.

    Args:
        cfg: The run config (to reload the matching base student).
        directory: Directory previously written by :func:`save_adapter`.

    Returns:
        A peft ``PeftModel`` with the saved adapter applied.
    """
    base = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.student_name_or_path)
    return PeftModel.from_pretrained(base, str(directory), is_trainable=True)


@torch.no_grad()
def teacher_logits(teacher: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute teacher logits for a batch without tracking gradients.

    Args:
        teacher: The frozen teacher model.
        batch: Dict with ``input_ids``, ``attention_mask``, ``labels``.

    Returns:
        Logits tensor of shape ``(batch, target_len, vocab)``.
    """
    out = teacher(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.logits
