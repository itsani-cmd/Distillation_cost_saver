"""Configuration dataclasses and a validated YAML loader.

The config is the single contract every other module reads from. It is split
into logical sections (model, lora, data, training, distillation) so that
contributors working on different modules only need the section they care about.

Example:
    >>> from src.config import load_config
    >>> cfg = load_config("configs/tiny.yaml")
    >>> cfg.model.student_name_or_path
    't5-small'
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Supported knowledge-distillation modes. Part 1 implements "word" (word-level
# KL on soft logits). "sequence" is a documented extension point for later.
KD_MODES = ("word", "sequence")


@dataclass
class ModelConfig:
    """Student and (optional) teacher model identifiers.

    Attributes:
        student_name_or_path: HF id or local path of the student (e.g. ``t5-small``).
        teacher_name_or_path: HF id or local path of the teacher used for
            distillation. ``None`` disables distillation (pure SFT).
        prefix: T5-style task prefix prepended to every source sentence.
    """

    student_name_or_path: str = "t5-small"
    teacher_name_or_path: str | None = None
    prefix: str = "translate English to German: "


@dataclass
class LoraConfig:
    """LoRA hyperparameters applied to the student via peft.

    Attributes:
        r: LoRA rank.
        alpha: LoRA scaling factor (``lora_alpha``).
        dropout: Dropout applied inside the LoRA layers.
        target_modules: Projection submodule names to adapt. For T5, ``q`` and
            ``v`` are the common minimal set; ``k``/``o``/``wi``/``wo`` optional.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q", "v"])


@dataclass
class DataConfig:
    """Dataset selection and tokenization limits.

    Attributes:
        dataset_name: HF dataset name (``opus100`` by default).
        source_lang: Source language code.
        target_lang: Target language code.
        config_name: Dataset config; for opus100 this is ``de-en``.
        max_source_length: Max source tokens.
        max_target_length: Max target tokens.
        train_subset: If set, cap the train split to this many examples.
        eval_subset: If set, cap the eval split to this many examples.
        num_workers: Dataloader workers.
    """

    dataset_name: str = "opus100"
    source_lang: str = "en"
    target_lang: str = "de"
    config_name: str = "de-en"
    max_source_length: int = 128
    max_target_length: int = 128
    train_subset: int | None = None
    eval_subset: int | None = None
    num_workers: int = 2


@dataclass
class TrainingConfig:
    """Optimization and loop hyperparameters.

    Attributes:
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Per-device training batch size.
        eval_batch_size: Per-device eval batch size.
        num_epochs: Number of epochs.
        warmup_ratio: Fraction of total steps used for LR warmup.
        label_smoothing: Label smoothing factor for the CE loss.
        max_grad_norm: Gradient clipping norm (``<= 0`` disables clipping).
        seed: Global RNG seed.
        mixed_precision: accelerate mixed precision: ``no``/``fp16``/``bf16``.
        log_every: Log training loss every N steps.
        gradient_accumulation_steps: Micro-batches per optimizer step.
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 2
    warmup_ratio: float = 0.05
    label_smoothing: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    mixed_precision: str = "no"
    log_every: int = 10
    gradient_accumulation_steps: int = 1


@dataclass
class DistillationConfig:
    """Knowledge-distillation settings.

    The combined loss is ``(1 - kd_weight) * CE + kd_weight * KD``.

    Attributes:
        kd_weight: Weight on the KD term in ``[0, 1]``. ``0`` -> pure SFT.
        temperature: Softmax temperature for soft-logit KL (``> 0``).
        kd_mode: One of :data:`KD_MODES`. Part 1 implements ``word``.
    """

    kd_weight: float = 0.0
    temperature: float = 2.0
    kd_mode: str = "word"


@dataclass
class Config:
    """Top-level config aggregating all sections.

    Attributes:
        run_name: Human-readable name used in artifact paths.
        model: :class:`ModelConfig`.
        lora: :class:`LoraConfig`.
        data: :class:`DataConfig`.
        training: :class:`TrainingConfig`.
        distillation: :class:`DistillationConfig`.
    """

    run_name: str = "tiny"
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)

    @property
    def distillation_enabled(self) -> bool:
        """Whether a teacher is configured and the KD term is active."""
        return self.model.teacher_name_or_path is not None and self.distillation.kd_weight > 0.0


def _build_section(section_cls: type, data: dict[str, Any]) -> Any:
    """Instantiate a dataclass section from a dict, rejecting unknown keys.

    Args:
        section_cls: The dataclass type to build.
        data: Mapping of field name -> value from YAML.

    Returns:
        An instance of ``section_cls``.

    Raises:
        ValueError: If ``data`` contains keys not defined on the dataclass.
    """
    if data is None:
        return section_cls()
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping for {section_cls.__name__}, got {type(data)}.")
    valid = {f.name for f in fields(section_cls)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**data)


def _validate(cfg: Config) -> None:
    """Validate a fully-built config, raising ``ValueError`` on bad values."""
    d = cfg.distillation
    if not 0.0 <= d.kd_weight <= 1.0:
        raise ValueError(f"distillation.kd_weight must be in [0, 1], got {d.kd_weight}")
    if d.temperature <= 0.0:
        raise ValueError(f"distillation.temperature must be > 0, got {d.temperature}")
    if d.kd_mode not in KD_MODES:
        raise ValueError(f"distillation.kd_mode must be one of {KD_MODES}, got {d.kd_mode!r}")

    t = cfg.training
    if t.learning_rate <= 0.0:
        raise ValueError(f"training.learning_rate must be > 0, got {t.learning_rate}")
    if t.batch_size <= 0 or t.eval_batch_size <= 0:
        raise ValueError("training batch sizes must be > 0")
    if t.num_epochs <= 0:
        raise ValueError(f"training.num_epochs must be > 0, got {t.num_epochs}")
    if not 0.0 <= t.warmup_ratio < 1.0:
        raise ValueError(f"training.warmup_ratio must be in [0, 1), got {t.warmup_ratio}")
    if not 0.0 <= t.label_smoothing < 1.0:
        raise ValueError(f"training.label_smoothing must be in [0, 1), got {t.label_smoothing}")
    if t.mixed_precision not in ("no", "fp16", "bf16"):
        raise ValueError(
            f"training.mixed_precision must be no/fp16/bf16, got {t.mixed_precision!r}"
        )
    if t.gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be > 0")

    lora = cfg.lora
    if lora.r <= 0 or lora.alpha <= 0:
        raise ValueError("lora.r and lora.alpha must be > 0")
    if not 0.0 <= lora.dropout < 1.0:
        raise ValueError(f"lora.dropout must be in [0, 1), got {lora.dropout}")
    if not lora.target_modules:
        raise ValueError("lora.target_modules must be a non-empty list")

    data = cfg.data
    if data.max_source_length <= 0 or data.max_target_length <= 0:
        raise ValueError("data max lengths must be > 0")

    # Distillation requested but no teacher -> misconfiguration worth catching.
    if cfg.distillation.kd_weight > 0.0 and cfg.model.teacher_name_or_path is None:
        raise ValueError(
            "distillation.kd_weight > 0 requires model.teacher_name_or_path to be set."
        )


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build and validate a :class:`Config` from a plain dict.

    Args:
        raw: Mapping with optional ``run_name`` and section keys.

    Returns:
        A validated :class:`Config`.

    Raises:
        ValueError: If unknown keys are present or validation fails.
    """
    raw = dict(raw or {})
    valid_top = {f.name for f in fields(Config)}
    unknown = set(raw) - valid_top
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    cfg = Config(
        run_name=raw.get("run_name", "tiny"),
        model=_build_section(ModelConfig, raw.get("model")),
        lora=_build_section(LoraConfig, raw.get("lora")),
        data=_build_section(DataConfig, raw.get("data")),
        training=_build_section(TrainingConfig, raw.get("training")),
        distillation=_build_section(DistillationConfig, raw.get("distillation")),
    )
    _validate(cfg)
    return cfg


def load_config(path: str | Path) -> Config:
    """Load, build, and validate a config from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        A validated :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is malformed or fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is not None and not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw)}.")
    return config_from_dict(raw or {})


def config_to_dict(cfg: Any) -> Any:
    """Recursively convert a config (dataclass tree) to plain dicts.

    Useful for logging a run's exact configuration alongside checkpoints.

    Args:
        cfg: A dataclass instance or nested value.

    Returns:
        A JSON/YAML-serializable structure.
    """
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: config_to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [config_to_dict(v) for v in cfg]
    return cfg
