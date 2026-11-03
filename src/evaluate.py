"""Generation + sacreBLEU evaluation and prediction export.

Generates translations for the eval split, scores them with sacreBLEU, and
writes a predictions file (source / reference / hypothesis) to the run's
Drive-routed output directory.

Public surface:
    - :func:`compute_bleu` -> pure, testable BLEU over strings
    - :func:`save_predictions` -> write a predictions JSONL file
    - :func:`evaluate_model` -> full generate + score + save
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import IGNORE_INDEX

if TYPE_CHECKING:  # type-only import; avoids requiring transformers for unit tests
    from transformers import PreTrainedTokenizerBase


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute a corpus sacreBLEU score.

    Args:
        predictions: Hypothesis strings.
        references: Reference strings (one per prediction).

    Returns:
        The corpus BLEU score (0-100).
    """
    import sacrebleu

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    # sacreBLEU expects references as a list of reference-streams.
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return float(bleu.score)


def save_predictions(
    path: str | Path,
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    bleu: float | None = None,
) -> Path:
    """Write predictions to a JSONL file and return the path.

    The first line is a header record with the corpus BLEU (if provided); each
    subsequent line is one ``{"source", "reference", "hypothesis"}`` record.

    Args:
        path: Destination file path.
        sources: Source strings.
        references: Reference strings.
        hypotheses: Hypothesis strings.
        bleu: Optional corpus BLEU to record in the header line.

    Returns:
        The destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "summary", "bleu": bleu, "n": len(hypotheses)}) + "\n")
        for src, ref, hyp in zip(sources, references, hypotheses):
            fh.write(
                json.dumps(
                    {"source": src, "reference": ref, "hypothesis": hyp},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _decode_labels(labels: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Decode label ids to strings, restoring pad ids masked as ``-100``."""
    labels = labels.clone()
    labels[labels == IGNORE_INDEX] = tokenizer.pad_token_id
    return tokenizer.batch_decode(labels, skip_special_tokens=True)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    cfg: Config,
    device: torch.device | None = None,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate translations, compute BLEU, and optionally save predictions.

    Args:
        model: The (LoRA-wrapped) student in eval mode.
        eval_loader: Dataloader over the eval split.
        tokenizer: The shared T5 tokenizer.
        cfg: The run config.
        device: Device to run generation on (defaults to the model's device).
        output_path: If set, write a predictions JSONL file here.

    Returns:
        A dict with ``bleu`` and the ``sources``/``references``/``hypotheses``.
    """
    was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    sources: list[str] = []
    references: list[str] = []
    hypotheses: list[str] = []

    gen_model = getattr(model, "module", model)
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated = gen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=cfg.data.max_target_length,
            num_beams=4,
        )
        hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        references.extend(_decode_labels(batch["labels"], tokenizer))
        sources.extend(tokenizer.batch_decode(input_ids, skip_special_tokens=True))

    bleu = compute_bleu(hypotheses, references)

    if output_path is not None:
        save_predictions(output_path, sources, references, hypotheses, bleu=bleu)

    if was_training:
        model.train()

    return {
        "bleu": bleu,
        "sources": sources,
        "references": references,
        "hypotheses": hypotheses,
    }
