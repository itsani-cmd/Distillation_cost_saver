"""Checkpoint 1: did the early decoder layers learn to guess?

Runs the eval split through the model with the correct translation fed in
("teacher forcing", no generation, so it is fast) and reports, for every
decoder layer:

* **accuracy** -- how often the layer's top word equals the correct word;
* **agreement_with_final** -- how often it equals the LAST layer's top word.
  This is the key early-exit number: if layer 2 agrees with layer 6 on 80% of
  words, those words could stop at layer 2 without changing the output;
* **mean_confidence** -- average softmax gap (top-1 minus top-2 probability),
  split into "when it agrees" vs "when it disagrees". A big difference means
  confidence is a good signal for when to stop;

plus an **oracle** estimate: if every word stopped at the earliest layer that
already agrees with the last layer, how many layers would we use on average?
That is the best case any stopping rule can reach (Part 2 measures real rules).

Note: with teacher forcing every layer is computed for every earlier word, so
this is an estimate; Part 2's early-exit decoder measures the real thing.

CLI::

    # report for a trained run (uses its latest checkpoint)
    python -m src.early_exit.analysis --config configs/ee_mid_sft.yaml
    python -m src.early_exit.analysis --config configs/ee_mid_selfdistill.yaml

    # same model with NO adapter, as a "before training" reference
    python -m src.early_exit.analysis --config configs/ee_mid_sft.yaml --no-adapter

    # side-by-side comparison of saved reports
    python -m src.early_exit.analysis --compare A_report.json B_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.config import Config, load_config
from src.data import (
    IGNORE_INDEX,
    Seq2SeqCollator,
    build_tokenizer,
    load_raw_dataset,
    tokenize_example,
)
from src.early_exit.confidence import softmax_gap
from src.early_exit.heads import exit_logits, num_decoder_layers


@torch.no_grad()
def layer_report(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Compute per-layer accuracy / agreement / confidence and the oracle estimate.

    Args:
        model: The (possibly LoRA-wrapped) T5 student.
        loader: Batches with ``input_ids``, ``attention_mask``, ``labels``.
        device: Device to run on.
        max_batches: Optional cap on the number of batches.

    Returns:
        A JSON-serializable dict (see :func:`format_report` for the layout).
    """
    was_training = model.training
    model.eval()
    n_layers = num_decoder_layers(model)

    correct = torch.zeros(n_layers, dtype=torch.long)
    agree = torch.zeros(n_layers, dtype=torch.long)
    gap_sum = torch.zeros(n_layers, dtype=torch.float64)
    gap_agree_sum = torch.zeros(n_layers, dtype=torch.float64)
    gap_disagree_sum = torch.zeros(n_layers, dtype=torch.float64)
    oracle_hist = torch.zeros(n_layers, dtype=torch.long)
    n_tokens = 0

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        labels = batch["labels"].to(device)
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels,
            output_hidden_states=True,
        )
        mask = labels != IGNORE_INDEX
        gold = labels[mask]  # (N,)
        n = gold.numel()
        if n == 0:
            continue

        final_pred = out.logits[mask].argmax(dim=-1)
        preds, gaps = [], []
        hidden = out.decoder_hidden_states
        for i in range(n_layers):
            # One layer at a time so we never hold six full-vocabulary tensors.
            logits = out.logits if i == n_layers - 1 else exit_logits(model, hidden[i + 1])
            logits = logits[mask]
            preds.append(logits.argmax(dim=-1))
            gaps.append(softmax_gap(logits))
            del logits
        preds_t = torch.stack(preds)  # (L, N)
        gaps_t = torch.stack(gaps).double()  # (L, N)
        agree_t = preds_t == final_pred.unsqueeze(0)  # (L, N); last row all True

        correct += (preds_t == gold.unsqueeze(0)).sum(dim=1).cpu()
        agree += agree_t.sum(dim=1).cpu()
        gap_sum += gaps_t.sum(dim=1).cpu()
        gap_agree_sum += (gaps_t * agree_t).sum(dim=1).cpu()
        gap_disagree_sum += (gaps_t * ~agree_t).sum(dim=1).cpu()
        # Earliest layer that agrees with the final layer (0-based index).
        first_agree = agree_t.float().argmax(dim=0)
        oracle_hist += torch.bincount(first_agree, minlength=n_layers).cpu()
        n_tokens += n

    if was_training:
        model.train()
    if n_tokens == 0:
        raise ValueError("no valid (non-pad) target tokens in the evaluated data")

    layers = []
    for i in range(n_layers):
        n_agree = int(agree[i])
        n_disagree = n_tokens - n_agree
        layers.append(
            {
                "layer": i + 1,
                "accuracy": float(correct[i]) / n_tokens,
                "agreement_with_final": n_agree / n_tokens,
                "mean_confidence": float(gap_sum[i]) / n_tokens,
                "mean_confidence_when_agree": (
                    float(gap_agree_sum[i]) / n_agree if n_agree else None
                ),
                "mean_confidence_when_disagree": (
                    float(gap_disagree_sum[i]) / n_disagree if n_disagree else None
                ),
            }
        )
    hist = (oracle_hist.double() / n_tokens).tolist()
    avg_exit = sum((i + 1) * p for i, p in enumerate(hist))
    return {
        "num_layers": n_layers,
        "num_tokens": n_tokens,
        "layers": layers,
        "oracle": {
            "avg_exit_layer": avg_exit,
            "max_speedup_estimate": n_layers / avg_exit,
            "exit_layer_fraction": {str(i + 1): p for i, p in enumerate(hist)},
        },
    }


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "   -  "
    return f"{100 * x:5.1f}%" if pct else f"{x:6.3f}"


def format_report(report: dict[str, Any], title: str = "") -> str:
    """Render a report dict as a readable table."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"tokens evaluated: {report['num_tokens']}")
    header = (
        "layer",
        "accuracy",
        "agrees w/ last",
        "confidence",
        "conf|agree",
        "conf|disagree",
    )
    widths = (5, 9, 14, 10, 10, 13)
    lines.append(" | ".join(h.rjust(w) for h, w in zip(header, widths)))
    for r in report["layers"]:
        cells = (
            f"L{r['layer']}",
            _fmt(r["accuracy"], pct=True),
            _fmt(r["agreement_with_final"], pct=True),
            _fmt(r["mean_confidence"]),
            _fmt(r["mean_confidence_when_agree"]),
            _fmt(r["mean_confidence_when_disagree"]),
        )
        lines.append(" | ".join(c.rjust(w) for c, w in zip(cells, widths)))
    o = report["oracle"]
    lines.append(
        f"oracle: avg exit layer {o['avg_exit_layer']:.2f} of {report['num_layers']} "
        f"(best-case ~{o['max_speedup_estimate']:.1f}x fewer decoder layers)"
    )
    lines.append(
        "oracle exit-layer mix: "
        + "  ".join(f"L{k}={100 * v:.0f}%" for k, v in o["exit_layer_fraction"].items())
    )
    return "\n".join(lines)


def format_comparison(a: dict[str, Any], b: dict[str, Any], name_a: str, name_b: str) -> str:
    """Side-by-side agreement / accuracy / oracle numbers for two reports."""
    lines = [
        f"{'':6}| {'agrees w/ last':^23} | {'accuracy':^23}",
        f"{'layer':6}| {name_a[:11]:>11} {name_b[:11]:>11} "
        f"| {name_a[:11]:>11} {name_b[:11]:>11}",
    ]
    for ra, rb in zip(a["layers"], b["layers"]):
        lines.append(
            f"  L{ra['layer']:<3} | {_fmt(ra['agreement_with_final'], True):>11} "
            f"{_fmt(rb['agreement_with_final'], True):>11} | "
            f"{_fmt(ra['accuracy'], True):>11} {_fmt(rb['accuracy'], True):>11}"
        )
    lines.append(
        f"oracle avg exit layer: {name_a} {a['oracle']['avg_exit_layer']:.2f}  vs  "
        f"{name_b} {b['oracle']['avg_exit_layer']:.2f}  (lower = more savings possible)"
    )
    return "\n".join(lines)


def build_eval_loader(cfg: Config, tokenizer: Any, batch_size: int | None = None) -> DataLoader:
    """Tokenize only the eval split (skips the train split, unlike build_dataloaders)."""
    _, eval_raw = load_raw_dataset(cfg)
    eval_tok = eval_raw.map(
        lambda ex: tokenize_example(ex, tokenizer, cfg),
        remove_columns=eval_raw.column_names,
        desc="tokenizing eval",
    )
    eval_tok.set_format(type="python")
    return DataLoader(
        eval_tok,
        batch_size=batch_size or cfg.training.eval_batch_size,
        shuffle=False,
        collate_fn=Seq2SeqCollator(tokenizer=tokenizer),
        num_workers=cfg.data.num_workers,
    )


def latest_adapter_dir(cfg: Config) -> Path:
    """Adapter directory of the run's latest end-of-epoch checkpoint."""
    from src.train import ADAPTER_SUBDIR, _epoch_dir, _latest_completed_epoch
    from utils.colab_utils import get_project_paths

    paths = get_project_paths(cfg.run_name, create=False)
    last = _latest_completed_epoch(paths)
    if last is None:
        raise FileNotFoundError(
            f"no checkpoints found under {paths.checkpoints}; train first or pass --adapter"
        )
    return _epoch_dir(paths, last) / ADAPTER_SUBDIR


def main() -> None:
    """CLI entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description="Per-layer early-exit report (Checkpoint 1).")
    parser.add_argument("--config", help="Run config YAML (same one used for training).")
    parser.add_argument("--adapter", help="Adapter dir; default = run's latest checkpoint.")
    parser.add_argument(
        "--no-adapter", action="store_true", help="Analyze the base model without LoRA."
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Cap eval batches.")
    parser.add_argument("--output", help="Where to write the JSON report.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A_JSON", "B_JSON"),
        help="Compare two saved reports.",
    )
    args = parser.parse_args()

    if args.compare:
        a_path, b_path = (Path(p) for p in args.compare)
        a, b = (json.loads(p.read_text(encoding="utf-8")) for p in (a_path, b_path))
        print(format_comparison(a, b, a_path.stem, b_path.stem))
        return
    if not args.config:
        parser.error("--config is required unless --compare is used")

    from transformers import AutoModelForSeq2SeqLM

    from src.model import load_adapter
    from utils.colab_utils import get_project_paths

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg.model.student_name_or_path)

    if args.no_adapter:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.student_name_or_path)
        tag = "no_adapter"
    else:
        adapter = Path(args.adapter) if args.adapter else latest_adapter_dir(cfg)
        print(f"[analysis] loading adapter from {adapter}")
        model = load_adapter(cfg, adapter)
        tag = "layer_report"
    model.to(device)

    report = layer_report(model, build_eval_loader(cfg, tokenizer), device, args.max_batches)
    report["run_name"] = cfg.run_name

    out = (
        Path(args.output)
        if args.output
        else get_project_paths(cfg.run_name).outputs / f"{cfg.run_name}_{tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_report(report, title=f"== {cfg.run_name} ({tag}) =="))
    print(f"[analysis] wrote {out}")


if __name__ == "__main__":
    main()
