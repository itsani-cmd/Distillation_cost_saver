"""Part 2 experiment: how many layers can we skip, and what does it cost?

Translates the eval sentences with the early-exit translator under every
stopping rule / setting and reports, for each run:

* ``bleu``          -- quality vs the correct translations (sacreBLEU);
* ``bleu_vs_full``  -- how close we stayed to the full 6-layer model's output
  (100 = identical translations);
* ``avg_layers``    -- average decoder layers used per word (6.0 = full model);
* ``layer_reduction`` -- ``L / avg_layers`` (2.0 = half the decoder layers);
* ``exit_L1..exit_L6`` -- fraction of words that stopped at each layer.

Runs: the full model (static, layer L = the baseline), static exits at
layers ``1..L-1``, the softmax rule over a threshold sweep, and the oracle.

All runs use **greedy** decoding, so the baseline BLEU is lower than the
beam-search BLEU printed at the end of training; compare rows with each other.

Outputs (in the run's outputs dir, ``early_exit_eval/``): ``results.csv``,
``results.json`` (rows + example translations) and ``tradeoff.png``.

CLI::

    python -m src.early_exit.run_eval --config configs/ee_mid_selfdistill.yaml
    python -m src.early_exit.run_eval --config ... --num-sentences 100   # quicker
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.config import Config, load_config
from src.data import build_tokenizer, load_raw_dataset
from src.early_exit.decode import early_exit_greedy
from src.early_exit.heads import num_decoder_layers
from src.evaluate import compute_bleu

DEFAULT_THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
NUM_EXAMPLES_SAVED = 5


@dataclass
class EvalSentence:
    """One eval example: tokenized source plus raw strings."""

    input_ids: torch.Tensor  # (1, src_len)
    attention_mask: torch.Tensor  # (1, src_len)
    source: str
    reference: str


@dataclass
class RunSpec:
    """One early-exit setting to evaluate."""

    name: str
    rule: str
    threshold: float = 1.0
    static_layer: int = 0


def load_eval_sentences(cfg: Config, tokenizer: Any, num_sentences: int) -> list[EvalSentence]:
    """First ``num_sentences`` eval pairs, tokenized like training (with the task prefix)."""
    _, eval_raw = load_raw_dataset(cfg)
    d = cfg.data
    out = []
    for i in range(min(num_sentences, len(eval_raw))):
        pair = eval_raw[i]["translation"]
        enc = tokenizer(
            cfg.model.prefix + pair[d.source_lang],
            max_length=d.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        out.append(
            EvalSentence(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                source=pair[d.source_lang],
                reference=pair[d.target_lang],
            )
        )
    return out


def build_run_specs(
    num_layers: int,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    static_layers: tuple[int, ...] | None = None,
    oracle: bool = True,
) -> list[RunSpec]:
    """The sweep: full baseline first, then static, softmax and oracle runs."""
    specs = [RunSpec(name="full", rule="static", static_layer=num_layers)]
    for k in static_layers if static_layers is not None else range(1, num_layers):
        if 1 <= k < num_layers:
            specs.append(RunSpec(name=f"static_L{k}", rule="static", static_layer=k))
    for lam in thresholds:
        specs.append(RunSpec(name=f"softmax_{lam:g}", rule="softmax", threshold=lam))
    if oracle:
        specs.append(RunSpec(name="oracle", rule="oracle"))
    return specs


def run_one(
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: list[EvalSentence],
    spec: RunSpec,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[list[str], list[int], float]:
    """Translate every sentence with one setting.

    Returns:
        ``(hypotheses, all_exit_layers, seconds_per_sentence)``.
    """
    hyps: list[str] = []
    exits: list[int] = []
    start = time.perf_counter()
    for s in sentences:
        out = early_exit_greedy(
            model,
            s.input_ids.to(device),
            s.attention_mask.to(device),
            rule=spec.rule,
            threshold=spec.threshold,
            static_layer=spec.static_layer,
            max_new_tokens=max_new_tokens,
        )
        hyps.append(tokenizer.decode(out.token_ids, skip_special_tokens=True))
        exits.extend(out.exit_layers)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return hyps, exits, (time.perf_counter() - start) / max(len(sentences), 1)


def summarize(
    spec: RunSpec,
    hyps: list[str],
    exits: list[int],
    sec_per_sentence: float,
    references: list[str],
    full_hyps: list[str],
    num_layers: int,
) -> dict[str, Any]:
    """One results row for a run."""
    avg = sum(exits) / max(len(exits), 1)
    row: dict[str, Any] = {
        "name": spec.name,
        "rule": spec.rule,
        "threshold": spec.threshold if spec.rule == "softmax" else None,
        "static_layer": spec.static_layer if spec.rule == "static" else None,
        "bleu": compute_bleu(hyps, references),
        "bleu_vs_full": compute_bleu(hyps, full_hyps),
        "avg_layers": avg,
        "layer_reduction": num_layers / avg if avg else None,
        "num_sentences": len(hyps),
        "num_tokens": len(exits),
        "sec_per_sentence": sec_per_sentence,
    }
    for layer in range(1, num_layers + 1):
        row[f"exit_L{layer}"] = exits.count(layer) / max(len(exits), 1)
    return row


def run_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: list[EvalSentence],
    specs: list[RunSpec],
    device: torch.device,
    max_new_tokens: int,
    log: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Run every spec. ``specs[0]`` must be the full-model baseline.

    Returns:
        ``(rows, hypotheses_by_run_name)``.
    """
    if not specs or specs[0].name != "full":
        raise ValueError("specs[0] must be the 'full' baseline (see build_run_specs)")
    num_layers = num_decoder_layers(model)
    references = [s.reference for s in sentences]
    rows: list[dict[str, Any]] = []
    all_hyps: dict[str, list[str]] = {}
    full_hyps: list[str] = []
    for spec in specs:
        hyps, exits, sps = run_one(model, tokenizer, sentences, spec, device, max_new_tokens)
        if spec.name == "full":
            full_hyps = hyps
        row = summarize(spec, hyps, exits, sps, references, full_hyps, num_layers)
        rows.append(row)
        all_hyps[spec.name] = hyps
        if log:
            print(
                f"[run_eval] {spec.name:>12}: BLEU={row['bleu']:6.2f}  "
                f"vs_full={row['bleu_vs_full']:6.2f}  avg_layers={row['avg_layers']:.2f}  "
                f"({row['layer_reduction'] or 0:.2f}x fewer)"
            )
    return rows, all_hyps


def format_table(rows: list[dict[str, Any]]) -> str:
    """Readable results table."""
    header = f"{'run':>12} | {'BLEU':>6} | {'vs full':>7} | {'avg layers':>10} | {'reduction':>9}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['name']:>12} | {r['bleu']:6.2f} | {r['bleu_vs_full']:7.2f} | "
            f"{r['avg_layers']:10.2f} | {r['layer_reduction'] or 0:8.2f}x"
        )
    return "\n".join(lines)


def save_results(
    out_dir: Path,
    rows: list[dict[str, Any]],
    all_hyps: dict[str, list[str]],
    sentences: list[EvalSentence],
    meta: dict[str, Any],
) -> dict[str, Path]:
    """Write ``results.csv``, ``results.json`` and (if matplotlib is available) ``tradeoff.png``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out_dir / "results.csv", "json": out_dir / "results.json"}

    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    examples = [
        {
            "source": s.source,
            "reference": s.reference,
            **{name: hyps[i] for name, hyps in all_hyps.items()},
        }
        for i, s in enumerate(sentences[:NUM_EXAMPLES_SAVED])
    ]
    paths["json"].write_text(
        json.dumps(
            {"meta": meta, "rows": rows, "examples": examples}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    chart = plot_tradeoff(rows, out_dir / "tradeoff.png")
    if chart is not None:
        paths["png"] = chart
    return paths


def plot_tradeoff(rows: list[dict[str, Any]], path: Path) -> Path | None:
    """Quality vs average layers chart; returns ``None`` if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[run_eval] matplotlib not installed; skipping chart")
        return None

    def pick(rule_rows: list[dict[str, Any]], key: str) -> list[float]:
        return [r[key] for r in rule_rows]

    full = [r for r in rows if r["name"] == "full"]
    static = sorted(
        [r for r in rows if r["rule"] == "static"], key=lambda r: r["avg_layers"]
    )  # includes full (layer L) as the right end of the static line
    soft = sorted([r for r in rows if r["rule"] == "softmax"], key=lambda r: r["avg_layers"])
    oracle = [r for r in rows if r["rule"] == "oracle"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in (
        (axes[0], "bleu", "BLEU vs correct translations"),
        (axes[1], "bleu_vs_full", "BLEU vs full model's output"),
    ):
        if static:
            ax.plot(
                pick(static, "avg_layers"),
                pick(static, key),
                "s--",
                label="static (stop at layer k)",
            )
        if soft:
            ax.plot(
                pick(soft, "avg_layers"), pick(soft, key), "o-", label="softmax gap (threshold λ)"
            )
            for r in soft:
                ax.annotate(
                    f"{r['threshold']:g}",
                    (r["avg_layers"], r[key]),
                    fontsize=7,
                    textcoords="offset points",
                    xytext=(3, 3),
                )
        if oracle:
            ax.plot(
                pick(oracle, "avg_layers"),
                pick(oracle, key),
                "*",
                markersize=14,
                label="oracle (upper bound)",
            )
        if full:
            ax.axhline(full[0][key], color="gray", linewidth=0.8, linestyle=":", label="full model")
        ax.set_xlabel("average decoder layers per word (lower = less work)")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    """CLI entry point (see module docstring)."""
    parser = argparse.ArgumentParser(description="Early-exit speed/quality sweep (Part 2).")
    parser.add_argument("--config", required=True, help="Run config YAML used for training.")
    parser.add_argument("--adapter", help="Adapter dir; default = run's latest checkpoint.")
    parser.add_argument("--no-adapter", action="store_true", help="Use the untrained base model.")
    parser.add_argument(
        "--num-sentences", type=int, default=300, help="Eval sentences to translate."
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Softmax λ values.",
    )
    parser.add_argument(
        "--static-layers", type=int, nargs="*", default=None, help="Static exit layers."
    )
    parser.add_argument("--no-oracle", action="store_true", help="Skip the oracle run.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Default: max_target_length - 1."
    )
    parser.add_argument("--output-dir", help="Default: <run outputs>/early_exit_eval[_no_adapter].")
    args = parser.parse_args()

    from transformers import AutoModelForSeq2SeqLM

    from src.early_exit.analysis import latest_adapter_dir
    from src.model import load_adapter
    from utils.colab_utils import get_project_paths

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg.model.student_name_or_path)

    if args.no_adapter:
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.student_name_or_path)
        adapter = None
    else:
        adapter = Path(args.adapter) if args.adapter else latest_adapter_dir(cfg)
        print(f"[run_eval] loading adapter from {adapter}")
        model = load_adapter(cfg, adapter)
    model.to(device).eval()

    num_layers = num_decoder_layers(model)
    specs = build_run_specs(
        num_layers,
        thresholds=tuple(args.thresholds),
        static_layers=tuple(args.static_layers) if args.static_layers is not None else None,
        oracle=not args.no_oracle,
    )
    sentences = load_eval_sentences(cfg, tokenizer, args.num_sentences)
    max_new = args.max_new_tokens or cfg.data.max_target_length - 1
    print(f"[run_eval] {len(sentences)} sentences x {len(specs)} runs on {device}")

    rows, all_hyps = run_sweep(model, tokenizer, sentences, specs, device, max_new)

    suffix = "_no_adapter" if args.no_adapter else ""
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else get_project_paths(cfg.run_name).outputs / f"early_exit_eval{suffix}"
    )
    meta = {
        "run_name": cfg.run_name,
        "adapter": str(adapter) if adapter else None,
        "num_sentences": len(sentences),
        "num_layers": num_layers,
        "decoding": "greedy, batch size 1",
    }
    paths = save_results(out_dir, rows, all_hyps, sentences, meta)
    print()
    print(format_table(rows))
    for kind, p in paths.items():
        print(f"[run_eval] wrote {kind}: {p}")


if __name__ == "__main__":
    main()
