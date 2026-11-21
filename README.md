# distill-translate

Modular, Colab-friendly pipeline that fine-tunes a small Transformer student
(`t5-small`) for **en-de translation** using **LoRA**, with optional
**knowledge distillation** from a larger teacher (`flan-t5-base`).

This repo is **Part 1: the base training loop**. It supports:

- Pure supervised fine-tuning (SFT) with LoRA, and
- Knowledge distillation (word-level KL on soft logits) from a frozen teacher,
- End-of-epoch checkpointing with full resume (model + LoRA + optimizer +
  scheduler + epoch + RNG state),
- BLEU evaluation (sacreBLEU) with saved predictions,
- Google Drive persistence when running on Colab.

The training loop is a **custom `accelerate` loop** (not `Seq2SeqTrainer`) so
that later parts can hook in cleanly:

- **Part 2** — per-layer observation (forward hooks on each encoder/decoder layer)
- **Part 3** — early-exit / layer-skipping (exit heads)

Those are out of scope here; the loop and model module leave hook points for them.

## Quick start (local, CPU, tiny)

```bash
pip install -r requirements.txt
pip install -e .
pytest                                   # sanity-check the modules
python -m src.train --config configs/tiny.yaml
python -m src.evaluate --config configs/tiny.yaml
```

## Quick start (Colab)

Open `notebooks/01_base_training.ipynb` and run all cells. The first cell calls
`setup_colab()` which installs deps, mounts Drive, and routes all checkpoints,
logs, and predictions to Drive.

## Part 3 (in progress): early exit, Milestone 1 — teach every layer to guess

Goal: let the decoder stop after fewer than 6 layers on "easy" words (CALM,
Schuster et al. 2022, `2207.07061v2.pdf`). Milestone 1 only changes
**training**: with `early_exit.enabled: true`, every decoder layer's hidden
state is turned into word predictions through the *shared* output head (no new
parameters) and graded, with later layers weighted more (`i/21`).

Two options are compared; their configs differ in one line:

| Option | Config | Each layer learns from |
|---|---|---|
| A: SFT | `configs/ee_*_sft.yaml` | the correct translation |
| B: SFT + self-distillation | `configs/ee_*_selfdistill.yaml` | the correct translation **+** layer 6's soft prediction (`self_distill_weight: 0.5`) |

```bash
# 1) smoke test (minutes, numbers meaningless)
python -m src.train --config configs/ee_tiny_sft.yaml
python -m src.train --config configs/ee_tiny_selfdistill.yaml

# 2) real runs on a GPU (150k sentence pairs x 2 epochs each)
python -m src.train --config configs/ee_mid_sft.yaml
python -m src.train --config configs/ee_mid_selfdistill.yaml

# 3) Checkpoint 1: per-layer report for each run (+ untrained reference)
python -m src.early_exit.analysis --config configs/ee_mid_sft.yaml --no-adapter
python -m src.early_exit.analysis --config configs/ee_mid_sft.yaml
python -m src.early_exit.analysis --config configs/ee_mid_selfdistill.yaml
python -m src.early_exit.analysis --compare <outputs>/ee_mid_sft_layer_report.json \
                                            <outputs>/ee_mid_selfdistill_layer_report.json
```

Training logs now include `ce_L1 … ce_L6` (each layer's loss) and `sd`
(self-distillation). The report shows, per layer, accuracy, **agreement with
the last layer** (the key number), and confidence, plus an oracle estimate of
the best-case average exit layer. If early layers stay weak, set
`early_exit.train_final_norm: true` so the shared final norm also trains.

Tests: `pytest` (fast) and `RUN_HEAVY_TESTS=1 pytest tests/test_early_exit_model.py`.

## Configuration

Configs are YAML, validated into dataclasses (`src/config.py`). Start from
`configs/tiny.yaml` (small subset, 1-2 epochs) and scale up with
`configs/base.yaml`.

## Caveat on teacher quality (en-de)

`t5-small` / `flan-t5-base` are English-centric and **not** strong en-de
translators out of the box. Distillation quality is bounded by teacher quality,
so in Part 1 **SFT is the primary quality driver** and KD is wired up and
validated as a mechanism. For real gains the teacher should be fine-tuned on
en-de first. This is flagged in `configs/tiny.yaml` and the notebook.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branching, commit style, and
the three Colab verification gates.
