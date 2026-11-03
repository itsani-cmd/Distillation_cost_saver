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
