# Contributing to distill-translate

Thanks for contributing. This project is built so multiple people can work on
different modules in parallel. Each module under `src/` is independently
testable with its own test file under `tests/`.

## 1. Local setup

```bash
git clone <repo-url>
cd distill-translate

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies and the package (editable)
pip install -r requirements.txt
pip install -e .

# Install the git hooks (runs black/isort/ruff before every commit)
pre-commit install
```

## 2. Project layout

| Path                  | Owner area            | What lives here                                  |
| --------------------- | --------------------- | ------------------------------------------------ |
| `src/config.py`       | Config                | Dataclasses + YAML loader + validation           |
| `src/data.py`         | Data                  | OPUS-100 load, tokenize, seq2seq collator        |
| `src/model.py`        | Model                 | Student + LoRA, optional frozen teacher          |
| `src/losses.py`       | Losses                | Label-smoothed CE + word-level KD (KL)           |
| `src/train.py`        | Training              | accelerate loop, epoch checkpoint/resume         |
| `src/evaluate.py`     | Eval                  | Generation + sacreBLEU + predictions export      |
| `utils/colab_utils.py`| Infra                 | Drive mount + path routing                       |
| `notebooks/`          | Demo                  | `01_base_training.ipynb`                          |
| `tests/`              | Everyone              | One `test_<module>.py` per module                |

Because every module has a dedicated test file, two people can safely work on,
say, `data.py` and `losses.py` at the same time without stepping on each other.

## 3. Branching

Never commit directly to `main`. Branch off `main`:

- `feat/<short-topic>`  — new functionality
- `fix/<short-topic>`   — bug fixes
- `test/<short-topic>`  — tests only
- `docs/<short-topic>`  — docs only
- `chore/<short-topic>` — tooling / housekeeping

## 4. Commit messages (Conventional Commits)

Each commit should be a working, tested increment.

```
feat: add word-level KD loss
fix: mask padding tokens in KL
test: cover config validation
docs: document Colab verification gates
chore: scaffold repo and tooling
```

## 5. Running checks locally

```bash
# Format + lint everything (same as the pre-commit hooks)
pre-commit run --all-files

# Run the test suite
pytest
```

All checks must pass before opening a PR.

## 6. Pull requests

1. Push your branch: `git push -u origin feat/<short-topic>`
2. Open a PR into `main` — one PR per task or logical group.
3. CI / reviewers confirm `pre-commit run --all-files` and `pytest` are green.
4. Squash-merge once approved.

## 7. Colab verification gates

Some work must be verified on a real Colab GPU, not just locally:

| Gate | After task | What to verify on Colab                                           |
| ---- | ---------- | ---------------------------------------------------------------- |
| 1    | Task 3     | Drive mounts and `get_project_paths()` resolves Drive paths      |
| 2    | Task 7     | SFT loop + end-of-epoch checkpoint/resume on GPU with `tiny.yaml`|
| 3    | Task 10    | Full acceptance: SFT + distillation + resume + BLEU, all on Drive|

Always verify locally on CPU with a tiny subset **before** a Colab run.

## 8. Secrets

Never commit secrets. `.env`, keys, and credential files are git-ignored. The
`detect-private-key` pre-commit hook is a backstop, not a replacement for care.
