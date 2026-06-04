# Development guide

## Why this document matters

Getting a new contributor productive on day one shouldn't take half a day of trial-and-error. This page covers the local setup, the test loop, and the conventions to follow when changing code.

## Prerequisites

- **Git** ≥ 2.30
- **Python** 3.11 (reference version — TensorFlow has no wheels for 3.14)
- **Docker** (optional, only for the container workflow)
- **HuggingFace account** (optional, only for deploy + Hub fallback)

## Initial setup

```bash
git clone https://github.com/ammarali-ai/Fake-News-Detection.git
cd Fake-News-Detection

# Virtual env (recommended)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install runtime + dev deps
pip install -r requirements.txt
pip install pytest

# Sanity check
python -m py_compile model_loader.py app.py api.py evaluate.py data_prep.py train.py
pytest -q tests/
```

The `pytest` run skips model-dependent tests when the heavy ML stack isn't installed, but it's the same command CI runs, so anything that passes locally should pass on CI.

## Project layout

```
.
├── app.py                       # Gradio UI
├── api.py                       # FastAPI REST service
├── model_loader.py              # Shared model + tokenizer loader (single owner)
├── evaluate.py                  # Offline evaluation CLI
├── requirements.txt
├── Dockerfile / docker-compose.yml / .dockerignore
├── deploy.ps1 / deploy.sh
├── tests/
│   └── test_model_loader.py
├── data/
│   └── sample_test.csv          # 6-row demo dataset
├── docs/
│   ├── README.md                # Doc index
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   ├── REQUIREMENTS.md
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── DEVELOPMENT.md           # ← you are here
│   └── SKILLS.md
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
```

## Running the apps

### Gradio (interactive UI)
```bash
python app.py
# → http://localhost:7860
```

### FastAPI (REST)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs   (Swagger)
```

### Both together via docker-compose
```bash
docker compose up --build
```

### Offline evaluation
```bash
python evaluate.py --csv data/sample_test.csv
```

All four require a SavedModel — either committed at `./saved_model/` or pulled from the Hub via `HF_MODEL_REPO_ID`. See [DEPLOYMENT.md](DEPLOYMENT.md#model-source-on-spaces).

## Tests

### Run the suite
```bash
pytest -q tests/
```

### What's covered
- `_has_savedmodel_files()` happy/missing/empty paths.
- `load_model()` missing-both-sources error path.
- `predict()` input-validation paths (empty string, non-string).
- Public constants (`LABELS`, `SUPPORTED_LANGUAGES`, `MAX_LENGTH`, `HF_MODEL_REPO_ENV`).

### What's not covered
- End-to-end inference. That requires the actual SavedModel and ~700 MB of dependencies; verify it manually before merging anything that touches `model_loader.predict()`.

### Adding tests
Put new tests in `tests/test_<module>.py`. Use `pytest.importorskip` for any test that needs the heavy ML stack so contributors without TF can still run the suite.

## Coding conventions

These are enforced by review, not by formatter (yet).

- **One source of truth for the model.** Only `model_loader.py` calls `tf.saved_model.load()` or `AutoTokenizer.from_pretrained()`. If you find yourself needing a second load, refactor through `model_loader` instead.
- **Docstrings on every public function.** Brief, focused on contract not internals.
- **No bare `except:` clauses.** Catch what you can handle, re-raise the rest with `raise ... from exc`.
- **String constants at module top.** Labels, supported languages, env var names — not inline in functions.
- **Pin pinned versions; don't loosen the spec list.** `pandas` and `scikit-learn` are intentionally unpinned because they're evaluation-only; everything else in `requirements.txt` is pinned for a reason.
- **No comments that restate the code.** Comments should explain *why* a non-obvious choice was made.

## Pre-commit checklist

Before opening a PR:

```bash
python -m py_compile model_loader.py app.py api.py evaluate.py
pytest -q tests/
docker build -t fake-news-detector .
```

If your change touches inference or input handling, also do a manual end-to-end:

```bash
python app.py
# Paste a known-fake and a known-real example, confirm the labels are sensible.
```

## Common tasks

### Adding a new language
1. Add the label to `SUPPORTED_LANGUAGES` in `model_loader.py`.
2. Add an example to `EXAMPLES` in `app.py`.
3. The Pydantic validators in `api.py` read from `SUPPORTED_LANGUAGES`, so they'll pick it up automatically.
4. **Most important:** retrain or fine-tune on data in that language before claiming support. The mBERT tokenizer handles ~100 languages, but the classification head needs training signal.
5. Add representative rows to `data/sample_test.csv`.
6. Update `docs/MODEL_CARD.md` performance section.

### Bumping a dependency
1. Edit `requirements.txt`.
2. `pip install -r requirements.txt --upgrade`.
3. Run tests and a manual end-to-end.
4. If you bump `fastapi`, double-check that `@app.on_event("startup")` still works or migrate to a lifespan handler.
5. If you bump `transformers`, verify the tokenizer still loads (`AutoTokenizer.from_pretrained("bert-base-multilingual-cased")`).

### Changing the model
1. Drop the new SavedModel into `./saved_model/` (or update `HF_MODEL_REPO_ID`).
2. Verify the new model's `serving_default` signature accepts `input_ids` and `attention_mask`. If it accepts different names, update [`model_loader.py:146-148`](../model_loader.py#L146).
3. Verify the head still outputs binary logits in `LABELS` order (`Real`, `Fake`). If the order is reversed, swap `LABELS`.
4. Update `docs/MODEL_CARD.md` with new metrics.
5. Re-run `evaluate.py` on your test split.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: SavedModel not found` | Empty `./saved_model/` and no `HF_MODEL_REPO_ID` | Set one of them. See `.env.example`. |
| `KeyError: 'serving_default'` | Model was saved without the default serving signature | Re-export the model: `model.save("./saved_model", signatures={"serving_default": model.serving_fn})` |
| Out-of-memory on import | Trying to load TF + transformers on a < 4 GB box | Use a bigger machine or run via Docker on a host with more RAM. |
| `git push hf main` fails with "fetch first" | Remote has commits you don't have | `git pull hf main --allow-unrelated-histories` then push, or use `--force` if you're confident your local is authoritative. |
| `pytest` collects tests but they all skip | Heavy ML stack not installed | `pip install -r requirements.txt`. |

## Where to ask for help

- GitHub Issues on the repo for reproducible bugs.
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for PR conventions.
