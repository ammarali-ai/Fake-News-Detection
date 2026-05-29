# Changelog

All notable changes to this project are documented here. Follows [Keep a Changelog](https://keepachangelog.com/) loosely and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Comprehensive `docs/` folder: architecture, model card, requirements, API reference, deployment guide, development guide, skills.
- `CONTRIBUTING.md` and `CHANGELOG.md`.
- `LICENSE` (MIT).
- `.env.example` documenting `HF_MODEL_REPO_ID` and optional `HF_TOKEN`.
- `tests/test_model_loader.py` covering error paths, constants, and the SavedModel detector. Guarded with `pytest.importorskip` so it runs without the heavy ML stack.
- `data/sample_test.csv` — 6-row demo dataset (2 per language) for `evaluate.py`.
- `.github/workflows/ci.yml` — syntax checks (`py_compile`, `bash -n`, PowerShell tokenizer) + `pytest` on every push and PR.

### Changed
- Top-level `README.md` updated with new project-structure tree, a documentation index pointing to `docs/`, a test-running section, and a license note.
- `deploy.sh` marked executable in the git index (mode `100755`).

## [0.2.0] — 2026-05-29

### Added
- `model_loader.py` HuggingFace Hub fallback: when `./saved_model/` is empty, `load_model()` downloads via `huggingface_hub.snapshot_download` using the `HF_MODEL_REPO_ID` env var. Raises a clear error naming both options when neither is configured.
- `_has_savedmodel_files()` helper that detects a real SavedModel directory by checking for `saved_model.pb`.
- `.dockerignore` keeping `.git/`, caches, virtualenvs, IDE folders, and logs out of the image.
- `deploy.ps1` (Windows) and `deploy.sh` (Linux/macOS) — single-command deploy to a HuggingFace Space. Both verify `huggingface-cli` login and idempotently wire up the `hf` remote.

### Changed
- `Dockerfile` no longer hard-fails when `./saved_model/` is absent locally — removed the redundant explicit `COPY saved_model/`. The `COPY . .` covers it when committed; the Hub fallback covers the empty case at runtime.
- `docker-compose.yml` dropped the unused top-level `models:` volume and its `/models` mount. Added `HF_MODEL_REPO_ID` env passthrough on both services.
- `requirements.txt` adds `huggingface_hub==0.19.4`.
- README expanded with a "Model source" section explaining the resolution order, and a "Deploy to HuggingFace Spaces" section documenting the deploy scripts.

## [0.1.0] — 2026-05-29

### Added
- Initial production scaffold.
- `model_loader.py` — shared TensorFlow SavedModel + `bert-base-multilingual-cased` tokenizer loader with module-level singletons. Public functions: `load_model()`, `load_tokenizer()`, `predict(text)`.
- `app.py` — Gradio Blocks UI for English / Urdu / Spanish input. Loads the model at module level for HuggingFace Spaces.
- `api.py` — FastAPI service with `GET /` health check, `POST /predict`, `POST /predict/batch`. Pydantic validation, CORS, startup hook that warms `model_loader`.
- `evaluate.py` — CLI that scores a labelled CSV (`text`, `language`, `label`) and writes `evaluation_results.json` with overall + per-language metrics and a confusion matrix.
- `requirements.txt` with pinned versions per spec, plus pandas / scikit-learn for evaluation.
- `Dockerfile` (Python 3.10 slim, layer-cached deps).
- `docker-compose.yml` — two-service stack on ports 7860 (Gradio) and 8000 (FastAPI).
- `README.md` with HuggingFace Spaces frontmatter, project structure, run/deploy/evaluation instructions, API examples, and tech-stack badges.
- `.gitignore` standard Python ignores; `saved_model/` explicitly kept tracked for HF Spaces deployment.

[Unreleased]: https://github.com/ammarali-ai/Fake-News-Detection/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ammarali-ai/Fake-News-Detection/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ammarali-ai/Fake-News-Detection/releases/tag/v0.1.0
