# Skills demonstrated

## Why this document matters

This project doubles as a portfolio artefact. This page makes the technical skills it demonstrates explicit so a reviewer doesn't have to reverse-engineer them from code. Each section names the skill, points to the concrete file or feature in this repo, and notes why it matters.

## Machine learning

| Skill | Where it shows up in this repo | Why it matters |
|---|---|---|
| Multilingual NLP | Three-language classifier built on `bert-base-multilingual-cased` (English / Urdu / Spanish) | Multilingual modelling is harder than monolingual — tokenizer choice, script handling, and per-language evaluation all have to be deliberate. |
| Transformer fine-tuning | TensorFlow SavedModel exported from a fine-tuned mBERT checkpoint | Demonstrates the full fine-tune → export → serve loop, not just inference against a frozen pretrained model. |
| Binary text classification | `LABELS = ["Real", "Fake"]`, softmax + argmax in [`model_loader.py:153`](../model_loader.py#L153) | The most common production NLP shape; clean implementation is table stakes. |
| Per-language evaluation | [`evaluate.py:compute_per_language_metrics`](../evaluate.py) | Reporting only the average accuracy hides language-level regressions. Per-slice metrics are how you catch them. |
| Responsible AI documentation | [`docs/MODEL_CARD.md`](MODEL_CARD.md) | Intended use, limitations, ethical considerations — required for any model touching real users. |

## Software engineering

| Skill | Where it shows up | Why it matters |
|---|---|---|
| Separation of concerns | [`model_loader.py`](../model_loader.py) is the single owner of model + tokenizer; `app.py` / `api.py` / `evaluate.py` are thin wrappers | Prevents duplicate model loads (700 MB × N processes), keeps inference logic in one place. |
| Load-once / process-singleton pattern | Module-level `_model`, `_tokenizer`, `_serving_fn` in [`model_loader.py:23`](../model_loader.py#L23) | Cold-start cost is paid once per process, not per request. |
| Type-safe input validation | Pydantic models in [`api.py`](../api.py): `PredictRequest`, `BatchPredictRequest` with `field_validator` | HTTP 422 with clear errors instead of stack traces. |
| Explicit error handling | No bare `except:` clauses. `ValueError` → 422, `RuntimeError` → 500, both with descriptive messages | Bare excepts hide bugs; explicit ones surface them. |
| Configuration via env | `HF_MODEL_REPO_ID` env var resolved in [`model_loader.py:68`](../model_loader.py#L68) and documented in [`.env.example`](../.env.example) | Twelve-factor config — same image, different envs. |

## DevOps / shipping

| Skill | Where it shows up | Why it matters |
|---|---|---|
| Containerisation | [`Dockerfile`](../Dockerfile) with layer-cached dependency install | Reproducible runtime, faster rebuilds — `pip install` only re-runs when `requirements.txt` changes. |
| Multi-service orchestration | [`docker-compose.yml`](../docker-compose.yml) runs Gradio on `:7860` and FastAPI on `:8000` sharing the same image | Demonstrates running multiple long-lived services against one model with one build. |
| Continuous integration | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs syntax checks + `pytest` on every push | Catches obvious regressions before they reach `main`. |
| Deployment automation | [`deploy.ps1`](../deploy.ps1) / [`deploy.sh`](../deploy.sh) wire up the `hf` remote and push to a HuggingFace Space | One-command ship to production. |
| Image hygiene | [`.dockerignore`](../.dockerignore) keeps `.git/`, caches, IDE folders out of the image | Smaller images, fewer accidental secret leaks. |

## API / interface design

| Skill | Where it shows up | Why it matters |
|---|---|---|
| RESTful design | [`api.py`](../api.py): `GET /` health, `POST /predict`, `POST /predict/batch` | Conventional, idempotent verbs; predictable for clients. |
| Documented contracts | [`docs/API.md`](API.md) and FastAPI's auto-generated `/docs` Swagger UI | Clients can integrate without reading source code. |
| Interactive demo UI | [`app.py`](../app.py) Gradio Blocks layout with examples for all three languages | Lowers the barrier for non-technical evaluators; HF Spaces ready. |
| Batch + single-request APIs | Both `/predict` and `/predict/batch` endpoints | Lets clients amortise HTTP overhead when scoring many texts. |

## Documentation

| Skill | Where it shows up | Why it matters |
|---|---|---|
| Architecture docs | [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) with diagram + design rationale | Onboards new contributors in minutes instead of hours. |
| Requirements engineering | [`docs/REQUIREMENTS.md`](REQUIREMENTS.md) explicit functional + non-functional spec | Forces clarity on what the system must do before debating how. |
| Model card | [`docs/MODEL_CARD.md`](MODEL_CARD.md) | Industry-standard transparency artefact. |
| Deployment runbook | [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) | Production failures aren't the time to invent the deploy process. |
| Contributor guide | [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Lowers the friction for outside contributions. |

## Soft / project skills

- **End-to-end ownership** — data → model → API → deploy → docs, no part outsourced or hand-waved.
- **Production-mindedness** — every choice (Hub fallback, load-once invariant, per-language metrics, CI) was made with "what breaks at 3 AM" in view.
- **Bilingual / trilingual engineering** — comfortable working across Latin and Arabic scripts in the same codebase, including UI examples.
- **Documentation discipline** — the docs are part of the deliverable, not a follow-up.

## Tech-stack snapshot

| Layer | Tool / library |
|---|---|
| Modelling | TensorFlow 2.13, HuggingFace Transformers 4.35, `bert-base-multilingual-cased` |
| Serving | FastAPI 0.104, Uvicorn 0.24, Pydantic 2.5 |
| UI | Gradio 4.7 (Blocks API) |
| Distribution | HuggingFace Hub (`huggingface_hub` 0.19) |
| Evaluation | scikit-learn, pandas |
| Packaging | Docker, docker-compose |
| CI | GitHub Actions |
| Tests | pytest |
| Languages | Python 3.10, Bash, PowerShell |
