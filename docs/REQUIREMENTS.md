# Requirements

## Why this document matters

A clear requirements doc is the contract between what the system *must* do and what it *actually* does. It lets reviewers verify completeness, lets engineers say "no" to scope creep, and gives operators something to test against before declaring the system production-ready.

## Functional requirements

### Core inference

| ID | Requirement |
|---|---|
| FR-1 | The system shall classify a piece of text as `Real` or `Fake`. |
| FR-2 | The system shall support input in English, Urdu, and Spanish. |
| FR-3 | The system shall return per-class confidence scores summing to 1.0 alongside the predicted label. |
| FR-4 | The system shall truncate input to 128 BERT tokens; longer inputs are silently truncated. |
| FR-5 | The system shall reject empty and whitespace-only input with a clear error. |

### Interfaces

| ID | Requirement |
|---|---|
| FR-6 | The system shall expose an interactive Gradio UI at port 7860. |
| FR-7 | The system shall expose a REST API at port 8000 with `GET /`, `POST /predict`, `POST /predict/batch`. |
| FR-8 | The REST API shall validate inputs and return HTTP 422 with a JSON error on invalid input. |
| FR-9 | The REST API shall return HTTP 500 with a descriptive message on inference failure. |
| FR-10 | The REST API shall enable CORS for all origins to allow browser-based clients. |

### Model lifecycle

| ID | Requirement |
|---|---|
| FR-11 | The system shall load the TensorFlow SavedModel exactly once per process. |
| FR-12 | The system shall resolve the SavedModel from `./saved_model/` (local) or from the HuggingFace Hub via `HF_MODEL_REPO_ID`. |
| FR-13 | The system shall raise a clear, actionable error if no model source is configured. |

### Evaluation

| ID | Requirement |
|---|---|
| FR-14 | The system shall provide an offline evaluation CLI accepting a labelled CSV (`text`, `language`, `label`). |
| FR-15 | The evaluation CLI shall report overall accuracy, macro/weighted precision/recall/F1, a confusion matrix, and a per-language breakdown. |
| FR-16 | The evaluation CLI shall write a machine-readable `evaluation_results.json` for downstream tooling. |

### Deployment

| ID | Requirement |
|---|---|
| FR-17 | The system shall ship as a Docker image runnable with no host dependencies beyond Docker itself. |
| FR-18 | The system shall be deployable to HuggingFace Spaces via `app.py` as the entry point. |
| FR-19 | The system shall provide a single-command deploy script for Windows (`deploy.ps1`) and Linux/macOS (`deploy.sh`). |

## Non-functional requirements

### Performance

| ID | Target | Notes |
|---|---|---|
| NFR-1 | Cold start ≤ 30 s | Time from process start to first ready response, on a 4-core CPU with the model already on disk. |
| NFR-2 | Single-prediction latency ≤ 500 ms (CPU) / ≤ 100 ms (GPU) | Excludes network round-trip. |
| NFR-3 | Memory footprint ≤ 2 GB resident | Per process, with the model loaded. |
| NFR-4 | Throughput ≥ 10 req/s per CPU core | At single-request granularity; batching scales further. |

### Reliability

| ID | Target |
|---|---|
| NFR-5 | The system shall not crash on malformed input — all errors must produce a response with a clear message. |
| NFR-6 | The system shall not silently mis-classify on input from unsupported languages — the language dropdown / `language` field is presentational; the model itself still runs and reports its confidence. (Consumers are responsible for treating low confidence as a "don't know.") |
| NFR-7 | Container restarts shall recover without manual intervention (`restart: unless-stopped` set in `docker-compose.yml`). |

### Maintainability

| ID | Target |
|---|---|
| NFR-8 | Every public function shall have a docstring. |
| NFR-9 | All non-stdlib imports shall appear in `requirements.txt` with pinned versions for the spec-listed packages. |
| NFR-10 | The model and tokenizer shall be loaded in exactly one place in the codebase (`model_loader.py`). |
| NFR-11 | CI shall syntax-check every Python module, the Bash deploy script, and the PowerShell deploy script on every push to `main`. |
| NFR-12 | Unit tests shall be runnable without the heavy ML stack installed (`pytest.importorskip` guard). |

### Security & privacy

| ID | Target |
|---|---|
| NFR-13 | Secrets (HF token, future API keys) shall be provided via environment variables, never committed. `.env` is git-ignored; `.env.example` documents the schema. |
| NFR-14 | The `/predict` endpoint shall cap input size at 1000 characters to prevent unbounded memory use from pathological payloads. |
| NFR-15 | The model shall not log user-supplied text in production logs. |

### Portability

| ID | Target |
|---|---|
| NFR-16 | The system shall run on Linux, macOS, and Windows hosts via Docker. |
| NFR-17 | The deploy scripts shall be provided in both Bash (Linux/macOS) and PowerShell (Windows). |
| NFR-18 | The system shall run on Python 3.10 as the reference version. |

## System requirements

### Runtime

- **Python:** 3.10
- **TensorFlow:** 2.13.0
- **Transformers:** 4.35.0
- **huggingface_hub:** 0.19.4
- **Gradio:** 4.7.1
- **FastAPI:** 0.104.1
- **Uvicorn:** 0.24.0
- **Pydantic:** 2.5.0
- **NumPy:** 1.24.3
- **scikit-learn, pandas:** for evaluation only

See [`requirements.txt`](../requirements.txt) for the authoritative list.

### Hardware

- **Minimum:** 2 CPU cores, 4 GB RAM, 2 GB disk for the model. CPU-only inference works.
- **Recommended:** 4+ CPU cores or a single NVIDIA GPU, 8 GB RAM, 5 GB disk. GPU accelerates inference by roughly 5×.
- **HuggingFace Spaces:** the free CPU tier handles this model comfortably for moderate traffic.

### Network

- Outbound to `huggingface.co` is required at startup if `HF_MODEL_REPO_ID` is set (to fetch the model) and at every cold start to fetch the BERT tokenizer (unless cached).

## Acceptance criteria

The system is considered acceptance-ready when:

1. `python -m py_compile model_loader.py app.py api.py evaluate.py` passes locally and in CI.
2. `pytest -q tests/` passes locally and in CI.
3. `docker build -t fake-news-detector .` succeeds without `./saved_model/` present.
4. With the SavedModel in place, `python app.py` boots and serves the Gradio UI on `http://localhost:7860`.
5. `uvicorn api:app --port 8000` boots and `curl http://localhost:8000/` returns the health payload.
6. `python evaluate.py --csv data/sample_test.csv` runs end-to-end and writes `evaluation_results.json`.
7. `./deploy.ps1` / `./deploy.sh` push to a configured HuggingFace Space and the Space serves the Gradio UI.

Steps 1–3 are testable in CI without a model. Steps 4–7 require the SavedModel.
