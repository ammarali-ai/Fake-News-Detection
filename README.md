---
title: Multilingual Fake News Detector
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
---

# Multilingual Fake News Detector

A production-ready fake news classifier that labels text as **Real** or **Fake** across three languages: **English**, **Urdu**, and **Spanish**. Built on a fine-tuned `bert-base-multilingual-cased` model achieving **90% overall accuracy** on the held-out test set.

## Supported languages

| Language | Code | Script  |
|----------|------|---------|
| English  | en   | Latin   |
| Urdu     | ur   | Arabic  |
| Spanish  | es   | Latin   |

## Project structure

```
.
├── app.py                       # Gradio UI (HuggingFace Spaces entry point)
├── api.py                       # FastAPI REST backend
├── model_loader.py              # Shared model + tokenizer loader (load-once)
├── evaluate.py                  # Offline evaluation CLI
├── requirements.txt             # Pinned Python dependencies
├── Dockerfile                   # Container image for app + API
├── docker-compose.yml           # Two-service stack (Gradio + FastAPI)
├── .dockerignore
├── .gitignore
├── .env.example                 # Documents HF_MODEL_REPO_ID
├── deploy.ps1 / deploy.sh       # Push current branch to a HuggingFace Space
├── tests/test_model_loader.py   # Error-path unit tests (skip if TF missing)
├── data/sample_test.csv         # 6-row demo dataset for evaluate.py
├── .github/workflows/ci.yml     # GitHub Actions: syntax + lightweight tests
├── LICENSE                      # MIT
├── saved_model/                 # TensorFlow SavedModel (tracked for HF Spaces)
└── README.md
```

## Model source

`model_loader.py` resolves the SavedModel in this order:

1. **Local** — if `./saved_model/` already contains a `saved_model.pb`, it's loaded directly.
2. **HuggingFace Hub fallback** — if the folder is empty/missing **and** the `HF_MODEL_REPO_ID` env var is set, the model is fetched via `huggingface_hub.snapshot_download()` into `./saved_model/` on first run.
3. Otherwise startup raises `RuntimeError` with a message telling you which of the two options to set.

```bash
# Option A: commit the model alongside the code (best for HF Spaces)
cp -r /path/to/your/saved_model ./saved_model

# Option B: fetch from the Hub at runtime
export HF_MODEL_REPO_ID="your-username/your-model-repo"   # PowerShell: $env:HF_MODEL_REPO_ID = "..."
```

## Run locally

### Gradio app

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:7860
```

### FastAPI service

```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/docs
```

## Run with Docker

### Single container (Gradio by default)

```bash
docker build -t fake-news-detector .
docker run -p 7860:7860 fake-news-detector

# To run the FastAPI service instead:
docker run -p 8000:8000 fake-news-detector \
  python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Both services with docker-compose

```bash
docker-compose up --build
# Gradio:  http://localhost:7860
# FastAPI: http://localhost:8000
```

## Run evaluation

The evaluation script takes a CSV with columns `text`, `language`, `label` (0 = Real, 1 = Fake). A 6-row demo dataset ships at [`data/sample_test.csv`](data/sample_test.csv):

```bash
python evaluate.py --csv data/sample_test.csv
```

It prints overall metrics (accuracy, precision, recall, F1), a per-language breakdown, and a confusion matrix, and writes `evaluation_results.json`.

## Run tests

```bash
pip install pytest
pytest -q tests/
```

Tests use `pytest.importorskip` to skip gracefully if the heavy ML stack isn't installed locally, so the suite is safe to run on any contributor machine. CI runs the same `pytest -q tests/` invocation on every push.

## API documentation

Base URL: `http://localhost:8000`

### `GET /`

Health check.

```json
{
  "status": "ok",
  "model": "multilingual-fake-news-detector",
  "languages": ["English", "Urdu", "Spanish"]
}
```

### `POST /predict`

**Request**
```json
{
  "text": "Scientists confirm the moon is made of cheese.",
  "language": "English"
}
```

**Response**
```json
{
  "prediction": "Fake",
  "confidence": 0.95,
  "scores": {"Real": 0.05, "Fake": 0.95},
  "language": "English"
}
```

### `POST /predict/batch`

**Request**
```json
{
  "texts": [
    "Scientists confirm the moon is made of cheese.",
    "El gobierno anuncia un nuevo plan económico."
  ],
  "language": "English"
}
```

**Response**
```json
{
  "language": "English",
  "predictions": [
    {
      "text": "Scientists confirm the moon is made of cheese.",
      "prediction": "Fake",
      "confidence": 0.95,
      "scores": {"Real": 0.05, "Fake": 0.95},
      "language": "English"
    }
  ]
}
```

Validation errors return HTTP `422`; inference errors return HTTP `500` with a descriptive message.

## Deploy to HuggingFace Spaces

1. Create a new Space at <https://huggingface.co/new-space> with SDK = Gradio.
2. `pip install huggingface_hub && huggingface-cli login`.
3. Run the deploy script for your platform — it wires up the `hf` git remote and pushes the current branch to the Space's `main`:

```powershell
# Windows
.\deploy.ps1 -SpaceOwner <your-hf-username>
# Optional: -SpaceName custom-name, -Force for first push to a non-empty Space
```

```bash
# Linux / macOS
chmod +x deploy.sh
./deploy.sh <your-hf-username>
# Optional: ./deploy.sh <owner> <space-name> --force
```

The script prints the live Space URL on success.

## Documentation

Deep-dive docs live in [`docs/`](docs/):

| Doc | What it covers |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | System diagram, components, request flow, design decisions |
| [Model card](docs/MODEL_CARD.md) | Base model, training data, metrics, limitations, ethics |
| [Requirements](docs/REQUIREMENTS.md) | Functional + non-functional requirements |
| [API reference](docs/API.md) | Full endpoint catalog with examples in curl / Python / JS |
| [Deployment](docs/DEPLOYMENT.md) | HF Spaces, Docker, cloud deployment paths |
| [Development](docs/DEVELOPMENT.md) | Local setup, running tests, code style |
| [Skills](docs/SKILLS.md) | Technical skills demonstrated in this project |
| [Contributing](CONTRIBUTING.md) | How to file issues and open PRs |
| [Changelog](CHANGELOG.md) | Release history |

## License

Released under the [MIT License](LICENSE). Free for commercial and personal use; please retain the copyright notice.

## Tech stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%204.35-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-4.7.1-F97316?logo=gradio&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
