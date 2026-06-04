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

A fake news classifier that labels text as **Real** or **Fake** across six languages: **English**, **Urdu**, **Spanish**, **German**, **Chinese**, and **Korean**. Built on a fine-tuned `bert-base-multilingual-cased` model. You train the model yourself from public datasets plus bundled seed data (see [Training](#training)) — there is no pre-trained checkpoint committed to this repo.

## Supported languages

| Language | Code | Script  |
|----------|------|---------|
| English  | en   | Latin   |
| Urdu     | ur   | Arabic  |
| Spanish  | es   | Latin   |
| German   | de   | Latin   |
| Chinese  | zh   | Han     |
| Korean   | ko   | Hangul  |

## Project structure

```
.
├── app.py                       # Gradio UI (HuggingFace Spaces entry point)
├── api.py                       # FastAPI REST backend
├── model_loader.py              # Shared model + tokenizer loader (load-once)
├── data_prep.py                 # Build the corpus (data/train.csv, data/test.csv)
├── train.py                     # Fine-tune mBERT and export ./saved_model/
├── evaluate.py                  # Offline evaluation CLI
├── notebooks/train_colab.ipynb  # Colab (free GPU) training runner
├── requirements.txt             # Pinned Python dependencies (Python 3.11)
├── Dockerfile                   # Container image for app + API
├── docker-compose.yml           # Two-service stack (Gradio + FastAPI)
├── .dockerignore
├── .gitignore
├── .env.example                 # Optional HF Hub model fallback config
├── deploy.ps1 / deploy.sh       # Push current branch to a HuggingFace Space
├── tests/test_model_loader.py   # Error-path unit tests (skip if TF missing)
├── data/seed/{de,zh,ko}_seed.csv# Hand-authored seed data (German/Chinese/Korean)
├── data/sample_test.csv         # 6-row demo dataset for evaluate.py
├── .github/workflows/ci.yml     # GitHub Actions: syntax + lightweight tests
├── LICENSE                      # MIT
├── saved_model/                 # TensorFlow SavedModel (created by train.py)
└── README.md
```

## Model source

There is **no pre-trained checkpoint in this repo** — you create `./saved_model/` by running [Training](#training) (locally or on Colab). `model_loader.py` resolves the model in this order:

1. **Local** — if `./saved_model/` contains a `saved_model.pb`, it's loaded directly (the normal path after training).
2. **HuggingFace Hub fallback** — only if the folder is empty *and* `HF_MODEL_REPO_ID` is set, `huggingface_hub.snapshot_download()` pulls that repo into `./saved_model/` (add `HF_TOKEN` if it's private).
3. Otherwise startup raises `RuntimeError` with instructions.

For the standard workflow you don't need any environment variables — just train the model. The optional HF Hub fallback is documented in [`.env.example`](.env.example).

## Training

The model is trained from public datasets plus bundled seed data. **Python 3.11 is required** (TensorFlow has no wheels for 3.14). Set up a virtual environment first:

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1. Build the corpus

`data_prep.py` downloads and normalizes the per-language datasets into `data/train.csv` and `data/test.csv` (schema `text, language, label`, where `0 = Real, 1 = Fake`):

| Language | Source |
|---|---|
| English | HF `ErfanMoosaviMonazzah/fake-news-detection-dataset-English` |
| Urdu | HF `community-datasets/urdu_fake_news` |
| Spanish | HF `mariagrandury/fake_news_corpus_spanish` |
| German / Chinese / Korean | bundled seed sets in `data/seed/{de,zh,ko}_seed.csv` |

```bash
python data_prep.py                 # full corpus
python data_prep.py --subset 200    # cap each language to 200 rows (fast smoke test)
```

> German/Chinese/Korean ship as small hand-authored **seed** sets because the large research corpora (FANG-COVID, CHECKED, AI-Hub) are distributed as thousands of per-article files rather than clean single-file downloads. To scale a language up, drop extra CSVs with `text,label` columns into `data/raw/<code>/` (e.g. `data/raw/de/`) and they are merged in automatically.

### 2. Fine-tune + export

`train.py` fine-tunes a multilingual transformer and writes `./saved_model/` with a serving signature that `model_loader.py` consumes directly.

```bash
# Full quality (needs a GPU or lots of RAM) — the default model is bert-base-multilingual-cased
python train.py --csv data/train.csv --epochs 3 --batch-size 16
```

**Training mBERT in TensorFlow needs several GB of free RAM** (the PyTorch→TF weight conversion peaks high). On a low-RAM / no-GPU laptop, either train on Colab (below) or use the smaller distilled model and a subset:

```bash
python train.py --csv data/train.csv --model distilbert-base-multilingual-cased \
  --subset 400 --batch-size 8 --epochs 1
```

`--model` accepts any HF sequence-classification model whose tokenizer matches `model_loader.TOKENIZER_NAME`; the export signature is model-agnostic, so `model_loader.py` needs no changes.

### Train on Colab (free GPU) — recommended for quality

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Google Colab, set the runtime to **GPU**, run all cells, and download the resulting `saved_model.zip`. Unzip it into the repo root so `./saved_model/saved_model.pb` exists, then run the app/API below.

## Run locally

> Requires a trained `./saved_model/` (see [Training](#training)).

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
  "languages": ["English", "Urdu", "Spanish", "German", "Chinese", "Korean"]
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

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%204.35-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-4.7.1-F97316?logo=gradio&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
