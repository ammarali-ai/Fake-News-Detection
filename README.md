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
├── app.py                  # Gradio UI (HuggingFace Spaces entry point)
├── api.py                  # FastAPI REST backend
├── model_loader.py         # Shared model + tokenizer loader (load-once)
├── evaluate.py             # Offline evaluation CLI
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Container image for app + API
├── docker-compose.yml      # Two-service stack (Gradio + FastAPI)
├── .gitignore
├── saved_model/            # TensorFlow SavedModel (tracked for HF Spaces)
└── README.md
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

The evaluation script takes a CSV with columns `text`, `language`, `label` (0 = Real, 1 = Fake).

```bash
python evaluate.py --csv data/test.csv
```

It prints overall metrics (accuracy, precision, recall, F1), a per-language breakdown, and a confusion matrix, and writes `evaluation_results.json`.

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

## Tech stack

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%204.35-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-4.7.1-F97316?logo=gradio&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
