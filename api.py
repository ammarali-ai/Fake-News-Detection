# Multilingual Fake News Detector — api.py
"""FastAPI backend for the Multilingual Fake News Detector.

Reuses the shared model_loader so the TensorFlow SavedModel and tokenizer
are loaded exactly once per process at FastAPI startup.
"""

from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from model_loader import (
    SUPPORTED_LANGUAGES,
    load_model,
    load_tokenizer,
    predict,
)

app = FastAPI(
    title="Multilingual Fake News Detector API",
    description="REST API for classifying news text as Real or Fake in English, Urdu, and Spanish.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Request body for single-text prediction."""

    text: str = Field(..., min_length=1, max_length=1000)
    language: str = Field(...)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        """Reject strings that are non-empty but contain only whitespace."""
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, value: str) -> str:
        """Reject any language outside the supported set."""
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {SUPPORTED_LANGUAGES}, got '{value}'"
            )
        return value


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction."""

    texts: List[str] = Field(..., min_length=1)
    language: str = Field(...)

    @field_validator("texts")
    @classmethod
    def texts_must_be_valid(cls, value: List[str]) -> List[str]:
        """Ensure every text in the batch is non-empty and within length limits."""
        for i, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string")
            if len(item) > 1000:
                raise ValueError(f"texts[{i}] exceeds the 1000-character limit")
        return value

    @field_validator("language")
    @classmethod
    def language_must_be_supported(cls, value: str) -> str:
        """Reject any language outside the supported set."""
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be one of {SUPPORTED_LANGUAGES}, got '{value}'"
            )
        return value


@app.on_event("startup")
def _startup_load() -> None:
    """Load the model and tokenizer once when the FastAPI process boots."""
    load_model()
    load_tokenizer()


@app.get("/")
def health_check() -> dict:
    """Return a simple health-check payload describing the service."""
    return {
        "status": "ok",
        "model": "multilingual-fake-news-detector",
        "languages": SUPPORTED_LANGUAGES,
    }


@app.post("/predict")
def predict_single(payload: PredictRequest) -> dict:
    """Classify a single piece of text as Real or Fake."""
    try:
        result = predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "scores": result["scores"],
        "language": payload.language,
    }


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> dict:
    """Classify a batch of texts and return one prediction per item."""
    results = []
    for text in payload.texts:
        try:
            result = predict(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        results.append(
            {
                "text": text,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "scores": result["scores"],
                "language": payload.language,
            }
        )

    return {"language": payload.language, "predictions": results}
