# Multilingual Fake News Detector — model_loader.py
"""Shared model and tokenizer loader for the Multilingual Fake News Detector.

This module loads the TensorFlow SavedModel and the bert-base-multilingual-cased
tokenizer exactly once per process. Both the Gradio app and the FastAPI
service import predict() from here so the model is never loaded twice.
"""

import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer

MODEL_PATH = "./saved_model"
TOKENIZER_NAME = "bert-base-multilingual-cased"
LABELS = ["Real", "Fake"]
SUPPORTED_LANGUAGES = ["English", "Urdu", "Spanish"]
MAX_LENGTH = 128

_model = None
_tokenizer = None
_serving_fn = None


def load_model():
    """Load the TensorFlow SavedModel from MODEL_PATH once per process.

    Returns:
        The loaded TensorFlow SavedModel object.

    Raises:
        RuntimeError: If the saved_model folder does not exist.
    """
    global _model, _serving_fn
    if _model is not None:
        return _model

    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(
            f"SavedModel folder not found at '{MODEL_PATH}'. "
            "Place the TensorFlow SavedModel files there before starting."
        )

    _model = tf.saved_model.load(MODEL_PATH)
    _serving_fn = _model.signatures["serving_default"]
    print("Model loaded successfully")
    return _model


def load_tokenizer():
    """Load the bert-base-multilingual-cased tokenizer once per process.

    Returns:
        The HuggingFace AutoTokenizer instance.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def _ensure_loaded():
    """Ensure both the model and tokenizer are loaded before inference."""
    if _model is None or _serving_fn is None:
        load_model()
    if _tokenizer is None:
        load_tokenizer()


def predict(text: str) -> dict:
    """Run inference on a single piece of text and return the prediction.

    Args:
        text: The input text to classify. Must be a non-empty string.

    Returns:
        A dict of the form:
            {
                "prediction": "Real" | "Fake",
                "confidence": float,
                "scores": {"Real": float, "Fake": float},
            }

    Raises:
        ValueError: If text is empty or not a string.
        RuntimeError: If inference fails for any reason.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    _ensure_loaded()

    try:
        encoded = _tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        outputs = _serving_fn(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

        logits_key = next(iter(outputs))
        logits = outputs[logits_key]

        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = LABELS[predicted_index]
        confidence = float(probabilities[predicted_index])

        scores = {label: float(probabilities[i]) for i, label in enumerate(LABELS)}

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "scores": scores,
        }
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Inference failed: {exc}") from exc
