# Multilingual Fake News Detector — model_loader.py
"""Shared model and tokenizer loader for the Multilingual Fake News Detector.

This module loads the TensorFlow SavedModel and the bert-base-multilingual-cased
tokenizer exactly once per process. Both the Gradio app and the FastAPI
service import predict() from here so the model is never loaded twice.
"""

import os

import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL_PATH = "./saved_model"
TOKENIZER_NAME = "bert-base-multilingual-cased"
LABELS = ["Real", "Fake"]
SUPPORTED_LANGUAGES = [
    "English",
    "Urdu",
    "Spanish",
    "German",
    "Chinese",
    "Korean",
]
MAX_LENGTH = 128
HF_MODEL_REPO_ENV = "HF_MODEL_REPO_ID"

_model = None
_tokenizer = None
_serving_fn = None


def _has_savedmodel_files(path: str) -> bool:
    """Return True iff `path` is a directory containing SavedModel artefacts.

    A real TensorFlow SavedModel directory always contains `saved_model.pb`
    (or its .pbtxt variant), so we use that as the readiness signal.
    """
    if not os.path.isdir(path):
        return False
    expected = {"saved_model.pb", "saved_model.pbtxt"}
    return any(name in expected for name in os.listdir(path))


def _download_from_hub(repo_id: str) -> None:
    """Download the SavedModel from a HuggingFace Hub repo into MODEL_PATH.

    Args:
        repo_id: The "<owner>/<repo>" identifier of the HF Hub model repo.
    """
    os.makedirs(MODEL_PATH, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=MODEL_PATH)


def load_model():
    """Load the TensorFlow SavedModel from MODEL_PATH once per process.

    Tries the local `./saved_model/` directory first; if that's empty, falls
    back to downloading from the HuggingFace Hub using the repo id named by
    the `HF_MODEL_REPO_ID` env var.

    Returns:
        The loaded TensorFlow SavedModel object.

    Raises:
        RuntimeError: If neither the local SavedModel nor the env var is set.
    """
    global _model, _serving_fn
    if _model is not None:
        return _model

    if not _has_savedmodel_files(MODEL_PATH):
        repo_id = os.environ.get(HF_MODEL_REPO_ENV, "").strip()
        if not repo_id:
            raise RuntimeError(
                f"SavedModel not found at '{MODEL_PATH}' and the "
                f"{HF_MODEL_REPO_ENV} env var is not set. Either place the "
                f"TensorFlow SavedModel files in '{MODEL_PATH}' or set "
                f"{HF_MODEL_REPO_ENV}=<owner>/<repo> to download from the "
                "HuggingFace Hub."
            )
        print(f"Downloading SavedModel from HuggingFace Hub: {repo_id}")
        _download_from_hub(repo_id)
        if not _has_savedmodel_files(MODEL_PATH):
            raise RuntimeError(
                f"Hub download from '{repo_id}' did not produce a valid "
                "SavedModel (saved_model.pb missing)."
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

        # The exported serving signature declares int32 inputs; cast explicitly so
        # the call never fails on tokenizer dtype differences across versions.
        input_ids = tf.cast(encoded["input_ids"], tf.int32)
        attention_mask = tf.cast(encoded["attention_mask"], tf.int32)

        outputs = _serving_fn(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
