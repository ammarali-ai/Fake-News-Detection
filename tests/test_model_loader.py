# Multilingual Fake News Detector — tests/test_model_loader.py
"""Lightweight tests for model_loader error paths.

These tests focus on logic that doesn't require the actual TensorFlow
SavedModel: the missing-both error path, constant exposure, and the
SavedModel-files detector. Tests requiring real inference are out of
scope here — run them manually with the model in place.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Skip the whole module if the heavy ML deps aren't installed, so the
# test suite still runs cleanly on contributor machines without them.
tf = pytest.importorskip("tensorflow")
pytest.importorskip("transformers")
pytest.importorskip("huggingface_hub")

import model_loader  # noqa: E402 — imported after dep check


def _reset_module_state():
    """Wipe cached model / tokenizer references on the module under test."""
    model_loader._model = None
    model_loader._tokenizer = None
    model_loader._serving_fn = None


def test_constants_are_exposed():
    """Public constants must match the project spec."""
    assert model_loader.LABELS == ["Real", "Fake"]
    assert model_loader.SUPPORTED_LANGUAGES == ["English", "Urdu", "Spanish"]
    assert model_loader.MAX_LENGTH == 128
    assert model_loader.HF_MODEL_REPO_ENV == "HF_MODEL_REPO_ID"


def test_has_savedmodel_files_false_for_missing_dir(tmp_path):
    """A directory that doesn't exist is not a SavedModel."""
    assert model_loader._has_savedmodel_files(str(tmp_path / "nope")) is False


def test_has_savedmodel_files_false_for_empty_dir(tmp_path):
    """An empty directory is not a SavedModel."""
    assert model_loader._has_savedmodel_files(str(tmp_path)) is False


def test_has_savedmodel_files_true_when_pb_present(tmp_path):
    """A directory with saved_model.pb is recognised as a SavedModel."""
    (tmp_path / "saved_model.pb").write_bytes(b"")
    assert model_loader._has_savedmodel_files(str(tmp_path)) is True


def test_load_model_raises_when_nothing_configured(tmp_path, monkeypatch):
    """No local files and no env var -> clear RuntimeError naming both options."""
    _reset_module_state()
    monkeypatch.setattr(model_loader, "MODEL_PATH", str(tmp_path / "missing"))
    monkeypatch.delenv("HF_MODEL_REPO_ID", raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        model_loader.load_model()

    msg = str(excinfo.value)
    assert "SavedModel not found" in msg
    assert "HF_MODEL_REPO_ID" in msg


def test_predict_rejects_empty_string():
    """predict() must reject empty input before touching the model."""
    _reset_module_state()
    with pytest.raises(ValueError):
        model_loader.predict("   ")


def test_predict_rejects_non_string():
    """predict() must reject non-string input before touching the model."""
    _reset_module_state()
    with pytest.raises(ValueError):
        model_loader.predict(None)  # type: ignore[arg-type]
