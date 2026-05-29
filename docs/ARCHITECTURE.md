# Architecture

## Why this document matters

Anyone touching this codebase needs a one-page mental model before reading code. This page explains how the pieces fit together, why the boundaries were drawn where they are, and what to keep invariant when changing things.

## High-level diagram

```
                    ┌──────────────────────────────┐
                    │     model_loader.py          │
                    │  (single source of truth)    │
                    │                              │
   ┌──────────┐     │   load_model()  ─────────┐   │
   │  app.py  │ ──► │   load_tokenizer()       │   │
   │ (Gradio) │     │   predict(text)          │   │
   └──────────┘     │                          │   │
                    │   ┌──────────────────┐   │   │
   ┌──────────┐     │   │  TF SavedModel   │ ◄┘   │
   │  api.py  │ ──► │   │  (./saved_model) │      │
   │(FastAPI) │     │   │  or HF Hub fetch │      │
   └──────────┘     │   └──────────────────┘      │
                    │                              │
   ┌──────────┐     │   ┌──────────────────┐      │
   │evaluate. │ ──► │   │ bert-base-       │      │
   │   py     │     │   │ multilingual-    │      │
   └──────────┘     │   │ cased tokenizer  │      │
                    │   └──────────────────┘      │
                    └──────────────────────────────┘
```

## Components

### `model_loader.py` — the load-once boundary
The TensorFlow SavedModel and the BERT tokenizer are expensive to load (≈ 700 MB RAM, ~10 s cold start). Loading them more than once in a process burns memory and adds latency on every request.

`model_loader.py` solves this by being the **only** module in the codebase that calls `tf.saved_model.load()` or `AutoTokenizer.from_pretrained()`. All callers go through `predict(text)`. Module-level singletons (`_model`, `_tokenizer`, `_serving_fn`) guarantee one load per process.

**Invariant — do not break:** if you ever add a second `tf.saved_model.load` or `AutoTokenizer.from_pretrained` call anywhere in the repo, the architecture has regressed. Use `model_loader` instead.

### `app.py` — the Gradio entry point
This is what HuggingFace Spaces launches. The frontmatter in [`README.md`](../README.md) (`app_file: app.py`) is what tells the Space which file to run. `app.py` imports from `model_loader` at module-level and calls `load_model()` + `load_tokenizer()` on import so the Space is "ready" the moment the UI binds the port.

### `api.py` — the REST surface
FastAPI service that exposes the same `predict()` function over HTTP for programmatic clients. Uses a `@app.on_event("startup")` hook to warm `model_loader` instead of loading at import — keeps `uvicorn --reload` cycles fast.

### `evaluate.py` — the offline harness
CLI that batch-scores a labelled CSV and writes `evaluation_results.json`. Uses `model_loader.predict()` row-by-row. No new model loading.

## Request flow

### Gradio path
```
user types text in browser
  → Gradio runs analyse(text, language)
    → model_loader.predict(text)
      → _tokenizer(text)
      → _serving_fn(input_ids, attention_mask)
      → softmax + argmax
    → returns {prediction, confidence, scores}
  → Gradio renders Textbox + Label outputs
```

### FastAPI path
```
client → POST /predict {text, language}
  → Pydantic validates body (422 on failure)
  → model_loader.predict(text)
  → returns JSON {prediction, confidence, scores, language}
```

## Model source resolution

At startup, `model_loader.load_model()` checks (in order):

1. **Local files** — does `./saved_model/saved_model.pb` exist? If yes, load directly.
2. **HF Hub fallback** — is the `HF_MODEL_REPO_ID` env var set? If yes, `huggingface_hub.snapshot_download()` into `./saved_model/`, then load.
3. **Hard fail** — raise `RuntimeError` naming both options.

This dual path means the same image works for HF Spaces (model committed alongside code) **and** for cloud deploys where you'd rather pull the model at boot from a HF Hub repo.

## Why these boundaries?

| Decision | Why |
|---|---|
| Three entry points (`app.py`, `api.py`, `evaluate.py`) reusing one loader | Different consumers (humans, services, offline jobs) need different interfaces. The model is the expensive shared resource — one loader, three thin wrappers. |
| Pydantic validation in `api.py`, manual checks in `app.py` | Gradio components already constrain input client-side; FastAPI can't trust HTTP clients, so it re-validates with Pydantic. |
| Hub fallback in `model_loader`, not `app.py` | All entry points benefit equally — the logic belongs at the shared layer. |
| Module-level singletons instead of a class | The state is process-global. A class would only add ceremony without giving us isolation we'd actually use. |
| `@app.on_event("startup")` in FastAPI, eager load in Gradio | FastAPI dev workflow uses `--reload`; eager loading on every reload is painful. Gradio + HF Spaces always boot fresh, so eager load is fine. |

## Future architecture notes

- **Batching** — `/predict/batch` currently loops one prediction at a time. The model could accept a batched tensor for ~5× throughput at high volume. See `evaluate.py`'s `run_predictions` for the same pattern.
- **Lifespan vs. on_event** — `@app.on_event` is deprecated in newer FastAPI. The pinned `fastapi==0.104.1` still supports it; migrate when bumping the pin.
- **Streaming responses** — not currently needed (predictions are sub-second), but the FastAPI `StreamingResponse` would be a natural fit if we ever score long documents.
