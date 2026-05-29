# Multilingual Fake News Detector — Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install Python deps first so they're cached independently of source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source (includes saved_model/ when committed locally).
# If saved_model/ is empty, model_loader.py falls back to downloading from
# the HuggingFace Hub using the HF_MODEL_REPO_ID env var at startup.
COPY . .

# Gradio (7860) and FastAPI (8000).
EXPOSE 7860
EXPOSE 8000

# Default: launch the Gradio app.
# To run the FastAPI service instead, override the command:
#   docker run -p 8000:8000 fake-news-detector \
#     python -m uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["python", "app.py"]
