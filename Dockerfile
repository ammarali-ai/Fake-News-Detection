# Multilingual Fake News Detector — Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install Python deps first so they're cached independently of source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source.
COPY . .

# Copy the TensorFlow SavedModel folder (kept as a separate COPY for clarity).
COPY saved_model/ ./saved_model/

# Gradio (7860) and FastAPI (8000).
EXPOSE 7860
EXPOSE 8000

# Default: launch the Gradio app.
# To run the FastAPI service instead, override the command:
#   docker run -p 8000:8000 fake-news-detector \
#     python -m uvicorn api:app --host 0.0.0.0 --port 8000
CMD ["python", "app.py"]
