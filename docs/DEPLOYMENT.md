# Deployment

## Why this document matters

Deployment is where most ML projects die. Local "it works on my machine" is the easy part; turning that into a service someone else can reach is several decisions that compound. This page enumerates the three deployment targets this project supports, what each one is good for, and what can go wrong.

## Pre-flight checklist

Run this once before any deploy:

```bash
# Code
python -m py_compile model_loader.py app.py api.py evaluate.py
pytest -q tests/

# Container build (will succeed even without saved_model/)
docker build -t fake-news-detector .

# Model source decided?
#   Option A: ./saved_model/ committed in the repo
#   Option B: HF_MODEL_REPO_ID set in the deploy environment
```

If any of those fail, fix locally before deploying.

## Target 1 — HuggingFace Spaces (recommended for demos)

### When to use it
- Public demo for a portfolio.
- Free CPU hosting for low-traffic interactive use.
- Anyone reviewing the project can click a link and try it.

### Prerequisites
1. HuggingFace account.
2. `pip install huggingface_hub`.
3. `huggingface-cli login`.
4. A Space created at <https://huggingface.co/new-space> with **SDK = Gradio**.

### Steps

**Windows:**
```powershell
.\deploy.ps1 -SpaceOwner <your-hf-username>
# Custom name:
.\deploy.ps1 -SpaceOwner <user> -SpaceName custom-name
# First push to a non-empty Space:
.\deploy.ps1 -SpaceOwner <user> -Force
```

**Linux / macOS:**
```bash
chmod +x deploy.sh
./deploy.sh <your-hf-username>
./deploy.sh <user> custom-name
./deploy.sh <user> custom-name --force
```

The script:
1. Verifies `huggingface-cli` is installed and you're logged in.
2. Adds/updates a git remote named `hf` pointing at the Space.
3. Pushes the current branch to the Space's `main`.
4. Prints the live Space URL.

### Model source on Spaces
- **Option A (simplest):** commit `saved_model/` directly in the repo. The `.gitignore` is configured to track it. Note that Spaces have a 50 GB limit on the repo, and Git LFS is recommended for files over 10 MB.
- **Option B:** set `HF_MODEL_REPO_ID` as a Space secret (Settings → Variables and secrets). The Hub fallback in `model_loader.py` picks it up at startup.

### Failure modes
| Symptom | Fix |
|---|---|
| Push rejected as "unrelated histories" | First push to a Space that already has commits — re-run with `-Force` / `--force`. |
| Space builds but errors on launch | Check the Space's "Logs" tab. Usually the SavedModel isn't available — either commit it or set `HF_MODEL_REPO_ID`. |
| "Killed" in build logs | Out of memory during `pip install`. Upgrade to the next tier of Space hardware. |

## Target 2 — Docker (recommended for self-hosting)

### When to use it
- You have a server (VPS, on-prem, cloud VM).
- You want one image to run both the Gradio UI and the FastAPI service.
- You're integrating with other services via docker-compose.

### Single container

```bash
docker build -t fake-news-detector .

# Run the Gradio UI on :7860
docker run --rm -p 7860:7860 \
  -e HF_MODEL_REPO_ID=<owner>/<repo> \
  fake-news-detector

# Run the FastAPI service on :8000 instead
docker run --rm -p 8000:8000 \
  -e HF_MODEL_REPO_ID=<owner>/<repo> \
  fake-news-detector \
  python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Two services with docker-compose

```bash
# Set the model source if you don't have ./saved_model/ locally
export HF_MODEL_REPO_ID="<owner>/<repo>"   # PowerShell: $env:HF_MODEL_REPO_ID = "..."

docker compose up --build
# Gradio:  http://localhost:7860
# FastAPI: http://localhost:8000
```

### Mounting the model from the host

If your SavedModel lives on the host and you don't want to bake it into the image, the compose file already bind-mounts `./saved_model:/app/saved_model`. Just put the SavedModel files in `./saved_model/` on the host and the containers will see them.

### Failure modes
| Symptom | Fix |
|---|---|
| `RuntimeError: SavedModel not found` on container start | Either bind-mount your `./saved_model/` or set `HF_MODEL_REPO_ID` via `-e` or compose `environment`. |
| Container exits immediately with no logs | `docker logs <container>` — usually a tokenizer download failure due to no network access. |
| OOM kill | Increase Docker's memory limit (Docker Desktop defaults are ~4 GB on Mac/Windows). |

## Target 3 — Cloud (general guidance)

The Dockerfile is platform-agnostic, so any container runtime works. Specifics differ:

### AWS — App Runner or ECS Fargate
- Push the image to ECR: `docker tag fake-news-detector <acct>.dkr.ecr.<region>.amazonaws.com/fake-news-detector && docker push ...`
- App Runner is the lowest-friction: point it at the ECR image and set `HF_MODEL_REPO_ID` as an env var.
- ECS Fargate is more configurable but needs a task definition + load balancer.

### GCP — Cloud Run
- `gcloud builds submit --tag gcr.io/<project>/fake-news-detector`
- `gcloud run deploy --image gcr.io/<project>/fake-news-detector --set-env-vars HF_MODEL_REPO_ID=<owner>/<repo>`
- Cloud Run scales to zero, so cold starts include the ~10 s model load. Set min instances ≥ 1 if you can't tolerate that.

### Azure — Container Apps
- `az containerapp up --name fake-news-detector --image <acr>/fake-news-detector --env-vars HF_MODEL_REPO_ID=<owner>/<repo>`

### Common gotchas across clouds
- **CPU vs GPU pricing.** Inference is fine on CPU at low traffic. Pay for GPU only when measured throughput justifies it.
- **Health checks.** Point the platform's health probe at `GET /` — the FastAPI service returns 200 with the model name once ready.
- **Concurrency.** The FastAPI service is stateless per request, so horizontal scaling works. Watch memory — each replica holds its own copy of the model (~700 MB).

## Environment variables reference

| Variable | Required? | What it does |
|---|---|---|
| `HF_MODEL_REPO_ID` | Required if `./saved_model/` is empty | HuggingFace Hub repo id (`<owner>/<repo>`) to download the SavedModel from on first run. |
| `HF_TOKEN` | Required if your model repo is private/gated | Standard HuggingFace token; `huggingface_hub` picks it up automatically. |

See [`.env.example`](../.env.example) for the documented schema.

## Monitoring

There are no built-in metrics today. Easy wins for production:

- Add `prometheus-fastapi-instrumentator` to expose request counts/latency at `/metrics`.
- Log structured JSON (one line per request) to stdout — your platform's log collector takes it from there.
- Alert on: 5xx rate, 95p latency, memory > 80%, container restarts.

## Rollback

There's no migration or schema drift to worry about — the service is stateless. Roll back by:

1. **HF Spaces:** revert the commit on the `main` branch you pushed, push again with `--force`.
2. **Docker / cloud:** redeploy the previous image tag.

Tag every release in git so previous versions are easy to find:
```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
```
