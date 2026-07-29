FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY data/synthetic/sessions/ ./data/synthetic/sessions/

EXPOSE 8000

# Default: persistent HTTP API (see docker-compose.yml's `api` service). This
# same image also runs the one-off batch scoring job — override the command:
#   docker compose --profile batch up inference
# SWAP: this is the image Cloud Run serves in prod (see docs/system-roadmap.md
# Phase 4) — local and prod run identical code paths.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
