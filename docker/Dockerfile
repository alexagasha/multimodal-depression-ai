FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/synthetic/sessions/ ./data/synthetic/sessions/

# Entry point: run inference on all participants in data/synthetic/sessions/
# SWAP: point DATA_ROOT at real data volume mount when deploying on Railway
CMD ["python", "-m", "src.fusion.run_pipeline"]
