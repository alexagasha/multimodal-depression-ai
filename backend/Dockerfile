# Persistent HTTP inference API — this is the "swap CMD for a FastAPI server"
# step the root README's docker-compose.yml comment anticipated. The batch
# CLI image (docker/Dockerfile) is untouched and still works standalone.
#
# Build from the depression-detection/ root so both src/ and backend/ are in
# the build context:
#   docker build -f backend/Dockerfile -t depression-api .
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY tests/ ./tests/
COPY backend/app ./backend/app

WORKDIR /app/backend
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
