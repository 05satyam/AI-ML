# Multi-stage-ish slim image. Talking points: pin base image, run as non-root,
# small layers, no secrets baked in (injected via env / k8s Secret at runtime).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval

# Non-root user for least privilege.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Tune workers to CPU at deploy time; keep one process per core typically.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
