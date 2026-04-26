# backend_fastapi/Dockerfile
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential curl gcc g++ libffi-dev libssl-dev \
    libpq-dev libxml2-dev libxslt1-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

RUN pip install --no-cache-dir poetry==$POETRY_VERSION

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

COPY . .
RUN poetry build && \
    poetry install --only main

# ---------- Final Image ----------
FROM python:3.11-slim

ENV PATH="/app/.venv/bin:$PATH" \
    HOME="/home/appuser"

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -s /bin/bash appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

RUN chown -R appuser:appuser /app && \
    chmod 755 /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8000/api/health'); exit(0 if r.status_code == 200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
