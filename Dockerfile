FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN if [ -f uv.lock ]; then uv sync --frozen --no-dev; else uv sync --no-dev; fi

COPY . .

RUN printf '%s\n' \
'#!/usr/bin/env bash' \
'set -e' \
'cd /app' \
'if [ -f .env ]; then set -a; . ./.env; set +a; fi' \
': "${API_PORT:=8000}"' \
'uvicorn planparser.api:app --host 0.0.0.0 --port "$API_PORT" &' \
'python -u planparser/app.py' \
> /app/run.sh && chmod +x /app/run.sh

EXPOSE 7860
CMD ["/app/run.sh"]
