# syntax=docker/dockerfile:1.4
# ==============================================================================
# Sim2Real PPO Navigation Stack Container
# Gymnasium + Stable-Baselines3 + Domain Randomization + ONNX Export
# ==============================================================================

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY envs/ /app/envs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

CMD ["bash", "scripts/run_training.sh"]
