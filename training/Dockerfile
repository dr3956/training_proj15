# Training container for Chameleon: run with MLFLOW_TRACKING_URI pointing at your tracking server.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY train.py .
COPY configs/ /app/configs/
COPY data/ /app/data/

# Default config is overridable (course: single script, config-selected candidates)
ENTRYPOINT ["python", "/app/train.py"]
CMD ["--config", "/app/configs/baseline_tone_nb.yaml"]
