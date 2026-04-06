# Training container for Chameleon: run with MLFLOW_TRACKING_URI pointing at your tracking server.
# Build with code version for MLflow (fills tag code_version_git_sha):
#   docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -f training/Dockerfile -t tone-train:proj15 ./training
FROM python:3.11-slim

ARG GIT_SHA=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_SHA=${GIT_SHA}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY train.py .
COPY configs/ /app/configs/
COPY data/ /app/data/

# Default config is overridable (course: single script, config-selected candidates)
ENTRYPOINT ["python", "/app/train.py"]
CMD ["--config", "/app/configs/baseline_tone_nb.yaml"]
