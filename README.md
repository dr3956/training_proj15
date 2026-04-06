# training_proj15

ML training stack for Chameleon Cloud: sklearn **tone classifier** (`train.py`) and **small LLM LoRA** generator (`train_llm.py`). Sample API contracts live in `samples/`.

## Training

- `training/` — Dockerfiles, YAML configs, `train.py`, `train_llm.py`, seed data.
- Build (from repo root): `docker build -f training/Dockerfile -t tone-train:proj15 ./training`
- LLM image: `docker build -f training/Dockerfile.llm -t llm-train:proj15 ./training`
- Set `MLFLOW_TRACKING_URI` to your Chameleon MLflow server before `docker run`.

## Docs

Course materials: `DOCS/`.
