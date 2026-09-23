FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir '.[asr]'

EXPOSE 8000
CMD ["uvicorn", "sts_ws_server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
