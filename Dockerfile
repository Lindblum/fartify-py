FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5181 \
    FLASK_DEBUG=0 \
    FARTIFY_STATE_DIR=/app/state \
    TORCH_HOME=/cache/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first, so demucs/torchcrepe don't pull the multi-GB CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN mkdir -p uploads output state /cache/torch

EXPOSE 5181
CMD ["python", "app.py"]
