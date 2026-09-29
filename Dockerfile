# Image d'inférence (API + modèle ONNX) — sans TensorFlow
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LOG_LEVEL=INFO \
    RATE_LIMIT=60/minute \
    INTRA_OP_THREADS=2

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY web_app/ web_app/
# Modèle versionné hors Git : models/model_v<version>.onnx + .json doivent exister au build
# (ou monter models/ en volume et fixer MODEL_PATH).
COPY models/ models/

RUN useradd --create-home --uid 10001 palunet && mkdir -p /app/logs && chown -R palunet /app/logs
USER palunet

EXPOSE 8000
# Readiness : /health répond 200 uniquement si le modèle est chargé
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
