# ─── Etapa 1: builder ─────────────────────────────────────────────────────────
# Instalamos dependencias en una imagen separada para mantener la imagen final liviana
FROM python:3.11-slim AS builder

WORKDIR /app

# Dependencias del sistema necesarias para compilar paquetes científicos
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt onnxruntime


# ─── Etapa 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="josefrodriguez@gmail.com"
LABEL description="API de Scoring de Riesgo Crediticio — Scotiabank Perú"
LABEL version="1.0.0"

WORKDIR /app

# Copiar dependencias instaladas desde el builder
COPY --from=builder /install /usr/local

# Copiar código fuente (no datos ni notebooks)
COPY src/ ./src/
COPY models/ ./models/

# Variables de entorno por defecto (sobreescribir vía docker-compose o -e)
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    API_RELOAD=false \
    ONNX_MODEL_PATH=models/lgbm_model.onnx \
    PREPROCESSOR_PATH=models/preprocessor.pkl \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Puerto expuesto
EXPOSE 8000

# Healthcheck interno: la API responde en /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Usuario no-root por seguridad
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Comando de inicio
CMD ["sh", "-c", "uvicorn src.api.app:app --host $API_HOST --port $API_PORT"]
