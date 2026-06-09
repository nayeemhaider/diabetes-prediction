# ─────────────────────────────────────────────────────────────────
# Stage 1: Builder — install Python deps into a clean layer
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# Copy only requirements first (better layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install/deps -r requirements.txt


# ─────────────────────────────────────────────────────────────────
# Stage 2: Runtime — lean final image
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="diabetes-api"
LABEL description="Diabetes Risk Prediction API — XGBoost + FastAPI"
LABEL version="1.0.0"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY app/         ./app/
COPY model_artifacts/ ./model_artifacts/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check — hits /health every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start Uvicorn
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
