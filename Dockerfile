# =============================================================================
# Dockerfile — compliance-api (ECS Fargate, CPU only)
# =============================================================================
# Target: app/server.py (FastAPI + uvicorn + in-process cross-encoder reranker)
# GPU inference (LLM + embeddings) runs on a separate vLLM EC2 instance.
#
# Build:
#   docker build -t compliance-api .
#
# Run (local test):
#   docker run --rm -p 8000:8000 --env-file .env compliance-api
#
# The reranker model (~2.2 GB) is baked into the image at build time via
# the HF_MODEL_DOWNLOAD stage so Fargate tasks don't pull it on cold start.
# =============================================================================

# ── Stage 1: download HuggingFace model cache ─────────────────────────────────
FROM python:3.11-slim AS hf-cache

ARG RERANKER_MODEL=BAAI/bge-reranker-v2-m3

RUN pip install --no-cache-dir huggingface_hub

# Download only the model files needed for inference (safetensors + config).
# The cache lands in /root/.cache/huggingface which is COPY'd into the final image.
RUN python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='${RERANKER_MODEL}',
    ignore_patterns=['*.msgpack', '*.h5', 'flax_model*', 'tf_model*'],
)
print('Model cached.')
"


# ── Stage 2: install Python deps ──────────────────────────────────────────────
FROM python:3.11-slim AS deps

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy lock files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all deps into the project venv (no editable install of the project itself)
RUN uv sync --frozen --no-install-project


# ── Stage 3: final runtime image ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --no-create-home appuser

WORKDIR /app

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy installed venv from deps stage
COPY --from=deps /app/.venv /app/.venv

# Copy pre-downloaded HuggingFace model cache so cold starts don't pull 2.2 GB
COPY --from=hf-cache /root/.cache/huggingface /home/appuser/.cache/huggingface

# Copy application source
COPY app/       ./app/
COPY data/      ./data/
COPY workflow/  ./workflow/
COPY prompts/   ./prompts/
COPY pyproject.toml uv.lock ./

# Ensure the venv is on PATH (uv run picks it up automatically)
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/home/appuser/.cache/huggingface"

# Switch to non-root user
RUN chown -R appuser:appgroup /app /home/appuser
USER appuser

EXPOSE 8000

# Uvicorn serves the FastAPI app. Workers=1 because the in-process reranker
# is not fork-safe; scale out by running more ECS tasks (horizontal scaling).
CMD ["uvicorn", "app.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
