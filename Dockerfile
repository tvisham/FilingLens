# FilingLens API / MCP server image. One image, two entry points:
#   docker run ... filinglens            -> FastAPI on :8000 (default)
#   docker run ... filinglens mcp        -> MCP server, Streamable HTTP on :8765
# Embeddings (torch) are included so semantic search works; build with
#   --build-arg EXTRAS=""  for a smaller image without them.
FROM python:3.12-slim AS build
ARG EXTRAS="--extra embeddings"
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project ${EXTRAS}
COPY src ./src
COPY config ./config
COPY alembic.ini ./
RUN uv sync --locked --no-dev ${EXTRAS}

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
COPY --chown=app:app evals/thresholds.yaml ./evals/thresholds.yaml
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 CACHE_DIR=/tmp/sec-cache
EXPOSE 8000 8765
# Configuration comes from the environment (SEC_USER_AGENT, DATABASE_URL, ANTHROPIC_API_KEY,
# API_TOKEN, API_CORS_ORIGINS); nothing secret is baked into the image.
ENTRYPOINT ["/bin/sh", "-c", "if [ \"$1\" = mcp ]; then exec filinglens-mcp --http --host 0.0.0.0 --port 8765; else exec uvicorn filinglens.api.app:app --host 0.0.0.0 --port 8000; fi", "--"]
