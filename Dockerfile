FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

FROM base AS dev
RUN pip install -e ".[dev]"
CMD ["uvicorn", "atlas_rag.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

FROM base AS prod
RUN pip install -e "."

RUN useradd -m -u 1001 atlas
USER atlas

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn", "atlas_rag.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
