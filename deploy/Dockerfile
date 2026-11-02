# FUTURE: not part of the current local release; see DEPLOY.md.
# One container: the Vite frontend served by FastAPI on port 7860. Everything retrieval needs is
# copied in at build time, so nothing downloads at startup. No key is ever copied or baked:
# GROQ_API_KEY, GEMINI_API_KEY and JINA_API_KEY come from the Space's secrets at runtime.

FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ENV VITE_API_URL=/
RUN npm run build

FROM python:3.11-slim
RUN useradd -m -u 1000 app
WORKDIR /app
COPY deploy/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY deploy/ ./deploy/
COPY ingest/ ./ingest/
COPY data/ ./data/
COPY --from=ui /ui/dist ./ui
RUN chown -R app:app /app
USER app

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    SAHAYI_REWRITE_CACHE=0 \
    SAHAYI_RERANK_CACHE=0 \
    PORT=7860 \
    STATIC_DIR=/app/ui \
    RETRIEVAL_CONFIGS_PATH=/app/deploy/configs.yaml \
    RETRIEVAL_CONFIG=deploy_local

EXPOSE 7860
# start-period covers model loading
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"
CMD ["python", "deploy/serve.py"]
