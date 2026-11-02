# Deploying Sahayi to Hugging Face Spaces (Docker, CPU basic)

> **Future work, not part of the current release.** Sahayi now ships as a local app with an
> optional demo link: see "How to run locally" and "Demo link" in the [README](README.md). Docker
> Spaces may also need a paid plan now. These notes and the `deploy/` files are kept for later.

One container: FastAPI serves the API and the built frontend on port 7860. The search index,
chunks and both int8 ONNX models are copied in at build time, so nothing downloads at startup.

**Status:** the image has not been built yet, because the development machine has no Docker. The
same files and start command were tested natively; see "What has been tested". The first build
on Hugging Face is the first real build, so watch its log.

## Before you start

You need a Hugging Face account, [Git LFS](https://git-lfs.com), and these on your machine:
`data/processed/`, `data/index/` and the exported query encoder. To re-export the encoder:

```bash
python -m backend.app.retrieval.export_onnx --query-encoder
```

## 1. Build the Space folder

```bash
python deploy/build_bundle.py
```

This writes `data/deploy_space/` (about 894 MB, mostly the two model files). It contains no
`.env` and no keys.

## 2. Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Choose **Docker** as the SDK, the **Blank** template and **CPU basic** hardware.
3. Create it, then clone it:

```bash
git lfs install
git clone https://huggingface.co/spaces/<you>/sahayi D:\rag\data\space_repo
```

## 3. Push the code and models

```bash
cp -r data/deploy_space/. data/space_repo/
cd data/space_repo
git lfs track "*.onnx" "*.sqlite"
git add .
git status            # check that no .env is listed
git commit -m "Deploy Sahayi"
git push
```

When Git asks for a password, use a Hugging Face access token with write permission. The Space
starts building by itself; the **Logs** tab shows progress.

## 4. Set secrets

In the Space, open **Settings → Variables and secrets** and add these as **secrets**, never as
plain variables:

| secret | needed for |
|---|---|
| `GROQ_API_KEY` | query rewrite and answers |
| `GEMINI_API_KEY` | last generator in the fallback chain (optional) |
| `JINA_API_KEY` | only if you switch to `deploy_hosted` |

The Space restarts after each change. Keys never enter the image: `.env` is not in the bundle.

**Before adding `GEMINI_API_KEY`:** Google's pricing page says free-tier usage may be used to
improve its products, and the fallback would send real users' questions there. Use a paid key
or leave this secret out.

To demo with no LLM calls, also add the variable `ANSWER_MODE=retrieval_only`. Answers then show
sources, cited excerpts and a busy notice.

## 5. Check it

```bash
curl https://<you>-sahayi.hf.space/health
```

Expect `"status":"ok"` and `"answerer_loaded":true`. `groq_key` should be `true`, and
`retrieval_config` should be `deploy_local`. Then open the Space and ask a question.

The API only answers once the models are loaded (10–15 s in the native test). A free Space
sleeps when idle, so the first visitor after that waits for a restart.

## Reranker choice

`deploy_local` (the default) reranks with the int8 model in the image. To try hosted Jina, set
the variable `RETRIEVAL_CONFIG=deploy_hosted` and the `JINA_API_KEY` secret. In the 2-core test
it was slower and returned HTTP 429 on 54 of 171 queries.

## What has been tested

`deploy/simulate_2cpu.py` and `deploy/smoke_local.py` run from `data/deploy_space/` in a clean
venv with no torch, pinned to 2 cores, using the same `deploy/serve.py`. This is Windows, not
Linux, and core pinning is not Docker's `--cpus=2`. **Not tested:** the Docker build, the
Linux wheels, the cgroup CPU limit, and Space hardware, which is usually slower than this laptop.
To test with Docker available:

```bash
docker build -t sahayi data/deploy_space
docker run --cpus=2 --memory=16g -p 7860:7860 -e ANSWER_MODE=retrieval_only sahayi
cd frontend && E2E_BASE_URL=http://localhost:7860 npx playwright test
docker images sahayi     # the real image size
```

## Limits to expect

- Groq's free tier allows qwen about 1–2 answers a minute and about 50 a day. Past that, the API
  falls through the chain, then shows the busy notice.
- `/ask` allows 10 requests a minute per client (`RATE_LIMIT_PER_MIN`).
- Logs and the answer cache live in the container and are lost on restart.
