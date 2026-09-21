# Philosophy QA Semantic Search — Render Ready

This application performs semantic search with FastEmbed and local extractive question answering. It does **not** require `HF_TOKEN`, Hugging Face credentials, or any external inference service.

## Render deployment

Create a Render **Web Service** for this repository. Render can use `render.yaml`, or configure:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`

No environment variables are required.

## How it works

1. FastEmbed creates dense embeddings for the five philosophy documents.
2. Cosine similarity selects the most relevant document.
3. A local extractive QA component selects the most relevant answer sentence.

The local QA design is intentional: it avoids 401 errors, API rate limits, cold-start network failures, and large local Transformer memory requirements on small Render instances.

The source documents are `epistemology.txt`, `ethics.txt`, `existentialism.txt`, `metaphysics.txt`, and `stoicism.txt`.

## Test

Open the deployed URL and ask, for example:

- `What does Stoicism teach about control?`
- `What is epistemology?`
- `What does virtue ethics focus on?`

Health endpoint: `/health`
