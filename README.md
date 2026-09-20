# Philosophy QA Semantic Search — Render Compatible

**Roll No.: 09 | Experiment 7 | Topic: Philosophy**

This version is designed for a small-memory Render web service.

## Why the original deployment failed

The original version loaded both Sentence-Transformers/PyTorch and a local Transformers QA model. Render reported:

`Out of memory (used over 512Mi)`

## Render-compatible architecture

```text
Question
   |
   v
FastEmbed (ONNX, lightweight)
   |
   v
True dense embedding
   |
   v
Cosine similarity
   |
   v
Best Philosophy document
   |
   v
Hugging Face Inference API
   |
   v
Extractive QA answer
```

FastEmbed uses ONNX Runtime and is designed to be lighter than Transformer/Sentence-Transformer runtime stacks. The QA model is called remotely so the Render process does not load PyTorch and the QA model into its 512 MiB RAM.

## Files

- `api.py` — Render web API + browser UI
- `semantic_search.py` — true embeddings + cosine similarity
- `qa.py` — Hugging Face extractive QA
- `app.py` — optional local Streamlit UI
- `run.py` — local Uvicorn launcher
- `render.yaml` — Render configuration
- `requirements.txt`
- `data/` — 5 philosophy documents
- `tests/`

## Render settings

Build Command:
```text
pip install -r requirements.txt
```

Start Command:
```text
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Environment Variable:
```text
HF_TOKEN = your Hugging Face access token
```

The token is used only by the Hugging Face client for the remote extractive QA call. Do not put the token inside GitHub source files.

## Test after deployment

Open:

```text
https://YOUR-RENDER-SERVICE.onrender.com/
```

Health check:

```text
https://YOUR-RENDER-SERVICE.onrender.com/health
```

Try:

`What does Stoicism teach about controlling emotions?`

## Important

The semantic-search stage still uses **true embeddings**, not TF-IDF or Bag-of-Words. FastEmbed's ONNX model produces dense vectors, and the code calculates cosine similarity using normalized vectors.

The QA stage uses an actual extractive QA model (`distilbert/distilbert-base-cased-distilled-squad`) through Hugging Face Inference Providers, avoiding local model memory usage on Render.
