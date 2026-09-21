from pathlib import Path
from typing import Dict, List

import numpy as np
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_NAMES = (
    "epistemology.txt",
    "ethics.txt",
    "existentialism.txt",
    "metaphysics.txt",
    "stoicism.txt",
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


class SemanticSearchEngine:
    def __init__(self):
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.documents = self._load_documents()
        self.embeddings = np.asarray(
            list(self.model.embed([d["content"] for d in self.documents])),
            dtype=np.float32,
        )
        self.embeddings = self._normalize(self.embeddings)

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _load_documents(self) -> List[Dict]:
        documents = []
        missing = []

        for filename in DOCUMENT_NAMES:
            path = BASE_DIR / filename
            if not path.is_file():
                missing.append(filename)
                continue

            content = path.read_text(encoding="utf-8").strip()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                raise ValueError(f"{filename} must contain at least 2 paragraphs.")

            documents.append(
                {
                    "id": path.stem,
                    "title": path.stem.replace("_", " ").title(),
                    "content": content,
                    "paragraphs": paragraphs,
                }
            )

        if missing:
            raise FileNotFoundError(
                "Missing required philosophy documents: " + ", ".join(missing)
            )
        if len(documents) < 5:
            raise ValueError("At least 5 documents are required.")
        return documents

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query = query.strip()
        if not query:
            raise ValueError("Question cannot be empty.")

        query_vector = np.asarray(
            list(self.model.embed([query])),
            dtype=np.float32,
        )
        query_vector = self._normalize(query_vector)[0]

        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in indices:
            result = dict(self.documents[int(idx)])
            result["score"] = float(scores[idx])
            results.append(result)
        return results
