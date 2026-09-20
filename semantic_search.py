from pathlib import Path
from typing import List, Dict
import numpy as np
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# FastEmbed uses ONNX Runtime rather than loading PyTorch/Sentence-Transformers.
# This keeps RAM usage much lower on small Render instances.
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
    def _normalize(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def _load_documents(self) -> List[Dict]:
        documents = []
        for path in sorted(DATA_DIR.glob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                raise ValueError(f"{path.name} must contain at least 2 paragraphs.")
            documents.append({
                "id": path.stem,
                "title": path.stem.replace("_", " ").title(),
                "content": content,
                "paragraphs": paragraphs,
            })
        if len(documents) < 5:
            raise ValueError("At least 5 documents are required.")
        return documents

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vector = np.asarray(
            list(self.model.embed([query])),
            dtype=np.float32,
        )
        query_vector = self._normalize(query_vector)[0]

        # Both vectors are L2-normalized, so their dot product is cosine similarity.
        scores = self.embeddings @ query_vector
        indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in indices:
            result = dict(self.documents[int(idx)])
            result["score"] = float(scores[idx])
            results.append(result)
        return results
