from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


class SemanticSearchEngine:
    """Semantic document search using true sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.documents = self._load_documents()
        self.embeddings = self.model.encode(
            [doc["content"] for doc in self.documents],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _load_documents(self) -> List[Dict[str, str]]:
        documents = []
        for path in sorted(DATA_DIR.glob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if len(paragraphs) < 2:
                raise ValueError(f"{path.name} must contain at least two paragraphs.")
            documents.append(
                {
                    "id": path.stem,
                    "title": path.stem.replace("_", " ").title(),
                    "content": content,
                    "paragraphs": paragraphs,
                }
            )

        if len(documents) < 5:
            raise ValueError("The project must contain at least five documents.")
        return documents

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            result = dict(self.documents[idx])
            result["score"] = float(scores[idx])
            results.append(result)
        return results


if __name__ == "__main__":
    engine = SemanticSearchEngine()
    question = input("Enter your question: ").strip()
    for result in engine.search(question):
        print(f"{result['title']}: {result['score']:.4f}")
