import re
from typing import Any, Dict, List, Tuple

_WORDS = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "what", "when", "which", "who", "why", "with",
}


class ExtractiveQAModel:
    """Small, local extractive QA implementation with no API key or model service.

    It selects the most relevant sentence(s) from the retrieved document. This
    keeps the Render deployment reliable and avoids external inference failures.
    """

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            word.lower()
            for word in _WORDS.findall(text)
            if word.lower() not in _STOPWORDS
        }

    def answer(self, question: str, context: str) -> Dict[str, Any]:
        question_tokens = self._tokens(question)
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", context.strip())
            if sentence.strip()
        ]
        if not sentences:
            return {"answer": "No answer found.", "score": 0.0, "start": 0, "end": 0}

        scored: List[Tuple[float, str]] = []
        for sentence in sentences:
            sentence_tokens = self._tokens(sentence)
            overlap = question_tokens & sentence_tokens
            coverage = len(overlap) / max(len(question_tokens), 1)
            density = len(overlap) / max(len(sentence_tokens), 1)
            scored.append((coverage * 0.75 + density * 0.25, sentence))

        score, answer = max(scored, key=lambda item: item[0])
        start = context.find(answer)
        return {
            "answer": answer,
            "score": float(min(score, 1.0)),
            "start": max(start, 0),
            "end": max(start, 0) + len(answer),
        }
