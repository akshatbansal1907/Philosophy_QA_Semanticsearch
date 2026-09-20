from typing import Dict
from transformers import pipeline


class ExtractiveQAModel:
    """Extracts an answer span from a supplied context."""

    def __init__(
        self,
        model_name: str = "distilbert-base-cased-distilled-squad",
    ):
        self.model_name = model_name
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
        )

    def answer(self, question: str, context: str) -> Dict:
        result = self.qa_pipeline(
            question=question,
            context=context,
            handle_impossible_answer=True,
        )
        return {
            "answer": result.get("answer", "").strip(),
            "score": float(result.get("score", 0.0)),
            "start": result.get("start"),
            "end": result.get("end"),
        }
