import os
from typing import Any, Dict

from huggingface_hub import InferenceClient

QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


class ExtractiveQAModel:
    """Render-safe extractive QA using the Hugging Face Inference API."""

    def __init__(self):
        token = os.getenv("HF_TOKEN")
        self.client = (
            InferenceClient(provider="hf-inference", api_key=token)
            if token
            else InferenceClient(provider="hf-inference")
        )

    def answer(self, question: str, context: str) -> Dict[str, Any]:
        try:
            result = self.client.question_answering(
                question=question,
                context=context,
                model=QA_MODEL,
                handle_impossible_answer=True,
                top_k=1,
            )
        except Exception as exc:
            message = str(exc)
            if "401" in message or "Unauthorized" in message or "Invalid username or password" in message:
                raise RuntimeError(
                    "Hugging Face authentication failed. Set a valid HF_TOKEN in the Render environment variables."
                ) from exc
            raise RuntimeError(f"QA request failed: {message}") from exc

        if isinstance(result, dict):
            answer = result.get("answer", "")
            score = result.get("score", 0.0)
            start = result.get("start", 0)
            end = result.get("end", 0)
        else:
            answer = getattr(result, "answer", "")
            score = getattr(result, "score", 0.0)
            start = getattr(result, "start", 0)
            end = getattr(result, "end", 0)

        return {
            "answer": str(answer).strip() or "No answer found in the selected document.",
            "score": float(score),
            "start": int(start),
            "end": int(end),
        }
