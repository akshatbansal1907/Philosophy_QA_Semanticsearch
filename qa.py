import os
from huggingface_hub import InferenceClient

QA_MODEL = "distilbert/distilbert-base-cased-distilled-squad"


class ExtractiveQAModel:
    """
    Render-safe extractive QA.

    The QA model runs through Hugging Face Inference Providers instead of
    loading PyTorch/Transformers into the 512 MiB Render process.
    """

    def __init__(self):
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not configured. Add a Hugging Face token in "
                "Render Environment Variables."
            )
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=token,
        )

    def answer(self, question: str, context: str):
        result = self.client.question_answering(
            question=question,
            context=context,
            model=QA_MODEL,
            handle_impossible_answer=True,
            top_k=1,
        )
        return {
            "answer": result.answer.strip(),
            "score": float(result.score),
            "start": int(result.start),
            "end": int(result.end),
        }
