from semantic_search import SemanticSearchEngine
from qa import ExtractiveQAModel


def main():
    print("=" * 60)
    print("Philosophy Semantic Search + Extractive QA")
    print("Roll No.: 09")
    print("=" * 60)

    question = input("\nEnter your question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    search_engine = SemanticSearchEngine()
    results = search_engine.search(question, top_k=3)

    print("\nSemantic Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}  | similarity = {result['score']:.4f}")

    best = results[0]
    print(f"\nSelected document: {best['title']}")

    qa_model = ExtractiveQAModel()
    answer = qa_model.answer(question, best["content"])

    print("\nExtractive QA Answer:")
    print(answer["answer"] or "No answer span was extracted.")
    print(f"QA confidence: {answer['score']:.4f}")


if __name__ == "__main__":
    main()
