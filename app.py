# Optional local Streamlit launcher.
# Render should use: uvicorn api:app --host 0.0.0.0 --port $PORT
import streamlit as st
from semantic_search import SemanticSearchEngine
from qa import ExtractiveQAModel

st.title("Philosophy Semantic Search + Extractive QA")
st.caption("Experiment 7 | Roll No. 09 | Philosophy")
st.write("Render-compatible version: FastEmbed for local embeddings + Hugging Face Inference for extractive QA.")

@st.cache_resource
def get_engine():
    return SemanticSearchEngine()

question = st.text_input("Enter your question")
if st.button("Search & Answer") and question.strip():
    results = get_engine().search(question, 3)
    for i, r in enumerate(results, 1):
        st.write(f"**{i}. {r['title']}** — cosine similarity: `{r['score']:.4f}`")
    try:
        qa = ExtractiveQAModel()
        answer = qa.answer(question, results[0]["content"])
        st.success(answer["answer"])
        st.write(f"QA confidence: `{answer['score']:.4f}`")
    except Exception as exc:
        st.error(str(exc))
