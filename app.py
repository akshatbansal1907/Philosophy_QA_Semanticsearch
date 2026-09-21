import streamlit as st

from qa import ExtractiveQAModel
from semantic_search import SemanticSearchEngine

st.set_page_config(page_title="Philosophy QA", page_icon="📚")
st.title("Philosophy Semantic Search + Extractive QA")
st.caption("Experiment 7 | Roll No. 09 | Philosophy")

@st.cache_resource
def get_engine():
    return SemanticSearchEngine()

question = st.text_input("Enter your question")
if st.button("Search & Answer") and question.strip():
    try:
        results = get_engine().search(question, 3)
        for i, result in enumerate(results, 1):
            st.write(f"**{i}. {result['title']}** — cosine similarity: `{result['score']:.4f}`")
        answer = ExtractiveQAModel().answer(question, results[0]["content"])
        st.success(answer["answer"])
        st.write(f"Confidence: `{answer['score']:.4f}`")
    except Exception as exc:
        st.error(str(exc))
