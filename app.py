import streamlit as st

from semantic_search import SemanticSearchEngine
from qa import ExtractiveQAModel


st.set_page_config(
    page_title="Philosophy Semantic Search + QA",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Philosophy Semantic Search + Extractive QA")
st.caption("Experiment 7 | Roll No. 09 | Topic: Philosophy")

st.write(
    "Ask a natural-language question. The system first finds the most "
    "semantically relevant philosophy document using sentence embeddings "
    "and cosine similarity, then extracts an answer from that document."
)


@st.cache_resource
def load_search_engine():
    return SemanticSearchEngine()


@st.cache_resource
def load_qa_model():
    return ExtractiveQAModel()


engine = load_search_engine()

question = st.text_input(
    "Enter your question",
    placeholder="Example: What does Stoicism teach about controlling emotions?",
)

top_k = st.slider("Number of semantic search results", 1, 5, 3)

if st.button("Search & Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching using sentence embeddings..."):
        results = engine.search(question, top_k=top_k)

    st.subheader("1. Semantic Search Results")

    for i, result in enumerate(results, 1):
        with st.container(border=True):
            st.write(f"**{i}. {result['title']}**")
            st.write(f"Cosine similarity: `{result['score']:.4f}`")
            preview = result["content"].replace("\n\n", " ")
            st.write(preview[:400] + ("..." if len(preview) > 400 else ""))

    best = results[0]

    st.subheader("2. Extractive Question Answering")
    st.info(f"Most relevant document: **{best['title']}**")

    with st.spinner("Extracting the answer..."):
        qa_model = load_qa_model()
        answer = qa_model.answer(question, best["content"])

    if answer["answer"]:
        st.success(answer["answer"])
        st.write(f"QA confidence: `{answer['score']:.4f}`")
    else:
        st.warning("The QA model could not extract a confident answer span.")

st.divider()
st.subheader("How it works")
st.markdown(
    """
**Step 1:** Convert all philosophy documents into dense sentence embeddings.

**Step 2:** Convert the user's question into the same embedding space.

**Step 3:** Calculate cosine similarity between the question vector and each document vector.

**Step 4:** Select the document with the highest semantic similarity.

**Step 5:** Give that document and the question to an extractive QA model.

**Step 6:** Return an exact span from the selected document as the answer.
"""
)
