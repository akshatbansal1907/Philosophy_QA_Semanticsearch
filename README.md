# Philosophy QA Semantic Search

## Experiment 7 — Semantic Search + Extractive QA

**Roll No.: 09**  
**Topic: Philosophy**

This project implements the Experiment 7 requirements using:
- True sentence embeddings with `sentence-transformers`
- Cosine similarity for semantic document retrieval
- Extractive Question Answering with a pretrained Hugging Face QA model
- 5 philosophy documents, each containing at least 2 paragraphs
- A Streamlit interface for interactive questions

### Pipeline

```text
User Question
      |
      v
Sentence Transformer
      |
      v
Question Embedding
      |
      v
Cosine Similarity
      |
      v
Most Relevant Philosophy Document
      |
      v
Extractive QA Model
      |
      v
Exact Answer Span
```

## Project Structure

```text
Philosophy_QA_Semanticsearch/
├── app.py
├── semantic_search.py
├── qa.py
├── run.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── Experiment_7_Philosophy_Semantic_Search_QA.ipynb
├── data/
│   ├── existentialism.txt
│   ├── ethics.txt
│   ├── epistemology.txt
│   ├── stoicism.txt
│   └── metaphysics.txt
└── tests/
    └── test_pipeline.py
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

The first run downloads the pretrained embedding and QA models from Hugging Face.

## Run from Terminal

```bash
python run.py
```

## Example Questions

- What is existentialism concerned with?
- What is the difference between knowledge and belief?
- What does Stoicism teach about controlling emotions?
- What is virtue ethics?
- What is metaphysics concerned with?
- How does existentialism describe human freedom?
- What is the role of reason in Stoicism?

## Important Requirement

This project uses **sentence embeddings**, not Bag-of-Words or TF-IDF. The question and documents are converted into dense vectors, and cosine similarity determines semantic relevance.

## Academic Note

The documents in `data/` are concise educational summaries prepared for this experiment. They are intended as the project's local document collection, not as original philosophical scholarship.
