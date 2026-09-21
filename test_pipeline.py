from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REQUIRED_DOCUMENTS = {"epistemology.txt", "ethics.txt", "existentialism.txt", "metaphysics.txt", "stoicism.txt"}


def test_documents_exist():
    assert REQUIRED_DOCUMENTS.issubset({p.name for p in DATA_DIR.glob("*.txt")})


def test_two_paragraphs_per_document():
    for filename in REQUIRED_DOCUMENTS:
        paragraphs = [p.strip() for p in (DATA_DIR / filename).read_text(encoding="utf-8").split("\n\n") if p.strip()]
        assert len(paragraphs) >= 2
