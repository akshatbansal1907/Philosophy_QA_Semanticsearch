from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REQUIRED_DOCUMENTS = {
    "epistemology.txt",
    "ethics.txt",
    "existentialism.txt",
    "metaphysics.txt",
    "stoicism.txt",
}


def test_minimum_documents():
    assert REQUIRED_DOCUMENTS.issubset(
        {path.name for path in DATA_DIR.glob("*.txt")}
    )


def test_two_paragraphs_per_document():
    for filename in REQUIRED_DOCUMENTS:
        path = DATA_DIR / filename
        paragraphs = [
            p.strip()
            for p in path.read_text(encoding="utf-8").split("\n\n")
            if p.strip()
        ]
        assert len(paragraphs) >= 2
