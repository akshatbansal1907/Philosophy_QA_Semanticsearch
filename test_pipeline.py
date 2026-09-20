from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def test_minimum_documents():
    assert len(list(DATA_DIR.glob("*.txt"))) >= 5

def test_two_paragraphs_per_document():
    for path in DATA_DIR.glob("*.txt"):
        paragraphs = [p for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        assert len(paragraphs) >= 2
