from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_minimum_document_count():
    documents = list(DATA_DIR.glob("*.txt"))
    assert len(documents) >= 5


def test_each_document_has_two_paragraphs():
    for path in DATA_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8").strip()
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 2, path.name


def test_expected_topics_exist():
    names = {p.stem for p in DATA_DIR.glob("*.txt")}
    expected = {
        "existentialism",
        "ethics",
        "epistemology",
        "stoicism",
        "metaphysics",
    }
    assert expected.issubset(names)
