"""Repository for the case-file documents (the data-access layer)."""

from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"


def load_documents() -> dict[str, str]:
    """Load every .txt case document as ``{stem: content}``."""
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(DOCUMENTS_DIR.glob("*.txt"))
    }
