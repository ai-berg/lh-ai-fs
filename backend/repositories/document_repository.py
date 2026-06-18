"""Repository for the case-file documents (the data-access layer)."""

from pathlib import Path

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"

# The document under audit. Single source of truth so agents and the orchestrator
# don't each re-declare the corpus contract.
MSJ_DOC = "motion_for_summary_judgment"


def load_documents() -> dict[str, str]:
    """Load every .txt case document as ``{stem: content}``."""
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(DOCUMENTS_DIR.glob("*.txt"))
    }
