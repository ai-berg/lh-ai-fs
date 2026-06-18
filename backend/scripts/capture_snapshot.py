"""Capture a real /analyze run into tests/fixtures/analyze_snapshot.json.

Run inside the backend container (needs OPENAI_API_KEY):
    docker compose exec backend python scripts/capture_snapshot.py

The committed snapshot lets the regression tests assert end-to-end behavior
without spending an API call on every run.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from repositories.document_repository import load_documents  # noqa: E402
from services.orchestrator import run_pipeline  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "analyze_snapshot.json"


def main() -> None:
    report = asyncio.run(run_pipeline(load_documents()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    d = report.model_dump()
    print(f"wrote {OUT} — citations={len(d['citations'])} flags={len(d['flags'])}")


if __name__ == "__main__":
    main()
