#!/usr/bin/env python
"""Repo-root shim so the documented `python run_evals.py` works from anywhere.

Delegates to the real harness at backend/eval/run_evals.py. Needs the backend
deps (pyyaml, pydantic, openai); inside the container or a venv with
`pip install -r backend/requirements.txt`.
"""

import runpy
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND))
runpy.run_path(str(BACKEND / "eval" / "run_evals.py"), run_name="__main__")
