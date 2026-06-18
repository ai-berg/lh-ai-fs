"""BS Detector API — analyzes a legal case file for citation and factual flaws."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from repositories.document_repository import load_documents
from schemas import VerificationReport
from services.orchestrator import EmptyCorpusError, run_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging at startup rather than import time, so it doesn't race or
    # duplicate handlers against Uvicorn's own logging setup. Only nudge the level
    # if the root logger is still at its unconfigured default, so an operator's
    # --log-level / explicit config is never overridden.
    root = logging.getLogger()
    if root.level == logging.WARNING:  # Python's default when untouched
        root.setLevel(logging.INFO)
    yield


app = FastAPI(title="BS Detector", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # localhost and 127.0.0.1 are distinct origins to a browser; allow both so
    # the UI works whether it's opened via either host.
    allow_origins=["http://localhost:5175", "http://127.0.0.1:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=VerificationReport)
async def analyze() -> VerificationReport:
    """Run the multi-agent verification pipeline over the case documents."""
    docs = load_documents()
    try:
        return await run_pipeline(docs)
    except EmptyCorpusError as exc:
        # No MSJ to audit is a bad-input condition, not a server fault — return 422
        # rather than a deceptively empty 200 report.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
