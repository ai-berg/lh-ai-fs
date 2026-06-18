"""Pipeline orchestration for the BS Detector.

Fans the agents out concurrently, grounds every finding against the corpus, and
assembles a ``VerificationReport``. Agents are injected so the orchestrator is
testable without any LLM calls; ``run_pipeline`` provides the real agents by
default.
"""

import asyncio
import logging
import os
from typing import Awaitable, Callable

from repositories.document_repository import MSJ_DOC
from schemas import Citation, Finding, VerificationReport
from services.agents.base import run_agent
from services.grounding import ground_citation_quotes, validate_grounding

logger = logging.getLogger(__name__)

# Fixture-specific heuristic: the sample Rivera MSJ cites 11 authorities. Used
# ONLY to log a soft warning on a suspiciously low extraction; it never fails the
# request, and is the one sample-coupled constant here (overridable via env).
EXPECTED_MIN_CITATIONS = int(os.getenv("EXPECTED_MIN_CITATIONS", "11"))

CitationAgent = Callable[[dict[str, str]], Awaitable[list[Citation]]]
CrossDocAgent = Callable[[dict[str, str]], Awaitable[list[Finding]]]


async def run_agents(
    docs: dict[str, str],
    *,
    citation_agent: CitationAgent | None = None,
    cross_doc_agent: CrossDocAgent | None = None,
) -> tuple[list[Citation], list[Finding], list[str]]:
    """Run the agents and return their RAW (pre-grounding) output.

    Split out from ``run_pipeline`` so the eval harness can measure the model's
    pre-gate fabrication propensity — the findings before the grounding gate
    clears unverifiable quotes — and compare it against the post-gate report.
    """
    if citation_agent is None or cross_doc_agent is None:
        # Imported lazily so tests can inject fakes without an OpenAI client.
        from services.agents.citation_audit import audit_citations
        from services.agents.cross_doc import check_cross_doc_consistency

        citation_agent = citation_agent or audit_citations
        cross_doc_agent = cross_doc_agent or check_cross_doc_consistency

    degraded: list[str] = []
    citations, findings = await asyncio.gather(
        run_agent("CitationAuditAgent", lambda: citation_agent(docs), fallback=[], degraded=degraded),
        run_agent("CrossDocConsistencyAgent", lambda: cross_doc_agent(docs), fallback=[], degraded=degraded),
    )
    return citations, findings, degraded


def apply_grounding(
    citations: list[Citation], findings: list[Finding], docs: dict[str, str]
) -> tuple[list[Citation], list[Finding]]:
    """Apply the grounding gate to raw agent output (the post-gate transform)."""
    grounded = [validate_grounding(f, docs) for f in findings]
    citations = ground_citation_quotes(citations, docs.get(MSJ_DOC, ""))
    return citations, grounded


async def run_pipeline(
    docs: dict[str, str],
    *,
    citation_agent: CitationAgent | None = None,
    cross_doc_agent: CrossDocAgent | None = None,
) -> VerificationReport:
    """Run the pipeline and return a grounded VerificationReport."""
    citations, findings, degraded = await run_agents(
        docs, citation_agent=citation_agent, cross_doc_agent=cross_doc_agent
    )
    citations, grounded = apply_grounding(citations, findings, docs)

    if "CitationAuditAgent" not in degraded and len(citations) < EXPECTED_MIN_CITATIONS:
        logger.warning(
            "citation_count_below_expected",
            extra={"got": len(citations), "expected_min": EXPECTED_MIN_CITATIONS},
        )

    return VerificationReport(citations=citations, flags=grounded, degraded_agents=degraded)
