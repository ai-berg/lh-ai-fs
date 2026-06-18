"""Pipeline orchestration for the BS Detector.

Fans the agents out concurrently, grounds every finding against the corpus, and
assembles a ``VerificationReport``. Agents are injected so the orchestrator is
testable without any LLM calls; ``run_pipeline`` provides the real agents by
default.
"""

import asyncio
import logging
from typing import Awaitable, Callable

from schemas import Citation, Finding, VerificationReport
from services.agents.base import run_agent
from services.grounding import ground_citation_quotes, validate_grounding

logger = logging.getLogger(__name__)

MSJ_DOC = "motion_for_summary_judgment"
# Known authority count in the sample MSJ; a soft floor used only to warn on a
# suspiciously low extraction, never to fail the request.
EXPECTED_MIN_CITATIONS = 11

CitationAgent = Callable[[dict[str, str]], Awaitable[list[Citation]]]
CrossDocAgent = Callable[[dict[str, str]], Awaitable[list[Finding]]]


async def run_pipeline(
    docs: dict[str, str],
    *,
    citation_agent: CitationAgent | None = None,
    cross_doc_agent: CrossDocAgent | None = None,
) -> VerificationReport:
    """Run the Phase 1 pipeline and return a grounded VerificationReport."""
    if citation_agent is None or cross_doc_agent is None:
        # Imported lazily so tests can inject fakes without an OpenAI client.
        from services.agents.citation_audit import audit_citations
        from services.agents.cross_doc import check_cross_doc_consistency

        citation_agent = citation_agent or audit_citations
        cross_doc_agent = cross_doc_agent or check_cross_doc_consistency

    degraded: list[str] = []

    citations, findings = await asyncio.gather(
        run_agent(
            "CitationAuditAgent",
            lambda: citation_agent(docs),
            fallback=[],
            degraded=degraded,
        ),
        run_agent(
            "CrossDocConsistencyAgent",
            lambda: cross_doc_agent(docs),
            fallback=[],
            degraded=degraded,
        ),
    )

    # Ground both paths: cross-doc findings against their source docs, and
    # citation direct quotes against the MSJ itself.
    grounded = [validate_grounding(f, docs) for f in findings]
    citations = ground_citation_quotes(citations, docs.get(MSJ_DOC, ""))

    if "CitationAuditAgent" not in degraded and len(citations) < EXPECTED_MIN_CITATIONS:
        logger.warning(
            "citation_count_below_expected",
            extra={"got": len(citations), "expected_min": EXPECTED_MIN_CITATIONS},
        )

    return VerificationReport(
        citations=citations,
        flags=grounded,
        degraded_agents=degraded,
    )
