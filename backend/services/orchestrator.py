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
from services.confidence import score_confidence
from services.grounding import ground_citation_quotes, normalize, validate_grounding

logger = logging.getLogger(__name__)

# Fixture-specific heuristic: the sample Rivera MSJ cites 11 authorities. Used
# ONLY to log a soft warning on a suspiciously low extraction; it never fails the
# request, and is the one sample-coupled constant here (overridable via env).
def _expected_min_citations() -> int:
    # Parse defensively: a malformed override must not crash (this value only gates a
    # soft warning), so fall back to the default. Read at the CALL SITE (not cached
    # at import) so a runtime/test env override — e.g. monkeypatch.setenv — actually
    # takes effect; a module-level constant would freeze the import-time value.
    try:
        return int(os.getenv("EXPECTED_MIN_CITATIONS", "11"))
    except ValueError:
        logger.warning("invalid EXPECTED_MIN_CITATIONS override; using default 11")
        return 11


class EmptyCorpusError(ValueError):
    """No usable MSJ in the corpus — the pipeline has nothing to audit.

    Raised (not silently returning an empty report) so a missing/empty MSJ is
    distinguishable from a clean audit that genuinely found nothing. The route
    surfaces it as a 4xx instead of returning a deceptively empty 200.
    """


CitationAgent = Callable[[dict[str, str]], Awaitable[list[Citation]]]
FindingAgent = Callable[[dict[str, str]], Awaitable[list[Finding]]]


async def run_agents(
    docs: dict[str, str],
    *,
    citation_agent: CitationAgent | None = None,
    cross_doc_agent: FindingAgent | None = None,
    quote_accuracy_agent: FindingAgent | None = None,
) -> tuple[list[Citation], list[Finding], list[str]]:
    """Run the fan-out agents concurrently and return their RAW (pre-grounding) output.

    THREE finding-producing agents now fan out in parallel (cross-doc + quote-accuracy
    both emit Findings; citation emits Citations). They are independent — none consumes
    another's output — so asyncio.gather isolates failures and there is no error-
    amplification path between them. The cross-doc and quote-accuracy findings are
    merged into one list because they share the Finding contract and flow through the
    same grounding gate; provenance (`raised_by`) preserves which agent found what.

    Split out from ``run_pipeline`` so the eval harness can measure the model's
    pre-gate fabrication propensity — the findings before the grounding gate clears
    unverifiable quotes — and compare it against the post-gate report.
    """
    if citation_agent is None or cross_doc_agent is None or quote_accuracy_agent is None:
        # Imported lazily so tests can inject fakes without an OpenAI client.
        from services.agents.citation_audit import audit_citations
        from services.agents.cross_doc import check_cross_doc_consistency
        from services.agents.quote_accuracy import check_quote_accuracy

        citation_agent = citation_agent or audit_citations
        cross_doc_agent = cross_doc_agent or check_cross_doc_consistency
        quote_accuracy_agent = quote_accuracy_agent or check_quote_accuracy

    degraded: list[str] = []
    citations, cross_doc_findings, quote_findings = await asyncio.gather(
        run_agent("CitationAuditAgent", lambda: citation_agent(docs), fallback=[], degraded=degraded),
        run_agent("CrossDocConsistencyAgent", lambda: cross_doc_agent(docs), fallback=[], degraded=degraded),
        run_agent("QuoteAccuracyAgent", lambda: quote_accuracy_agent(docs), fallback=[], degraded=degraded),
    )
    # One finding stream out: both agents emit Findings through the same gate; raised_by
    # keeps them attributable.
    return citations, [*cross_doc_findings, *quote_findings], degraded


def apply_grounding(
    citations: list[Citation], findings: list[Finding], docs: dict[str, str]
) -> tuple[list[Citation], list[Finding]]:
    """Apply the grounding gate to raw agent output (the post-gate transform)."""
    grounded = [validate_grounding(f, docs) for f in findings]
    citations = ground_citation_quotes(citations, docs.get(MSJ_DOC, ""))
    return citations, grounded


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Collapse co-located findings that two agents raised about the SAME defect.

    WHY this is needed: CrossDoc and QuoteAccuracy genuinely overlap on the highest-
    value case — a quotation edited to drop a limiting clause is *both* an unfaithful
    quote (QuoteAccuracy's lane) *and* the factual conflict that creates (CrossDoc's
    lane). On the committed synthetic run both agents flag the same Section 7.2 edit,
    so without dedup a judge would see TWO findings for ONE defect — inflating the
    count, which for a calibrated-counts product is a credibility failure.

    Dedup key: normalized msj_claim ALONE — deliberately NOT flag_type. The two agents
    are prompted to LABEL the same edit differently (CrossDoc may call it
    `factual_contradiction` per its prompt; QuoteAccuracy calls it `quote_altered`), so
    keying on flag_type would let the same defect through twice under different labels —
    which is exactly the live-run failure mode. The shared signal that it's ONE defect
    is the MSJ claim under scrutiny, so we collapse on the normalized claim. Nor the
    evidence span, since the two agents quote different spans (the altered MSJ text vs
    the original source clause). The FIRST finding wins (stable order: cross-doc runs
    first); the survivor records BOTH agents in `raised_by` so attribution isn't lost.
    """
    seen: dict[str, Finding] = {}
    order: list[str] = []
    for f in findings:
        key = normalize(f.msj_claim)
        if key in seen:
            prior = seen[key]
            others = {a.strip() for a in prior.raised_by.split("+")}
            if f.raised_by and f.raised_by not in others:
                seen[key] = prior.model_copy(update={"raised_by": f"{prior.raised_by}+{f.raised_by}"})
            continue
        seen[key] = f
        order.append(key)
    return [seen[k] for k in order]


async def run_pipeline(
    docs: dict[str, str],
    *,
    citation_agent: CitationAgent | None = None,
    cross_doc_agent: FindingAgent | None = None,
    quote_accuracy_agent: FindingAgent | None = None,
    memo_agent: Callable[[list[Finding]], Awaitable[object]] | None = None,
) -> VerificationReport:
    """Run the full pipeline and return a grounded, confidence-scored report with memo."""
    # Guard the input BEFORE running agents: an empty/whitespace MSJ would make the
    # agents return [] for benign reasons, producing a report indistinguishable from a
    # clean audit (degraded_agents would be empty). Fail loudly instead.
    if not (docs.get(MSJ_DOC) or "").strip():
        raise EmptyCorpusError(f"no MSJ found under '{MSJ_DOC}' in the provided corpus")

    citations, findings, degraded = await run_agents(
        docs,
        citation_agent=citation_agent,
        cross_doc_agent=cross_doc_agent,
        quote_accuracy_agent=quote_accuracy_agent,
    )
    citations, grounded = apply_grounding(citations, findings, docs)

    # Dedup BEFORE scoring/memo: collapse the same defect raised by two agents (the
    # CrossDoc/QuoteAccuracy overlap on meaning-altering quote edits) so it is counted,
    # scored, and reported to the judge ONCE, with both agents attributed.
    deduped = _dedupe_findings(grounded)

    # Confidence is scored AFTER grounding, on purpose: a finding the gate downgraded to
    # could_not_verify must score as a low-confidence non-claim, so confidence reflects
    # the GROUNDED truth, not the model's raw assertion. Deterministic — no LLM here.
    scored = [f.model_copy(update={"confidence": score_confidence(f)}) for f in deduped]

    # The memo synthesizes the confirmed findings AND flagged citations into one
    # paragraph for a judge. It runs LAST (it consumes the others' output) and degrades
    # gracefully: a memo failure records the agent but never sinks the report — the
    # structured flags remain the source of truth. Passing citations matters: a brief
    # whose only defect is a bad authority must still produce a memo for the judge.
    memo = await _run_memo(scored, citations, degraded, memo_agent)

    expected_min = _expected_min_citations()
    if "CitationAuditAgent" not in degraded and len(citations) < expected_min:
        logger.warning(
            "citation_count_below_expected",
            extra={"got": len(citations), "expected_min": expected_min},
        )

    return VerificationReport(
        citations=citations, flags=scored, judicial_memo=memo, degraded_agents=degraded
    )


async def _run_memo(findings, citations, degraded, memo_agent):
    """Run the memo agent under the SAME resilience contract as the fan-out agents.

    Routed through ``run_agent`` so the memo gets the identical timeout + one-retry +
    degraded-tracking the other agents have — previously it had a bare try/except with
    NO timeout, so a hung memo call could pin the whole /analyze request. The memo runs
    sequentially (it consumes the others' output), so its bounded slice adds on top of
    the fan-out, not in parallel.
    """
    if memo_agent is None:
        from services.agents.judicial_memo import write_judicial_memo

        memo_agent = write_judicial_memo
    return await run_agent(
        "JudicialMemoAgent", lambda: memo_agent(findings, citations), fallback=None, degraded=degraded
    )
