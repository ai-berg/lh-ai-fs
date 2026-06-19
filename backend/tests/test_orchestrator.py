"""Tests for the pipeline orchestrator — written before the implementation (TDD).

The orchestrator fans the agents out, grounds their findings against the corpus,
and assembles a VerificationReport. A failing agent must degrade gracefully:
recorded in ``degraded_agents``, never crashing the run.
"""

import pytest

from schemas import (
    Citation,
    EvidenceRef,
    Finding,
    FlagType,
    VerificationStatus,
)
from services.orchestrator import _dedupe_findings, run_pipeline


def _f(claim, flag_type, raised_by, status=VerificationStatus.CONTRADICTED):
    return Finding(
        flag_type=flag_type, msj_claim=claim, comparison_reasoning="r", status=status,
        evidence=[EvidenceRef(source_doc="d", quote="q")], explanation="e", raised_by=raised_by,
    )


def test_dedup_collapses_same_defect_despite_different_flag_types():
    # The real live-run case: the two agents are prompted to LABEL the same Section 7.2
    # edit differently — CrossDoc emits factual_contradiction (its prompt forbids
    # quote_altered), QuoteAccuracy emits quote_altered. Dedup must STILL collapse them
    # (keyed on the shared msj_claim, not flag_type) and attribute both agents.
    findings = [
        _f("Section 7.2 quoted without limitation", FlagType.FACTUAL_CONTRADICTION, "CrossDocConsistencyAgent"),
        _f("Section 7.2 quoted without limitation", FlagType.QUOTE_ALTERED, "QuoteAccuracyAgent"),
    ]
    out = _dedupe_findings(findings)
    assert len(out) == 1
    assert "CrossDocConsistencyAgent" in out[0].raised_by
    assert "QuoteAccuracyAgent" in out[0].raised_by


def test_dedup_keeps_distinct_defects():
    # Different claims (or different flag types) are NOT the same defect — keep both.
    findings = [
        _f("date is wrong", FlagType.CROSS_DOC_INCONSISTENCY, "CrossDocConsistencyAgent"),
        _f("quantity is wrong", FlagType.CROSS_DOC_INCONSISTENCY, "CrossDocConsistencyAgent"),
    ]
    assert len(_dedupe_findings(findings)) == 2

DOCS = {
    "motion_for_summary_judgment": "The incident occurred on March 14, 2021.",
    "police_report": "Date of Incident: March 12, 2021.",
}


def _good_citation():
    return Citation(
        authority="Privette v. Superior Court",
        reporter="5 Cal.4th 689 (1993)",
        proposition="A hirer is never liable for an independent contractor's injuries.",
        is_direct_quote=True,
        quoted_text="A hirer is never liable",
        assessment_reasoning="Absolute claim, no internal support.",
        support_assessment=VerificationStatus.COULD_NOT_VERIFY,
        issue="Absolute 'never' claim is an overstatement.",
    )


def _grounded_finding():
    return Finding(
        flag_type=FlagType.CROSS_DOC_INCONSISTENCY,
        msj_claim="The incident occurred on March 14, 2021.",
        comparison_reasoning="MSJ says March 14; police report says March 12.",
        status=VerificationStatus.CONTRADICTED,
        evidence=[EvidenceRef(source_doc="police_report", quote="March 12, 2021")],
        explanation="Police report states March 12, not March 14.",
        raised_by="CrossDocConsistencyAgent",
    )


async def _citation_agent_ok(docs):
    return [_good_citation()]


async def _cross_doc_agent_ok(docs):
    return [_grounded_finding()]


async def _quote_agent_empty(docs):
    # Default fake for the 4th fan-out agent: most tests don't exercise quote-accuracy,
    # so it returns nothing and stays out of the way.
    return []


async def _memo_noop(findings, citations=None):
    # Default fake memo: avoids a real LangChain/LLM call in the orchestrator tests.
    return None


async def _agent_boom(docs):
    raise RuntimeError("simulated agent failure")


@pytest.mark.asyncio
async def test_assembles_report_from_the_agents():
    report = await run_pipeline(
        DOCS,
        citation_agent=_citation_agent_ok,
        cross_doc_agent=_cross_doc_agent_ok,
        quote_accuracy_agent=_quote_agent_empty,
        memo_agent=_memo_noop,
    )

    assert len(report.citations) == 1
    assert len(report.flags) == 1
    assert report.degraded_agents == []
    # Confidence is now scored on every flag.
    assert report.flags[0].confidence is not None


@pytest.mark.asyncio
async def test_grounds_findings_before_reporting():
    # A finding citing text absent from the corpus must collapse to could_not_verify.
    async def hallucinating_agent(docs):
        return [
            Finding(
                flag_type=FlagType.FACTUAL_CONTRADICTION,
                msj_claim="X",
                comparison_reasoning="r",
                status=VerificationStatus.CONTRADICTED,
                evidence=[EvidenceRef(source_doc="police_report", quote="fabricated text")],
                explanation="not real",
                raised_by="CrossDocConsistencyAgent",
            )
        ]

    report = await run_pipeline(
        DOCS,
        citation_agent=_citation_agent_ok,
        cross_doc_agent=hallucinating_agent,
        quote_accuracy_agent=_quote_agent_empty,
        memo_agent=_memo_noop,
    )

    assert report.flags[0].status == VerificationStatus.COULD_NOT_VERIFY
    assert report.flags[0].evidence == []


@pytest.mark.asyncio
async def test_failed_agent_is_degraded_not_fatal():
    report = await run_pipeline(
        DOCS,
        citation_agent=_agent_boom,
        cross_doc_agent=_cross_doc_agent_ok,
        quote_accuracy_agent=_quote_agent_empty,
        memo_agent=_memo_noop,
    )

    # Pipeline still returns; the cross-doc flag survives; failure is recorded.
    assert len(report.flags) == 1
    assert report.citations == []
    assert "CitationAuditAgent" in report.degraded_agents


@pytest.mark.asyncio
async def test_all_fanout_agents_failing_still_returns_valid_report():
    report = await run_pipeline(
        DOCS,
        citation_agent=_agent_boom,
        cross_doc_agent=_agent_boom,
        quote_accuracy_agent=_agent_boom,
        memo_agent=_memo_noop,
    )

    assert report.citations == []
    assert report.flags == []
    # All three fan-out agents degraded; the pipeline still returns a valid report.
    assert len(report.degraded_agents) == 3


@pytest.mark.asyncio
async def test_memo_failure_degrades_gracefully():
    # A memo agent that raises must not sink the report — it's recorded and memo is None.
    async def _memo_boom(findings, citations=None):
        raise RuntimeError("memo failure")

    report = await run_pipeline(
        DOCS,
        citation_agent=_citation_agent_ok,
        cross_doc_agent=_cross_doc_agent_ok,
        quote_accuracy_agent=_quote_agent_empty,
        memo_agent=_memo_boom,
    )

    assert report.judicial_memo is None
    assert "JudicialMemoAgent" in report.degraded_agents
    assert len(report.flags) == 1  # the report still carries its findings


@pytest.mark.asyncio
async def test_empty_msj_raises_rather_than_returning_empty_report():
    # An empty/whitespace MSJ must NOT look like a clean audit that found nothing —
    # it's a bad-input condition the route turns into a 422.
    from services.orchestrator import EmptyCorpusError

    with pytest.raises(EmptyCorpusError):
        await run_pipeline(
            {"motion_for_summary_judgment": "   ", "police_report": "x"},
            citation_agent=_citation_agent_ok,
            cross_doc_agent=_cross_doc_agent_ok,
        )


def test_expected_min_citations_reads_env_at_call_site(monkeypatch):
    # The override must take effect at runtime, not be frozen at import.
    from services.orchestrator import _expected_min_citations

    monkeypatch.setenv("EXPECTED_MIN_CITATIONS", "3")
    assert _expected_min_citations() == 3
    monkeypatch.setenv("EXPECTED_MIN_CITATIONS", "not_a_number")
    assert _expected_min_citations() == 11  # falls back, doesn't crash
