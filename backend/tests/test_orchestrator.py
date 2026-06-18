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
from services.orchestrator import run_pipeline

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


async def _agent_boom(docs):
    raise RuntimeError("simulated agent failure")


@pytest.mark.asyncio
async def test_assembles_report_from_both_agents():
    report = await run_pipeline(
        DOCS,
        citation_agent=_citation_agent_ok,
        cross_doc_agent=_cross_doc_agent_ok,
    )

    assert len(report.citations) == 1
    assert len(report.flags) == 1
    assert report.degraded_agents == []


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
    )

    assert report.flags[0].status == VerificationStatus.COULD_NOT_VERIFY
    assert report.flags[0].evidence == []


@pytest.mark.asyncio
async def test_failed_agent_is_degraded_not_fatal():
    report = await run_pipeline(
        DOCS,
        citation_agent=_agent_boom,
        cross_doc_agent=_cross_doc_agent_ok,
    )

    # Pipeline still returns; the cross-doc flag survives; failure is recorded.
    assert len(report.flags) == 1
    assert report.citations == []
    assert "CitationAuditAgent" in report.degraded_agents


@pytest.mark.asyncio
async def test_all_agents_failing_still_returns_valid_report():
    report = await run_pipeline(
        DOCS,
        citation_agent=_agent_boom,
        cross_doc_agent=_agent_boom,
    )

    assert report.citations == []
    assert report.flags == []
    assert len(report.degraded_agents) == 2
