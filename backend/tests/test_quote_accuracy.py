"""Tests for the Quote-Accuracy agent.

The LLM is monkeypatched so we test the agent's deterministic contract — input
guards, provenance stamping, and that it passes the case-file documents as the
sources to compare against — without spending tokens. The substance (catching an
altered quote) is exercised in the eval harness against real model output.
"""

import pytest

import services.agents.quote_accuracy as quote_accuracy
from schemas import EvidenceRef, FindingDraft, FlagType, QuoteAccuracyOutput, VerificationStatus

DOCS = {
    "motion_for_summary_judgment": 'The report says he was "not wearing" his harness.',
    "police_report": "Rivera was wearing a hard hat and harness.",
    "witness_statement": "He had his harness on.",
}


def _altered_quote_draft():
    # The agent emits a FindingDraft (no confidence/raised_by — those are stamped/scored
    # downstream); the agent promotes it to a Finding with provenance.
    return FindingDraft(
        flag_type=FlagType.QUOTE_ALTERED,
        msj_claim='The report says he was "not wearing" his harness.',
        comparison_reasoning="MSJ quotes 'not wearing'; the report says he WAS wearing it.",
        status=VerificationStatus.CONTRADICTED,
        evidence=[EvidenceRef(source_doc="police_report", quote="Rivera was wearing a hard hat and harness")],
        explanation="The quotation inverts the source by inserting 'not'.",
    )


@pytest.mark.asyncio
async def test_stamps_provenance(monkeypatch):
    async def fake_call(messages, schema, **kwargs):
        return QuoteAccuracyOutput(findings=[_altered_quote_draft()])

    monkeypatch.setattr(quote_accuracy, "call_llm_structured", fake_call)

    findings = await quote_accuracy.check_quote_accuracy(DOCS)

    assert findings[0].raised_by == quote_accuracy.AGENT_NAME
    assert findings[0].flag_type == FlagType.QUOTE_ALTERED


@pytest.mark.asyncio
async def test_returns_empty_without_an_msj(monkeypatch):
    # No MSJ → nothing to check; must not call the model.
    called = False

    async def fake_call(messages, schema, **kwargs):
        nonlocal called
        called = True
        return QuoteAccuracyOutput(findings=[])

    monkeypatch.setattr(quote_accuracy, "call_llm_structured", fake_call)

    findings = await quote_accuracy.check_quote_accuracy({"police_report": "x"})

    assert findings == []
    assert called is False


@pytest.mark.asyncio
async def test_case_file_documents_are_passed_as_sources(monkeypatch):
    # The agent must hand the reference (case-file) documents to the model as the
    # sources to compare quotes against — not just the MSJ.
    seen = {}

    async def fake_call(messages, schema, **kwargs):
        seen["user"] = next(m["content"] for m in messages if m["role"] == "user")
        return QuoteAccuracyOutput(findings=[])

    monkeypatch.setattr(quote_accuracy, "call_llm_structured", fake_call)

    await quote_accuracy.check_quote_accuracy(DOCS)

    assert "police_report" in seen["user"] and "witness_statement" in seen["user"]
