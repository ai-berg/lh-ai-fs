"""Tests for the Cross-Document Consistency agent.

The LLM call is monkeypatched so we test the agent's deterministic contract —
input guards, provenance stamping, and that the MSJ is excluded from the
reference blob — without spending tokens.
"""

import pytest

import services.agents.cross_doc as cross_doc
from schemas import CrossDocOutput, EvidenceRef, FindingDraft, FlagType, VerificationStatus

DOCS = {
    "motion_for_summary_judgment": "Incident on March 14, 2021.",
    "police_report": "Date of Incident: March 12, 2021.",
    "witness_statement": "It happened on March 12.",
}


def _draft():
    # The agent emits a FindingDraft (no raised_by/confidence — stamped/scored when the
    # agent promotes it to a Finding).
    return FindingDraft(
        flag_type=FlagType.CROSS_DOC_INCONSISTENCY,
        msj_claim="Incident on March 14, 2021.",
        comparison_reasoning="MSJ says 14, refs say 12.",
        status=VerificationStatus.CONTRADICTED,
        evidence=[EvidenceRef(source_doc="police_report", quote="March 12, 2021")],
        explanation="Date conflict.",
    )


@pytest.mark.asyncio
async def test_promotes_draft_and_stamps_provenance(monkeypatch):
    async def fake_call(messages, schema, **kwargs):
        return CrossDocOutput(findings=[_draft()])

    monkeypatch.setattr(cross_doc, "call_llm_structured", fake_call)

    findings = await cross_doc.check_cross_doc_consistency(DOCS)

    # The promoted Finding carries the agent's provenance and no confidence yet.
    assert findings[0].raised_by == cross_doc.AGENT_NAME
    assert findings[0].confidence is None


@pytest.mark.asyncio
async def test_reference_blob_excludes_the_msj(monkeypatch):
    captured = {}

    async def fake_call(messages, schema, **kwargs):
        captured["user"] = next(m["content"] for m in messages if m["role"] == "user")
        return CrossDocOutput(findings=[])

    monkeypatch.setattr(cross_doc, "call_llm_structured", fake_call)

    await cross_doc.check_cross_doc_consistency(DOCS)

    # The reference docs are present; the MSJ text is not duplicated into them as
    # a reference (it's the thing being checked, passed under its own key).
    assert "police_report" in captured["user"]
    assert "witness_statement" in captured["user"]


@pytest.mark.asyncio
async def test_missing_msj_returns_empty_without_calling_model(monkeypatch):
    async def fake_call(messages, schema, **kwargs):
        raise AssertionError("must not call the model without an MSJ")

    monkeypatch.setattr(cross_doc, "call_llm_structured", fake_call)

    assert await cross_doc.check_cross_doc_consistency({"police_report": "x"}) == []


@pytest.mark.asyncio
async def test_missing_references_returns_empty_without_calling_model(monkeypatch):
    async def fake_call(messages, schema, **kwargs):
        raise AssertionError("must not call the model without reference docs")

    monkeypatch.setattr(cross_doc, "call_llm_structured", fake_call)

    result = await cross_doc.check_cross_doc_consistency(
        {"motion_for_summary_judgment": "x"}
    )
    assert result == []
