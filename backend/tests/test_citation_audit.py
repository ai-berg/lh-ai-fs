"""Tests for the Citation Audit agent — the core of Tier 1.

The LLM call is monkeypatched so we test the agent's deterministic contract
(input wiring, output passthrough, empty-MSJ handling) without spending tokens
or depending on model behavior.
"""

import pytest

import services.agents.citation_audit as citation_audit
from schemas import Citation, CitationAuditOutput, FlagType, VerificationStatus


def _sample_output():
    return CitationAuditOutput(
        citations=[
            Citation(
                authority="Privette v. Superior Court",
                reporter="5 Cal.4th 689 (1993)",
                proposition="A hirer is never liable for a contractor's injuries.",
                is_direct_quote=True,
                quoted_text="A hirer is never liable",
                assessment_reasoning="r",
                support_assessment=VerificationStatus.COULD_NOT_VERIFY,
                flag_type=FlagType.OVERSTATEMENT,
                issue="Absolute 'never' claim is an overstatement.",
            ),
            Citation(
                authority="Cal. Code Civ. Proc. § 335.1",
                reporter=None,
                proposition="Two-year limitations period for personal injury.",
                is_direct_quote=False,
                quoted_text=None,
                assessment_reasoning="r",
                support_assessment=VerificationStatus.COULD_NOT_VERIFY,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_returns_citations_from_the_model(monkeypatch):
    async def fake_call(messages, schema, **kwargs):
        assert schema is CitationAuditOutput
        return _sample_output()

    monkeypatch.setattr(citation_audit, "call_llm_structured", fake_call)

    citations = await citation_audit.audit_citations(
        {"motion_for_summary_judgment": "... Privette v. Superior Court ..."}
    )

    assert len(citations) == 2
    # Problematic citation carries a machine-readable flag_type, not just prose.
    assert citations[0].flag_type == FlagType.OVERSTATEMENT
    # A statute is represented with a null reporter.
    assert citations[1].reporter is None


@pytest.mark.asyncio
async def test_msj_goes_in_user_role_instructions_in_system(monkeypatch):
    # Role split (injection defense): the untrusted MSJ must land in the user
    # message, never the system message that carries the instructions.
    captured = {}

    async def fake_call(messages, schema, **kwargs):
        captured["messages"] = messages
        return CitationAuditOutput(citations=[])

    monkeypatch.setattr(citation_audit, "call_llm_structured", fake_call)

    await citation_audit.audit_citations(
        {"motion_for_summary_judgment": "UNIQUE_MARKER_TEXT"}
    )

    by_role = {m["role"]: m["content"] for m in captured["messages"]}
    assert "UNIQUE_MARKER_TEXT" in by_role["user"]
    assert "UNIQUE_MARKER_TEXT" not in by_role["system"]
    assert "forensic legal auditor" in by_role["system"]  # instructions in system


@pytest.mark.asyncio
async def test_empty_msj_short_circuits_without_calling_the_model(monkeypatch):
    async def fake_call(messages, schema, **kwargs):
        raise AssertionError("must not call the model when the MSJ is missing")

    monkeypatch.setattr(citation_audit, "call_llm_structured", fake_call)

    assert await citation_audit.audit_citations({}) == []
