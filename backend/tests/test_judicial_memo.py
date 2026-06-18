"""Tests for the Judicial-Memo agent (LangChain LCEL synthesis).

The LangChain chain is monkeypatched so we test the agent's deterministic contract —
input selection (only confirmed, higher-confidence findings feed the memo), the
"no findings → no memo" short-circuit, and traceability (grounded_in) — without an
LLM call. The prose quality is exercised live in the eval/E2E.
"""

import pytest

import services.agents.judicial_memo as judicial_memo
from schemas import (
    ConfidenceBand,
    ConfidenceScore,
    EvidenceRef,
    Finding,
    FlagType,
    JudicialMemo,
    VerificationStatus,
)


def _finding(claim, band, status=VerificationStatus.CONTRADICTED):
    f = Finding(
        flag_type=FlagType.CROSS_DOC_INCONSISTENCY,
        msj_claim=claim,
        comparison_reasoning="r",
        status=status,
        evidence=[EvidenceRef(source_doc="police_report", quote="q")],
        explanation="e",
        raised_by="CrossDocConsistencyAgent",
    )
    return f.model_copy(update={"confidence": ConfidenceScore(value=0.8, band=band, reasoning="r")})


@pytest.mark.asyncio
async def test_no_findings_returns_no_memo(monkeypatch):
    # Nothing confirmed → no memo to write; must not invoke the chain.
    invoked = False

    async def fake_chain(payload):
        nonlocal invoked
        invoked = True
        return JudicialMemo(summary="should not happen")

    monkeypatch.setattr(judicial_memo, "_run_chain", fake_chain)

    memo = await judicial_memo.write_judicial_memo([])

    assert memo is None
    assert invoked is False


@pytest.mark.asyncio
async def test_abstentions_alone_produce_no_memo(monkeypatch):
    # A memo summarizes CONFIRMED problems; if everything is could_not_verify there is
    # nothing to report to the judge.
    monkeypatch.setattr(judicial_memo, "_run_chain",
                        lambda payload: (_ for _ in ()).throw(AssertionError("should not run")))

    memo = await judicial_memo.write_judicial_memo(
        [_finding("x", ConfidenceBand.LOW, status=VerificationStatus.COULD_NOT_VERIFY)]
    )

    assert memo is None


@pytest.mark.asyncio
async def test_memo_is_grounded_in_the_findings_it_used(monkeypatch):
    async def fake_chain(payload):
        # The agent fills grounded_in itself from the selected findings, so the memo
        # stays traceable regardless of what the model returns for it.
        return JudicialMemo(summary="The audit found a date contradiction.", grounded_in=[])

    monkeypatch.setattr(judicial_memo, "_run_chain", fake_chain)

    findings = [
        _finding("Incident date is March 14, 2021.", ConfidenceBand.HIGH),
        _finding("PPE was not worn.", ConfidenceBand.MEDIUM),
    ]
    memo = await judicial_memo.write_judicial_memo(findings)

    assert memo is not None
    assert set(memo.grounded_in) == {"Incident date is March 14, 2021.", "PPE was not worn."}
