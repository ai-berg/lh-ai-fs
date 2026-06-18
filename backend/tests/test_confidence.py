"""Tests for deterministic confidence scoring — written before the implementation.

Confidence is derived from VERIFIABLE signals on a grounded finding, never a number
the model self-reports. The whole point (and the reason this is unit-testable with
no LLM) is that the same finding always yields the same score, traceable to the
signals that produced it — the "verified citations / traceable reasoning" posture
the product sells. These tests pin that the signals move the score the right way.
"""

from schemas import ConfidenceBand, EvidenceRef, Finding, FlagType, VerificationStatus
from services.confidence import score_confidence


def _finding(status, evidence_docs, flag_type=FlagType.CROSS_DOC_INCONSISTENCY):
    return Finding(
        msj_claim="The incident occurred on March 14, 2021.",
        comparison_reasoning="MSJ says March 14; references say March 12.",
        status=status,
        flag_type=flag_type,
        evidence=[EvidenceRef(source_doc=d, quote="March 12, 2021") for d in evidence_docs],
        explanation="Date conflict.",
        raised_by="CrossDocConsistencyAgent",
    )


def test_abstention_is_low_confidence():
    # A could_not_verify finding asserts nothing — confidence in a *flag* it isn't
    # making must be low, never high.
    c = score_confidence(_finding(VerificationStatus.COULD_NOT_VERIFY, []))
    assert c.band == ConfidenceBand.LOW
    assert c.value < 0.5


def test_single_source_contradiction_is_at_least_medium():
    # An assertive contradiction grounded in one reference document is a real flag.
    c = score_confidence(_finding(VerificationStatus.CONTRADICTED, ["police_report"]))
    assert c.band in (ConfidenceBand.MEDIUM, ConfidenceBand.HIGH)
    assert c.value >= 0.5


def test_multi_source_corroboration_beats_single_source():
    # The same contradiction confirmed by TWO documents is more certain than by one —
    # cross-corroboration is the strongest deterministic signal we have.
    one = score_confidence(_finding(VerificationStatus.CONTRADICTED, ["police_report"]))
    two = score_confidence(
        _finding(VerificationStatus.CONTRADICTED, ["police_report", "witness_statement"])
    )
    assert two.value > one.value


def test_high_confidence_requires_corroboration():
    # HIGH is reserved for an assertive, multi-source-corroborated flag — so the band
    # means something stronger than "the model sounded sure".
    c = score_confidence(
        _finding(VerificationStatus.CONTRADICTED,
                 ["police_report", "witness_statement", "medical_records_excerpt"])
    )
    assert c.band == ConfidenceBand.HIGH


def test_reasoning_is_populated_and_cites_the_signals():
    # The score must carry a human-readable reason DERIVED from the signals, not an
    # opaque float — that is what makes it auditable.
    c = score_confidence(_finding(VerificationStatus.CONTRADICTED, ["police_report"]))
    assert c.reasoning
    assert "1" in c.reasoning or "source" in c.reasoning.lower()


def test_value_is_bounded():
    for status in (VerificationStatus.CONTRADICTED, VerificationStatus.COULD_NOT_VERIFY):
        c = score_confidence(_finding(status, ["police_report", "witness_statement"]))
        assert 0.0 <= c.value <= 1.0
