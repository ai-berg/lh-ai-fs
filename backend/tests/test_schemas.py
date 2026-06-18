"""Schema-level invariants — written before the implementation (TDD).

A Finding that claims a verified/contradicted status without any evidence is
internally inconsistent. Rather than reject it (the grounding pass runs later
and may legitimately clear evidence), the model normalizes it to
``could_not_verify`` — a fail-safe second net mirroring the grounding layer.
"""

from schemas import EvidenceRef, Finding, FlagType, VerificationStatus


def _finding(status, evidence):
    return Finding(
        flag_type=FlagType.FACTUAL_CONTRADICTION,
        msj_claim="X",
        comparison_reasoning="reasoning",
        status=status,
        evidence=evidence,
        explanation="...",
        raised_by="Test",
    )


def test_contradicted_without_evidence_is_normalized_to_could_not_verify():
    f = _finding(VerificationStatus.CONTRADICTED, [])
    assert f.status == VerificationStatus.COULD_NOT_VERIFY


def test_verified_without_evidence_is_normalized():
    f = _finding(VerificationStatus.VERIFIED, [])
    assert f.status == VerificationStatus.COULD_NOT_VERIFY


def test_contradicted_with_evidence_is_kept():
    f = _finding(
        VerificationStatus.CONTRADICTED,
        [EvidenceRef(source_doc="police_report", quote="March 12, 2021")],
    )
    assert f.status == VerificationStatus.CONTRADICTED


def test_could_not_verify_without_evidence_is_fine():
    f = _finding(VerificationStatus.COULD_NOT_VERIFY, [])
    assert f.status == VerificationStatus.COULD_NOT_VERIFY
