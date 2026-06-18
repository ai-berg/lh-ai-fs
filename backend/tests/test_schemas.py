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


def test_citation_verified_without_quote_is_downgraded():
    from schemas import Citation

    c = Citation(
        authority="Whitmore v. Delgado",
        proposition="A hirer is not liable.",
        is_direct_quote=False,
        quoted_text=None,
        assessment_reasoning="r",
        support_assessment=VerificationStatus.VERIFIED,
    )
    # No quote to stand on -> can't claim verified for an unlookup-able authority.
    assert c.support_assessment == VerificationStatus.COULD_NOT_VERIFY


def test_citation_verified_with_a_quote_is_kept():
    from schemas import Citation

    c = Citation(
        authority="Whitmore v. Delgado",
        proposition="A hirer is not liable.",
        is_direct_quote=True,
        quoted_text="a hirer is not liable",
        assessment_reasoning="r",
        support_assessment=VerificationStatus.VERIFIED,
    )
    assert c.support_assessment == VerificationStatus.VERIFIED
