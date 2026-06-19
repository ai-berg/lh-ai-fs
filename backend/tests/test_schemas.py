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


def test_any_verified_citation_is_downgraded_no_caselaw_lookup():
    from schemas import Citation

    # HONEST CEILING: with no case-law lookup, "verified" is never confirmable, so it
    # always fails safe to could_not_verify — whether the citation is a direct quote
    # WITH text (the Kellerman seam the snapshot exposed: a quote existing in the brief
    # doesn't prove the case says it) or a paraphrase.
    quoted = Citation(
        authority="Kellerman v. Pacific Coast Construction",
        proposition="OSHA compliance creates a presumption of care.",
        is_direct_quote=True,
        quoted_text="Where an employer demonstrates full compliance...",
        assessment_reasoning="r",
        support_assessment=VerificationStatus.VERIFIED,
    )
    paraphrase = Citation(
        authority="Privette v. Superior Court",
        proposition="A hirer is presumptively not liable.",
        is_direct_quote=False,
        quoted_text=None,
        assessment_reasoning="The brief states the doctrine accurately.",
        support_assessment=VerificationStatus.VERIFIED,
    )
    assert quoted.support_assessment == VerificationStatus.COULD_NOT_VERIFY
    assert paraphrase.support_assessment == VerificationStatus.COULD_NOT_VERIFY


# (removed test_citation_verified_with_a_quote_is_kept: it encoded the pre-honest-
# ceiling behavior. With no case-law lookup, no citation can be confirmably `verified`,
# so a quote no longer "keeps" a verified verdict — see
# test_any_verified_citation_is_downgraded_no_caselaw_lookup above.)
