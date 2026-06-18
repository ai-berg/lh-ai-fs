"""Tests for the grounding layer — written before the implementation (TDD).

Grounding is the anti-hallucination guarantee: a finding may only keep its
evidence if every quote literally exists in the cited source document. Anything
unverifiable collapses to ``could_not_verify`` with evidence cleared.
"""

from schemas import EvidenceRef, Finding, FlagType, VerificationStatus
from services.grounding import validate_grounding

DOCS = {
    "police_report": (
        "Site supervisor Mark Ellison confirmed that Rivera was wearing a hard "
        "hat and harness consistent with site requirements at the time of the "
        "collapse."
    ),
    "motion_for_summary_judgment": (
        "The incident occurred on March 14, 2021 at the West Olympic site."
    ),
}


def _finding(status, quotes):
    return Finding(
        flag_type=FlagType.FACTUAL_CONTRADICTION,
        msj_claim="Rivera was not wearing required PPE.",
        comparison_reasoning="MSJ says no PPE; police report says he wore a harness.",
        status=status,
        evidence=[EvidenceRef(source_doc=doc, quote=q) for doc, q in quotes],
        explanation="MSJ contradicts the police report.",
        raised_by="CrossDocConsistencyAgent",
    )


def test_keeps_finding_when_quote_exists_verbatim():
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "Rivera was wearing a hard hat and harness")],
    )

    result = validate_grounding(finding, DOCS)

    assert result.status == VerificationStatus.CONTRADICTED
    assert len(result.evidence) == 1


def test_tolerates_whitespace_and_case_differences():
    # The LLM may re-wrap whitespace or change case; normalization must forgive that.
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "RIVERA WAS WEARING a hard hat   and harness")],
    )

    result = validate_grounding(finding, DOCS)

    assert result.status == VerificationStatus.CONTRADICTED
    assert len(result.evidence) == 1


def test_collapses_to_could_not_verify_when_quote_absent():
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "Rivera was drunk and reckless")],  # not in the doc
    )

    result = validate_grounding(finding, DOCS)

    assert result.status == VerificationStatus.COULD_NOT_VERIFY
    assert result.evidence == []
    assert "grounding" in result.explanation.lower()


def test_partial_token_does_not_count_as_grounded():
    # "March 1" must NOT be considered present in "March 12, 2021": a quote that
    # only matches a prefix of a longer token is not literal grounding.
    docs = {"motion_for_summary_judgment": "The incident occurred on March 12, 2021."}
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("motion_for_summary_judgment", "March 1")],
    )

    result = validate_grounding(finding, docs)

    assert result.status == VerificationStatus.COULD_NOT_VERIFY
    assert result.evidence == []


def test_quote_followed_by_punctuation_is_still_grounded():
    # The grounded quote may end at a token boundary marked by punctuation.
    docs = {"police_report": "Date of Incident: March 12, 2021. Time: 10:45."}
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "March 12, 2021")],
    )

    result = validate_grounding(finding, docs)

    assert result.status == VerificationStatus.CONTRADICTED
    assert len(result.evidence) == 1


def test_does_not_mutate_the_input_finding():
    # Purity: collapsing an ungrounded finding must not touch the caller's object.
    original = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "this text is not in the document")],
    )

    result = validate_grounding(original, DOCS)

    assert original.status == VerificationStatus.CONTRADICTED  # unchanged
    assert len(original.evidence) == 1  # unchanged
    assert result.status == VerificationStatus.COULD_NOT_VERIFY  # the copy collapsed
    assert result is not original


def test_collapses_when_source_doc_unknown():
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("nonexistent_doc", "anything at all")],
    )

    result = validate_grounding(finding, DOCS)

    assert result.status == VerificationStatus.COULD_NOT_VERIFY
    assert result.evidence == []


def test_could_not_verify_without_evidence_is_left_untouched():
    finding = _finding(VerificationStatus.COULD_NOT_VERIFY, [])

    result = validate_grounding(finding, DOCS)

    assert result.status == VerificationStatus.COULD_NOT_VERIFY
    assert result.evidence == []


def test_could_not_verify_with_evidence_is_still_grounded():
    # A could_not_verify finding that nonetheless carries a quote must have that
    # quote verified — an ungrounded one is cleared so it can't leak to clients.
    finding = _finding(
        VerificationStatus.COULD_NOT_VERIFY,
        [("police_report", "this fabricated quote is not in the source")],
    )

    result = validate_grounding(finding, DOCS)

    assert result.evidence == []


def test_typographic_variants_still_ground():
    # Legal-PDF extraction swaps straight/curly quotes, hyphen/em-dash, "..."/…
    # The model may regenerate a quote with the other variant; normalization must
    # treat them as equal so a real quote isn't a false negative.
    docs = {
        "police_report": (
            'The supervisor — a "competent person" — confirmed it... fully.'
        )
    }
    # Quote uses straight quotes, a hyphen, and a literal "..." where the source
    # has curly quotes, an em-dash, and a unicode ellipsis.
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", 'a "competent person"')],
    )

    result = validate_grounding(finding, docs)

    assert result.status == VerificationStatus.CONTRADICTED
    assert len(result.evidence) == 1


def test_superscript_footnote_marker_does_not_fold_into_a_digit():
    # NFC (not NFKC): a footnote-marked source like "exceeded 90²" must NOT ground a
    # quote that swaps the superscript marker for a real digit ("exceeded 902").
    # Under NFKC the ² folds to 2 and this would falsely ground a fabricated number.
    docs = {"police_report": "Harmon exceeded 90² percent compliance."}
    fabricated = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "exceeded 902 percent")],
    )

    result = validate_grounding(fabricated, docs)

    assert result.status == VerificationStatus.COULD_NOT_VERIFY
    assert result.evidence == []


def test_pdf_ligatures_still_ground_under_nfc():
    # NFC keeps superscripts distinct (good) but, unlike NFKC, would drop ligature
    # folding; we re-add it explicitly. A model quote in plain "office fly" must
    # still ground against a PDF source that emitted ﬁ/ﬂ ligatures.
    docs = {"police_report": "The oﬃce reviewed the ﬂight log."}  # ﬃ, ﬂ ligatures
    finding = _finding(
        VerificationStatus.CONTRADICTED,
        [("police_report", "the office reviewed the flight log")],
    )

    result = validate_grounding(finding, docs)

    assert result.status == VerificationStatus.CONTRADICTED
    assert len(result.evidence) == 1


def test_could_not_verify_keeps_a_grounded_quote():
    # If the quote does exist, there's no reason to strip it.
    finding = _finding(
        VerificationStatus.COULD_NOT_VERIFY,
        [("police_report", "Rivera was wearing a hard hat and harness")],
    )

    result = validate_grounding(finding, DOCS)

    assert len(result.evidence) == 1
