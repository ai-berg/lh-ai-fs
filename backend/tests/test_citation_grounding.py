"""Grounding for citation direct quotes — written before implementation (TDD).

A citation's quoted_text claims to be lifted from the MSJ. The grounding layer
is verify-only: if that text is not literally present in the MSJ, it withdraws
confidence (support -> could_not_verify) and annotates the issue, but it does NOT
invent a flag_type — categorizing the problem (e.g. quote_altered) is the agent's
judgment, not the verifier's.
"""

from schemas import Citation, FlagType, VerificationStatus
from services.grounding import ground_citation_quotes

MSJ = (
    "The court held in Privette that a hirer is generally not liable. "
    "The statute of limitations is two years."
)


# Default support is contradicted (a real assertive verdict that survives the schema's
# "no confirmable verified" ceiling) so these tests isolate ground_citation_quotes'
# own behavior — whether the QUOTE is present in the MSJ — not the verified-downgrade.
def _citation(quoted_text, support=VerificationStatus.CONTRADICTED):
    return Citation(
        authority="Privette v. Superior Court",
        proposition="A hirer is never liable.",
        is_direct_quote=quoted_text is not None,
        quoted_text=quoted_text,
        assessment_reasoning="r",
        support_assessment=support,
    )


def test_quote_present_in_msj_is_left_alone():
    # A quote that IS in the MSJ: the grounding pass must not touch the support verdict
    # (it only withdraws confidence when the quote is ABSENT).
    c = _citation("a hirer is generally not liable")
    [result] = ground_citation_quotes([c], MSJ)
    assert result.support_assessment == VerificationStatus.CONTRADICTED  # unchanged
    assert result.flag_type is None


def test_quote_absent_from_msj_withdraws_confidence_without_inventing_a_flag():
    c = _citation("a hirer is ALWAYS fully liable for everything")
    [result] = ground_citation_quotes([c], MSJ)

    # Verify-only: support is downgraded and the reason is annotated...
    assert result.support_assessment == VerificationStatus.COULD_NOT_VERIFY
    assert "msj" in (result.issue or "").lower()
    # ...but the verifier does not create a quote_altered finding itself.
    assert result.flag_type is None


def test_citation_without_quote_is_untouched():
    # A citation with no quoted_text has nothing for the quote-grounding pass to
    # check, so ground_citation_quotes must leave it as-is. (Base support is
    # could_not_verify here because the schema downgrades a quoteless 'verified'
    # on its own — that's a separate, tested invariant.)
    c = _citation(None, support=VerificationStatus.COULD_NOT_VERIFY)
    [result] = ground_citation_quotes([c], MSJ)
    assert result.support_assessment == VerificationStatus.COULD_NOT_VERIFY
    assert result.flag_type is None


def test_agents_own_flag_is_preserved():
    # If the agent already judged the quote altered, the verifier keeps that
    # finding; it only withdraws the support verdict it can't confirm.
    c = _citation("a hirer is ALWAYS fully liable")
    c.flag_type = FlagType.QUOTE_ALTERED
    [result] = ground_citation_quotes([c], MSJ)
    assert result.flag_type == FlagType.QUOTE_ALTERED
    assert result.support_assessment == VerificationStatus.COULD_NOT_VERIFY
