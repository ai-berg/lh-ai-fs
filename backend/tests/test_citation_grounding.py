"""Grounding for citation direct quotes — written before implementation (TDD).

A citation's quoted_text claims to be lifted from the MSJ. If that text is not
literally present in the MSJ, the quote may be altered; we flag it rather than
silently trusting it. This extends the anti-hallucination guarantee to the
Tier 1 citation path (cross-doc findings were already grounded).
"""

from schemas import Citation, FlagType, VerificationStatus
from services.grounding import ground_citation_quotes

MSJ = (
    "The court held in Privette that a hirer is generally not liable. "
    "The statute of limitations is two years."
)


def _citation(quoted_text):
    return Citation(
        authority="Privette v. Superior Court",
        proposition="A hirer is never liable.",
        is_direct_quote=quoted_text is not None,
        quoted_text=quoted_text,
        assessment_reasoning="r",
        support_assessment=VerificationStatus.COULD_NOT_VERIFY,
    )


def test_quote_present_in_msj_is_left_alone():
    c = _citation("a hirer is generally not liable")
    [result] = ground_citation_quotes([c], MSJ)
    assert result.flag_type is None


def test_quote_absent_from_msj_is_flagged_as_altered():
    c = _citation("a hirer is ALWAYS fully liable for everything")
    [result] = ground_citation_quotes([c], MSJ)
    assert result.flag_type == FlagType.QUOTE_ALTERED
    assert "msj" in (result.issue or "").lower()


def test_citation_without_quote_is_untouched():
    c = _citation(None)
    [result] = ground_citation_quotes([c], MSJ)
    assert result.flag_type is None


def test_existing_flag_type_is_not_overwritten():
    c = _citation("a hirer is ALWAYS fully liable")
    c.flag_type = FlagType.OVERSTATEMENT
    [result] = ground_citation_quotes([c], MSJ)
    # Don't clobber a more specific flag the agent already set.
    assert result.flag_type == FlagType.OVERSTATEMENT
