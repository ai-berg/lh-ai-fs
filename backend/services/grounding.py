"""Anti-hallucination grounding for pipeline findings.

A finding earns its evidence only if every quote literally appears in the cited
source document. We normalize whitespace and case so the model may re-wrap text,
but a quote that is genuinely absent (a fabricated citation) downgrades the
finding to ``could_not_verify`` and clears its evidence.

Design note: the "model must point at real source text, never invent it" rule is
carried over from prior production experience on a legal-domain RAG assistant.

This is a verbatim *attribution* check (does the cited quote literally exist in
the source?), chosen for determinism and auditability — which legal review wants.
It is deliberately NOT an entailment/NLI faithfulness check (e.g. MiniCheck,
RAGAS faithfulness), which judge whether a *paraphrased* claim is semantically
supported. Those are the production upgrade path for catching unsupported
inferences that use real words; they were out of scope (and out of the
no-fabrication budget) for this exercise.
"""

import re
import unicodedata

from schemas import Citation, FlagType, Finding, VerificationStatus

_GROUNDING_FAILED_NOTE = " [grounding check failed: quote not found in source]"
_QUOTE_NOT_IN_MSJ_NOTE = "Quoted text was not found verbatim in the MSJ; the quote may be altered."

# Fold typographic variants that legal-PDF extraction introduces, so a quote the
# model regenerates with a hyphen/straight-quote still matches a source that uses
# an em-dash/curly-quote (a benign difference, not an altered quote). Applied to
# both sides, so it cannot let a genuinely fabricated quote through.
_TYPOGRAPHIC = str.maketrans(
    {
        "“": '"', "”": '"',          # curly double quotes
        "‘": "'", "’": "'",          # curly single quotes / apostrophe
        "–": "-", "—": "-", "‒": "-", "−": "-",  # en/em/figure/minus dashes
        "…": "...",                        # ellipsis
        " ": " ", "​": "",            # nbsp, zero-width space
    }
)


def _normalize(text: str) -> str:
    """Lowercase, fold typographic variants, and collapse whitespace.

    NFKC unifies compatibility forms (ligatures, superscripts) before the manual
    quote/dash/ellipsis fold; the goal is that benign extraction artifacts don't
    cause false-negative grounding on otherwise-identical text.
    """
    text = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHIC)
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_grounded(quote: str, document: str) -> bool:
    """True if ``quote`` appears in ``document`` at word boundaries.

    Plain substring matching would treat "March 1" as present in "March 12",
    silently grounding a partial-token quote. We require the match to start and
    end on a word boundary so a quote that only covers a prefix/suffix of a
    longer token is rejected.
    """
    needle = _normalize(quote)
    haystack = _normalize(document)
    if not needle or not haystack:
        return False
    # \b alone is unreliable around punctuation; anchor on the needle itself.
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack) is not None


def validate_grounding(finding: Finding, docs: dict[str, str]) -> Finding:
    """Return a grounded copy of ``finding``, clearing ungrounded evidence.

    Pure: the input is never mutated. Evidence is verified whenever it is present
    — regardless of status — so a ``could_not_verify`` finding cannot smuggle an
    unverified (possibly fabricated) quote through to the client. The only
    short-circuit is the already-clean case: ``could_not_verify`` with no
    evidence. Any miss collapses the finding to ``could_not_verify`` with evidence
    cleared and the explanation annotated so the downgrade is visible.
    """
    if finding.status == VerificationStatus.COULD_NOT_VERIFY and not finding.evidence:
        return finding

    for ev in finding.evidence:
        if not _is_grounded(ev.quote, docs.get(ev.source_doc, "")):
            return finding.model_copy(
                update={
                    "status": VerificationStatus.COULD_NOT_VERIFY,
                    "evidence": [],
                    "explanation": finding.explanation + _GROUNDING_FAILED_NOTE,
                }
            )

    return finding


def ground_citation_quotes(citations: list[Citation], msj: str) -> list[Citation]:
    """Return citations with altered-quote flags, without mutating the inputs.

    A citation's ``quoted_text`` purports to be lifted from the brief. If that
    text cannot be found verbatim in the MSJ, the quote may be altered, so we
    set ``flag_type=quote_altered`` (unless the agent already assigned a more
    specific flag) instead of trusting it. This extends the verbatim-grounding
    guarantee to the Tier 1 citation path. Pure: returns new/unchanged models.
    """
    result = []
    for citation in citations:
        flag_quote = (
            citation.quoted_text
            and citation.flag_type is None
            and not _is_grounded(citation.quoted_text, msj)
        )
        if flag_quote:
            issue = (citation.issue + " " if citation.issue else "") + _QUOTE_NOT_IN_MSJ_NOTE
            result.append(
                citation.model_copy(
                    update={"flag_type": FlagType.QUOTE_ALTERED, "issue": issue}
                )
            )
        else:
            result.append(citation)
    return result
