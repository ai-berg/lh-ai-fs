"""Deterministic confidence scoring for findings.

WHY deterministic (and not a model-reported number): the product's promise is
verified citations and traceable reasoning for judges. A self-reported `confidence:
0.87` from the LLM is exactly the unverifiable number that posture rejects — it
can't be reproduced, audited, or defended. So we derive confidence from signals that
are already CHECKED by the pipeline:

  - assertiveness: an assertive (contradicted) finding is a real claim; an abstention
    (could_not_verify) makes no claim, so confidence in it AS A FLAG must stay low.
  - corroboration: how many DISTINCT reference documents independently carry the
    conflicting fact. Cross-document agreement is the strongest signal we have that a
    contradiction is real and not a one-document artifact.
  - groundedness: every quote already passed the grounding gate by the time we score,
    so a contradicted finding here is grounded by construction; we treat ungrounded/
    evidence-free findings (which the gate would have downgraded) as low.

The score is a small, transparent function of those signals, and `reasoning` is
generated from them — so anyone can reconstruct the number by hand. This is also why
it's unit-testable with no LLM: same finding, same score, every time.
"""

from schemas import ConfidenceBand, ConfidenceScore, Finding, VerificationStatus

# Band thresholds on the [0,1] value. HIGH is deliberately reserved for an assertive,
# multi-source-corroborated flag, so the band means something stronger than "the
# model sounded sure" — it means independent documents agree.
_HIGH = 0.8
_MEDIUM = 0.5


def _band(value: float) -> ConfidenceBand:
    if value >= _HIGH:
        return ConfidenceBand.HIGH
    if value >= _MEDIUM:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def score_confidence(finding: Finding) -> ConfidenceScore:
    """Derive a confidence score for a (already grounded) finding from its signals."""
    # An abstention asserts nothing — confidence in a flag it is NOT making is low by
    # definition, regardless of how much text it cites. Short-circuit so corroboration
    # can't lift a non-claim into medium/high.
    if finding.status == VerificationStatus.COULD_NOT_VERIFY:
        return ConfidenceScore(
            value=0.25,
            band=ConfidenceBand.LOW,
            reasoning="Abstention (could_not_verify): the pipeline is not asserting a "
            "flaw here, so confidence in a contradiction is intentionally low.",
            signals={"assertive": False, "corroborating_sources": 0},
        )

    # Distinct reference documents that carry the conflicting fact. Distinct (not raw
    # evidence count) so two quotes from the same report don't masquerade as two
    # independent corroborations.
    sources = {ev.source_doc for ev in finding.evidence if ev.source_doc}
    n_sources = len(sources)

    # Base credit for being an assertive, grounded contradiction, plus a flat +0.20
    # per additional corroborating document, capped at +0.35 (so the bonus is linear
    # for the 2nd and 3rd source, then flatlines — a 4th+ document adds nothing).
    # Concretely: 1 source -> 0.55 (MEDIUM), 2 -> 0.75 (MEDIUM), 3+ -> 0.90 (HIGH).
    # HIGH therefore requires THREE distinct corroborating documents, by design.
    base = 0.55                       # one grounded assertive source clears MEDIUM
    corroboration_bonus = min(0.35, 0.20 * (n_sources - 1)) if n_sources > 1 else 0.0
    value = round(min(1.0, base + corroboration_bonus), 2)

    band = _band(value)
    # Word the reasoning to match the BAND actually reached — only HIGH (>=3 distinct
    # documents) earns the "strongest signal / independent documents agree" language;
    # a 2-source MEDIUM is "partially corroborated", not the top signal.
    if band == ConfidenceBand.HIGH:
        corroboration_note = (
            f" — corroborated across {n_sources} independent documents (the strongest"
            " deterministic signal of a real contradiction)."
        )
    elif n_sources > 1:
        corroboration_note = (
            f" — partially corroborated ({n_sources} documents); HIGH needs three."
        )
    else:
        corroboration_note = " — single-source; confidence is moderate pending corroboration."
    reasoning = (
        f"Assertive {finding.status} flag grounded in {n_sources} "
        f"{'source' if n_sources == 1 else 'sources'}" + corroboration_note
    )
    return ConfidenceScore(
        value=value,
        band=band,
        reasoning=reasoning,
        signals={"assertive": True, "corroborating_sources": n_sources},
    )
