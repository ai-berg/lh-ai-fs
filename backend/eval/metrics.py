"""Eval metrics for the BS Detector: recall, precision, grounding consistency.

Pure functions over (gold_set, report, docs) so the arithmetic is testable
without any model call. Matching is deterministic and per-axis; anything the
pipeline flags that matches neither a planted flaw nor a negative is routed to a
`pending_adjudication` bucket rather than silently scored — counting unplanted
findings as errors would punish a better pipeline, counting them as hits would
inflate precision.

The grounding-consistency check re-applies the pipeline's own `_is_grounded` to
every cited quote. NOTE: this is NOT an independent hallucination oracle — it's
the same check the pipeline's grounding gate runs, so on the shipped report it is
~0 by construction. It is a regression guard (catches a quote that slipped the
gate) and the basis for the --live pre/post-gate ablation, not a measure of the
raw model's faithfulness. A true independent check would need a different
mechanism (entailment/NLI), named as future work in services/grounding.py.
"""

import math

from services.grounding import is_grounded, normalize


def wilson_ci(successes: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson 95% confidence interval for a proportion.

    With our tiny denominators (n=2..8) a bare "100%"/"0%" point estimate is
    misleading — the Wilson interval is the standard small-sample correction and
    keeps the bounds inside [0, 1]. Returned as [low, high], rounded.
    """
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [round(max(0.0, center - half), 2), round(min(1.0, center + half), 2)]


def _contains(text: str, needle: str) -> bool:
    return normalize(needle) in normalize(text or "")


def _flag_text(flag: dict) -> str:
    """All the model-authored prose on a finding, for keyword matching."""
    return " ".join(
        [flag.get("msj_claim", ""), flag.get("explanation", "")]
        + [e.get("quote", "") for e in flag.get("evidence", [])]
    )


def _flaw_caught(flaw: dict, report: dict) -> bool:
    """Did the pipeline catch this planted flaw? Deterministic, per-axis."""
    axis = flaw["scoring_axis"]

    if flaw.get("expectation") == "no_citation_marked_verified":
        # Honesty axis for fictional authorities, two rules:
        #  - NEVER "verified" — asserting a fabricated authority is good is the
        #    cardinal sin, even if some flag_type is also attached.
        #  - Otherwise abstain (could_not_verify), OR carry a flag_type that
        #    justifies an assertive "contradicted". A bare "contradicted" with no
        #    flag is an unfounded ruling, not honesty.
        # No credit when the citation agent degraded or extracted nothing: an
        # empty all([]) is vacuously True and would inflate recall on exactly the
        # failure path — there were no authorities to audit honestly.
        if not report["citations"] or "CitationAuditAgent" in report.get("degraded_agents", []):
            return False

        def _honest(c: dict) -> bool:
            sa = c.get("support_assessment")
            if sa == "verified":
                return False
            return sa == "could_not_verify" or bool(c.get("flag_type"))

        return all(_honest(c) for c in report["citations"])

    if flaw.get("expected_flag_type"):
        # Citation-support axis: the named authority carries the expected flag.
        for c in report["citations"]:
            if _contains(c.get("authority", ""), flaw["authority_contains"]) and (
                c.get("flag_type") == flaw["expected_flag_type"]
            ):
                return True
        return False

    if axis == "cross_doc":
        # A flag that ASSERTS the contradiction (status==contradicted), references
        # the MSJ assertion, and cites the expected source document. Requiring the
        # contradicted status means an abstention or wrong-side verdict on the
        # right sentence does not earn catch credit (which would inflate recall).
        for f in report["flags"]:
            if f.get("status") != "contradicted":
                continue
            claim_hit = _contains(f.get("msj_claim", ""), flaw["msj_claim_contains"])
            doc_hit = any(
                e.get("source_doc") == flaw["proof_doc"] for e in f.get("evidence", [])
            )
            if claim_hit and doc_hit:
                return True
        return False

    return False


def _matches_negative(flag: dict, negatives: list[dict]) -> dict | None:
    """A flag is a false positive if it targets a known-true negative span."""
    text = _flag_text(flag)
    for neg in negatives:
        if _contains(text, neg["proof_span"]):
            return neg
    return None


def _matches_any_flaw(flag: dict, flaws: list[dict]) -> bool:
    for flaw in flaws:
        if flaw.get("where") == "cross_doc" and _contains(
            flag.get("msj_claim", ""), flaw.get("msj_claim_contains", "\x00")
        ):
            return True
    return False


_ASSERTIVE = {"contradicted", "verified"}


def grounding_consistency_rate(flags: list[dict], docs: dict) -> dict:
    """Quote-grounding + unsupported-assertion checks over a finding list.

    Two fabrication signals, on either pre-gate (raw agent) or post-gate (report)
    findings:
      - ungrounded_quotes: cited quotes that don't literally exist in their source
        (the --live ablation quantity). Not an independent oracle — see module doc.
      - unsupported_assertions: findings that assert contradicted/verified while
        citing NO evidence. A fabricated *finding* carries no quote for the
        grounding check to catch, so it would otherwise be invisible. (~0 on the
        shipped report, since the Finding validator downgrades these — measured
        here so a regression in that validator can't hide.)
    """
    ungrounded = [
        {"doc": ev.get("source_doc"), "quote": ev.get("quote")}
        for flag in flags
        for ev in flag.get("evidence", [])
        if not is_grounded(ev.get("quote", ""), docs.get(ev.get("source_doc", ""), ""))
    ]
    unsupported = [
        flag.get("msj_claim")
        for flag in flags
        if flag.get("status") in _ASSERTIVE and not flag.get("evidence")
    ]
    total = sum(len(f.get("evidence", [])) for f in flags)
    return {
        "rate": (len(ungrounded) / total) if total else 0.0,
        "ungrounded_quotes": len(ungrounded),
        "total_quotes": total,
        "unsupported_assertions": len(unsupported),
        "unsupported_detail": unsupported,
        "detail": ungrounded,
    }


def score(gold: dict, report: dict, docs: dict) -> dict:
    flaws = gold["flaws"]
    negatives = gold.get("negatives", [])

    # ---- recall: planted flaws caught ----
    per_flaw = [{"id": f["id"], "axis": f["scoring_axis"], "caught": _flaw_caught(f, report)}
                for f in flaws]
    caught = sum(1 for f in per_flaw if f["caught"])

    # ---- precision: false positives are flags landing on negatives ----
    # SCOPE (deliberate): precision is measured over the cross-doc *flag* stream
    # only. Negatives are true MSJ statements that must not be flagged as
    # contradictions; a citation-support verdict is a different kind of judgment
    # and would need its own labeled "citations that must not be flagged" set to
    # score honestly, which this fixture doesn't have. Scoping precision to the
    # flag stream keeps the denominator meaningful rather than padding it.
    false_positives, pending = [], []
    for flag in report["flags"]:
        neg = _matches_negative(flag, negatives)
        if neg:
            false_positives.append({"negative": neg["id"], "claim": flag.get("msj_claim")})
        elif not _matches_any_flaw(flag, flaws):
            # Plausible but unplanted — neither rewarded nor penalized.
            pending.append(flag.get("msj_claim"))

    true_positives = sum(1 for f in per_flaw if f["caught"] and f["axis"] == "cross_doc")
    denom = true_positives + len(false_positives)
    precision = (true_positives / denom) if denom else None
    # Precision as a range over the pending bucket: if every pending finding were
    # ultimately a false positive, precision would fall to its lower bound — so we
    # report the band rather than the optimistic point estimate.
    precision_low = (
        (true_positives / (denom + len(pending)))
        if (denom + len(pending))
        else None
    )

    # ---- grounding consistency: cited quotes absent from their source (post-gate)
    grounding = grounding_consistency_rate(report["flags"], docs)

    return {
        "recall": {
            "caught": caught,
            "total": len(flaws),
            "per_flaw": per_flaw,
            "ci95": wilson_ci(caught, len(flaws)),
        },
        "precision": {
            "value": precision,
            "value_low": precision_low,  # if all pending turn out FP
            "true_positives": true_positives,
            "false_positives": len(false_positives),
            "fp_detail": false_positives,
            "pending_adjudication": pending,
            "negatives_checked": len(negatives),
            "ci95": wilson_ci(true_positives, denom) if denom else None,
        },
        "grounding_consistency": grounding,
    }
