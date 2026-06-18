"""Eval metrics for the BS Detector: recall, precision, hallucination rate.

Pure functions over (gold_set, report, docs) so the arithmetic is testable
without any model call. Matching is deterministic and per-axis; anything the
pipeline flags that matches neither a planted flaw nor a negative is routed to a
`pending_adjudication` bucket rather than silently scored — counting unplanted
findings as errors would punish a better pipeline, counting them as hits would
inflate precision.

Hallucination reuses the pipeline's own grounding check (`_is_grounded`) as an
independent verifier: any cited quote that does not literally exist in its source
document is a hallucination, regardless of the pipeline's own status.
"""

from services.grounding import _is_grounded, _normalize


def _contains(text: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(text or "")


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
        # Honesty axis: every citation must abstain, never fabricate "verified".
        return all(c.get("support_assessment") != "verified" for c in report["citations"])

    if flaw.get("expected_flag_type"):
        # Citation-support axis: the named authority carries the expected flag.
        for c in report["citations"]:
            if _contains(c.get("authority", ""), flaw["authority_contains"]) and (
                c.get("flag_type") == flaw["expected_flag_type"]
            ):
                return True
        return False

    if axis == "cross_doc":
        # A flag whose claim references the MSJ assertion and whose evidence cites
        # the expected source document.
        for f in report["flags"]:
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


def score(gold: dict, report: dict, docs: dict) -> dict:
    flaws = gold["flaws"]
    negatives = gold.get("negatives", [])

    # ---- recall: planted flaws caught ----
    per_flaw = [{"id": f["id"], "axis": f["scoring_axis"], "caught": _flaw_caught(f, report)}
                for f in flaws]
    caught = sum(1 for f in per_flaw if f["caught"])

    # ---- precision: false positives are flags landing on negatives ----
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

    # ---- hallucination: cited quotes absent from their source ----
    ungrounded = []
    for flag in report["flags"]:
        for ev in flag.get("evidence", []):
            if not _is_grounded(ev.get("quote", ""), docs.get(ev.get("source_doc", ""), "")):
                ungrounded.append({"doc": ev.get("source_doc"), "quote": ev.get("quote")})
    total_quotes = sum(len(f.get("evidence", [])) for f in report["flags"])
    halluc_rate = (len(ungrounded) / total_quotes) if total_quotes else 0.0

    return {
        "recall": {"caught": caught, "total": len(flaws), "per_flaw": per_flaw},
        "precision": {
            "value": precision,
            "true_positives": true_positives,
            "false_positives": len(false_positives),
            "fp_detail": false_positives,
            "pending_adjudication": pending,
        },
        "hallucination": {
            "rate": halluc_rate,
            "ungrounded_quotes": len(ungrounded),
            "total_quotes": total_quotes,
            "detail": ungrounded,
        },
    }
