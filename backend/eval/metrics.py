"""Eval metrics for the BS Detector: recall, precision, grounding consistency.

Pure functions over (gold_set, report, docs) so the arithmetic is testable
without any model call. Matching is deterministic and per-axis; anything the
pipeline flags that matches neither a planted flaw nor a negative is routed to a
`pending_adjudication` bucket rather than silently scored — counting unplanted
findings as errors would punish a better pipeline, counting them as hits would
inflate precision.

The grounding-consistency check re-applies the pipeline's own `is_grounded` to
every cited quote. NOTE: this is NOT an independent hallucination oracle — it's
the same check the pipeline's grounding gate runs, so on the shipped report it is
~0 by construction. It is a regression guard (catches a quote that slipped the
gate) and the basis for the --live pre/post-gate ablation, not a measure of the
raw model's faithfulness. A true independent check would need a different
mechanism (entailment/NLI), named as future work in services/grounding.py.

Attribution: the reused grounding primitive (`is_grounded`) carries over the
literal-source-citation discipline from prior production experience on a
legal-domain LLM/RAG assistant. The *eval methodology* here — blind-frozen gold
set, labeled negatives, pending-adjudication, Wilson CIs, the gate ablation —
comes from grounded-generation research (FActScore, SAFE, RAGAS, the Stanford
RegLab legal-hallucination audits), not from that prior system, which had no
output-quality eval of this kind.
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
        cites = report["citations"]
        if not cites or "CitationAuditAgent" in report.get("degraded_agents", []):
            return False

        # Scope the honesty check to the fabricated authorities this flaw names.
        targets = flaw.get("fictional_authorities")
        scoped = (
            [c for c in cites if any(_contains(c.get("authority", ""), a) for a in targets)]
            if targets
            else cites
        )

        # COVERAGE GATE (why this and not a total-count floor): honesty is judged
        # only over `scoped`, so we must confirm the fabricated authorities were
        # actually extracted before crediting "they all abstained". A total-count
        # check (len(cites) >= min_citations) does NOT do this — a run could extract
        # 11 citations, only 2 of them fabricated (both abstaining) plus 9 real ones
        # asserted "verified", and still pass. Gate on coverage of the LIST by
        # DISTINCT target: every named fabricated authority must be matched by at
        # least one citation. Counting len(scoped) would let two duplicate entries of
        # one authority stand in for a missing sibling (len==targets without full
        # coverage); count distinct targets matched instead.
        if targets:
            covered = sum(
                1 for a in targets
                if any(_contains(c.get("authority", ""), a) for c in cites)
            )
            if covered < len(targets):
                return False
        min_cov = flaw.get("min_citations", 1)
        if len(cites) < min_cov or not scoped:
            return False

        def _honest(c: dict) -> bool:
            sa = c.get("support_assessment")
            if sa == "verified":
                return False
            return sa == "could_not_verify" or bool(c.get("flag_type"))

        return all(_honest(c) for c in scoped)

    if flaw.get("expectation") == "no_real_authority_marked_contradicted":
        # Real-authority control (the OTHER honesty failure mode): a precise pipeline
        # must not condemn a genuine, accurately cited authority. We credit the flaw
        # when every named real authority that was extracted is NOT asserted
        # "contradicted" without an internal basis. could_not_verify is fine (the
        # tool has no case-law lookup); "verified" is fine (it recognized the case);
        # only a bare "contradicted" with no justifying flag_type is the error.
        cites = report["citations"]
        if "CitationAuditAgent" in report.get("degraded_agents", []):
            return False
        targets = flaw.get("real_authorities", [])
        scoped = [c for c in cites if any(_contains(c.get("authority", ""), a) for a in targets)]
        if not scoped:
            return False

        def _not_condemned(c: dict) -> bool:
            return not (c.get("support_assessment") == "contradicted" and not c.get("flag_type"))

        return all(_not_condemned(c) for c in scoped)

    # Citation-support on the CITATION stream: the named authority carries the
    # expected flag AND, when the flaw names a proof_span, the citation's quoted_text
    # contains it — so an overstatement flag on the WRONG proposition of a
    # same-named authority (the fixture has two Privette citations) doesn't get
    # credit for catching the planted absolute quote.
    if flaw.get("expected_flag_type") and flaw.get("where") == "citation":
        span = flaw.get("proof_span")
        for c in report["citations"]:
            if not _contains(c.get("authority", ""), flaw["authority_contains"]):
                continue
            if c.get("flag_type") != flaw["expected_flag_type"]:
                continue
            if span and not _contains(c.get("quoted_text", "") or "", span):
                continue
            return True
        return False

    if axis in ("cross_doc", "intra_doc_arithmetic"):
        # A flag that ASSERTS the contradiction (status==contradicted), references
        # the MSJ assertion, cites the expected source document AND whose evidence
        # actually contains the gold proof_span (not just any sentence from the
        # right doc). If the flaw names an expected_flag_type, require it too — this
        # is how a flag-stream quote_altered finding is scored. Requiring all of
        # these stops a grounded-but-irrelevant quote, an abstention, or a
        # wrong-flag finding from inflating recall. intra_doc_arithmetic shares
        # this path so a future arithmetic checker that emits such a flag is
        # credited instead of permanently counted as missed.
        want_flag = flaw.get("expected_flag_type")
        for f in report["flags"]:
            if f.get("status") != "contradicted":
                continue
            if want_flag and f.get("flag_type") != want_flag:
                continue
            claim_hit = _contains(f.get("msj_claim", ""), flaw["msj_claim_contains"])
            span_hit = any(
                e.get("source_doc") == flaw["proof_doc"]
                and _contains(e.get("quote", ""), flaw["proof_span"])
                for e in f.get("evidence", [])
            )
            if claim_hit and span_hit:
                return True
        return False

    return False


def _matches_negative(flag: dict, negatives: list[dict]) -> dict | None:
    """A flag is a false positive if the CLAIM it challenges is a known-true negative.

    Matches against msj_claim + explanation only — NOT the cited evidence. A valid
    contradiction can legitimately cite a reference sentence that happens to also
    contain a negative's span (e.g. a fuller witness line); matching on evidence
    would wrongly score that true positive as a false positive too.
    """
    text = " ".join([flag.get("msj_claim", ""), flag.get("explanation", "")])
    for neg in negatives:
        if _contains(text, neg["proof_span"]):
            return neg
    return None


def _matches_any_flaw(flag: dict, flaws: list[dict]) -> bool:
    """Whether a flag corresponds to a planted cross-doc flaw.

    Strict: requires the claim, the expected flag_type (when named), and the
    proof_span in the cited evidence — the same bar _flaw_caught uses. A loose
    claim-substring-only match would wrongly keep a genuinely-unplanted or
    wrong-flag finding OUT of pending_adjudication, skewing the precision band.
    """
    for flaw in flaws:
        if flaw.get("where") != "cross_doc":
            continue
        if not _contains(flag.get("msj_claim", ""), flaw.get("msj_claim_contains", "\x00")):
            continue
        want_flag = flaw.get("expected_flag_type")
        if want_flag and flag.get("flag_type") != want_flag:
            continue
        if not any(
            e.get("source_doc") == flaw.get("proof_doc")
            and _contains(e.get("quote", ""), flaw.get("proof_span", "\x00"))
            for e in flag.get("evidence", [])
        ):
            continue
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
        # Only an ASSERTIVE flag can be a false positive. A finding the grounding
        # gate downgraded to could_not_verify makes no false claim, so it must not
        # be counted against precision even if it mentions a negative's span.
        if flag.get("status") != "contradicted":
            continue
        neg = _matches_negative(flag, negatives)
        if neg:
            false_positives.append({"negative": neg["id"], "claim": flag.get("msj_claim")})
        elif not _matches_any_flaw(flag, flaws):
            # Plausible but unplanted — neither rewarded nor penalized.
            pending.append(flag.get("msj_claim"))

    # TP counts caught flaws that live in the FLAG stream — keyed on `where ==
    # "cross_doc"`, the SAME predicate _matches_any_flaw uses for the pending gate.
    # (Earlier this keyed on scoring_axis == "cross_doc", which dropped a caught
    # `where:cross_doc / axis:intra_doc_arithmetic` flag out of both TP and pending —
    # it counted in recall but vanished from precision. Aligning both predicates on
    # `where` keeps the flag-stream accounting internally consistent.)
    # HONEST COUPLING NOTE: TP reuses recall's "caught" flags, so the precision POINT
    # estimate is partly tautological (a caught planted flaw is, by construction, a
    # true positive). What precision adds independently is the FP/pending denominator
    # — flags landing on labeled negatives or on nothing planted. Read the band and
    # FP count, not the point %; the negatives are what make precision falsifiable.
    flag_stream_ids = {f["id"] for f in flaws if f.get("where") == "cross_doc"}
    true_positives = sum(1 for f in per_flaw if f["caught"] and f["id"] in flag_stream_ids)
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

    # ---- citation-support stream diagnostic ----
    # WHY a separate diagnostic, not a precision FP: the precision band scopes to the
    # cross-doc FLAG stream (it has labeled negatives; the citation stream does not).
    # But the citation stream can still emit a malformed verdict that would otherwise
    # cost nothing — e.g. an `overstatement` flag with no quoted_text to overstate.
    # An overstatement is a claim ABOUT a direct quote, so it must carry one; a
    # citation flagged overstatement with quoted_text empty is an unsupported verdict.
    # We surface the count (and the flag_type distribution) so the report can't hide
    # a stream of quote-free overstatements behind an unscored axis.
    # Require a non-empty quoted_text REGARDLESS of the is_direct_quote boolean: an
    # overstatement is a verdict about the quote's wording, so the quote must be
    # present. A truthy is_direct_quote with an empty quoted_text is exactly the
    # malformed case this diagnostic exists to catch, so it must not suppress it.
    citation_issues = [
        {"authority": c.get("authority"), "flag_type": c.get("flag_type")}
        for c in report["citations"]
        if c.get("flag_type") == "overstatement" and not c.get("quoted_text")
    ]
    flag_type_dist: dict[str, int] = {}
    for c in report["citations"]:
        ft = c.get("flag_type")
        if ft:
            flag_type_dist[ft] = flag_type_dist.get(ft, 0) + 1

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
        "citation_support": {
            "flag_type_distribution": flag_type_dist,
            "malformed_overstatements": citation_issues,
        },
    }
