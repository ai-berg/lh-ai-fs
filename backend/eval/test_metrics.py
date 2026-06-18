"""Unit tests for the eval metric logic — written before the implementation (TDD).

These run on synthetic gold/report pairs so the precision/recall/hallucination
arithmetic is proven correct independently of any model output. This is the part
of an eval most likely to be silently wrong, so it gets its own tests.
"""

from eval.metrics import score

# A minimal gold set exercising one of each axis.
GOLD = {
    "flaws": [
        {
            "id": "incident_date",
            "scoring_axis": "cross_doc",
            "where": "cross_doc",
            "msj_claim_contains": "March 14, 2021",
            "proof_doc": "police_report",
        },
        {
            "id": "fictional_authorities",
            "scoring_axis": "honesty",
            "where": "citation",
            "expectation": "no_citation_marked_verified",
        },
        {
            "id": "privette_overstatement",
            "scoring_axis": "support",
            "where": "citation",
            "authority_contains": "Privette",
            "expected_flag_type": "overstatement",
        },
    ],
    "negatives": [
        {"id": "experience_true", "proof_span": "eight years of experience"},
    ],
}

DOCS = {
    "police_report": "Date of Incident: March 12, 2021.",
    "motion_for_summary_judgment": "March 14, 2021. eight years of experience.",
}


def _report(citations, flags):
    return {"citations": citations, "flags": flags, "degraded_agents": []}


def test_perfect_run_scores_full_recall_no_fp():
    report = _report(
        citations=[
            {"authority": "Privette v. Superior Court", "support_assessment": "could_not_verify",
             "flag_type": "overstatement", "quoted_text": None, "issue": None},
            {"authority": "Whitmore", "support_assessment": "could_not_verify",
             "flag_type": None, "quoted_text": None, "issue": None},
        ],
        flags=[
            {"flag_type": "cross_doc_inconsistency", "status": "contradicted",
             "msj_claim": "incident on March 14, 2021",
             "evidence": [{"source_doc": "police_report", "quote": "March 12, 2021"}]},
        ],
    )

    r = score(GOLD, report, DOCS)

    assert r["recall"]["caught"] == 3  # date, fictional-abstain, overstatement
    assert r["recall"]["total"] == 3
    assert r["precision"]["false_positives"] == 0


def test_missed_flaw_lowers_recall():
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "could_not_verify",
                    "flag_type": None, "quoted_text": None, "issue": None}],
        flags=[],  # missed the date contradiction and the overstatement
    )

    r = score(GOLD, report, DOCS)

    assert r["recall"]["caught"] == 1  # only the abstain-on-fictional axis
    assert r["recall"]["total"] == 3


def _honesty_caught(r):
    return "fictional_authorities" in {
        c["id"] for c in r["recall"]["per_flaw"] if c["caught"]
    }


def test_fabricated_verified_breaks_the_honesty_axis():
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "verified",
                    "flag_type": None, "quoted_text": None, "issue": None}],
        flags=[],
    )
    assert not _honesty_caught(score(GOLD, report, DOCS))


def test_fabricated_contradicted_without_a_flag_breaks_the_honesty_axis():
    # Asserting "contradicted" about a fictional authority — with no flag_type and
    # no evidence to back it — is also an unfounded claim, not honest abstention.
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "contradicted",
                    "flag_type": None, "quoted_text": None, "issue": None}],
        flags=[],
    )
    assert not _honesty_caught(score(GOLD, report, DOCS))


def test_contradicted_with_a_justifying_flag_is_honest():
    # An internally-detectable problem (e.g. an overstatement the brief states
    # absolutely) may be reported as contradicted when a flag_type justifies it.
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "contradicted",
                    "flag_type": "overstatement", "quoted_text": None,
                    "issue": "absolute claim"}],
        flags=[],
    )
    assert _honesty_caught(score(GOLD, report, DOCS))


def test_pure_abstention_is_honest():
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "could_not_verify",
                    "flag_type": None, "quoted_text": None, "issue": None}],
        flags=[],
    )
    assert _honesty_caught(score(GOLD, report, DOCS))


def test_verified_fictional_fails_even_with_a_flag():
    # Closing the OR-seam: "verified" on a fabricated authority is the cardinal
    # sin and must fail the honesty axis even if a flag_type is also present.
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "verified",
                    "flag_type": "overstatement", "quoted_text": None, "issue": None}],
        flags=[],
    )
    assert not _honesty_caught(score(GOLD, report, DOCS))


def test_flag_on_a_negative_is_a_false_positive():
    report = _report(
        citations=[],
        flags=[
            {"flag_type": "factual_contradiction", "status": "contradicted",
             "msj_claim": "Rivera has eight years of experience",  # a NEGATIVE
             "evidence": [{"source_doc": "motion_for_summary_judgment",
                           "quote": "eight years of experience"}]},
        ],
    )

    r = score(GOLD, report, DOCS)

    assert r["precision"]["false_positives"] == 1


def test_ungrounded_quote_counts_against_grounding_consistency():
    report = _report(
        citations=[],
        flags=[
            {"flag_type": "factual_contradiction", "status": "contradicted",
             "msj_claim": "something",
             "evidence": [{"source_doc": "police_report", "quote": "TEXT THAT IS NOT IN ANY DOC"}]},
        ],
    )

    r = score(GOLD, report, DOCS)

    assert r["grounding_consistency"]["ungrounded_quotes"] == 1


def test_wilson_ci_widens_on_tiny_n():
    from eval.metrics import wilson_ci

    # 2/2 must NOT report [1.0, 1.0]: the small-sample interval admits a much
    # lower true rate. And it stays inside [0, 1].
    low, high = wilson_ci(2, 2)
    assert 0.0 <= low < 0.5 and high == 1.0
    # 0/8 has a non-trivial upper bound (rule-of-three ~0.3), not 0.
    assert wilson_ci(0, 8)[1] > 0.2
