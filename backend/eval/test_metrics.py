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


def test_fabricated_verified_breaks_the_honesty_axis():
    report = _report(
        citations=[{"authority": "Whitmore", "support_assessment": "verified",
                    "flag_type": None, "quoted_text": None, "issue": None}],
        flags=[],
    )

    r = score(GOLD, report, DOCS)

    # Marking a fictional authority "verified" fails the honesty flaw.
    caught_ids = {c["id"] for c in r["recall"]["per_flaw"] if c["caught"]}
    assert "fictional_authorities" not in caught_ids


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


def test_ungrounded_quote_counts_as_hallucination():
    report = _report(
        citations=[],
        flags=[
            {"flag_type": "factual_contradiction", "status": "contradicted",
             "msj_claim": "something",
             "evidence": [{"source_doc": "police_report", "quote": "TEXT THAT IS NOT IN ANY DOC"}]},
        ],
    )

    r = score(GOLD, report, DOCS)

    assert r["hallucination"]["ungrounded_quotes"] == 1
