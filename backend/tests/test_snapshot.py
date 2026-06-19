"""Regression checks against a committed real /analyze run.

`fixtures/analyze_snapshot.json` is the verbatim output of the pipeline on the
real case file (GPT-5.5), captured and committed so the end-to-end claims are
evidence, not assertion. These tests assert the load-bearing properties of that
captured run without spending an API call; regenerate the snapshot with
`scripts/capture_snapshot.py` when the pipeline changes intentionally.
"""

import json
from pathlib import Path

import pytest

from schemas import VerificationReport

SNAPSHOT = Path(__file__).parent / "fixtures" / "analyze_snapshot.json"


@pytest.fixture(scope="module")
def report() -> VerificationReport:
    # Round-trips through the schema, so the snapshot is also a contract check.
    return VerificationReport.model_validate_json(SNAPSHOT.read_text())


def test_snapshot_is_a_valid_report(report):
    assert report.citations
    assert report.degraded_agents == []  # a clean run, no agent failed


def test_all_authorities_extracted(report):
    # The MSJ cites 11 authorities (4 in-body cases, the § 335.1 statute, and a
    # 6-case footnote string-cite); the run must recover them all.
    assert len(report.citations) >= 11


def test_fictional_authorities_are_not_fabricated_as_verified(report):
    # The authorities are fictional; the honest outcome is could_not_verify,
    # never "verified". This is the core no-fabrication guarantee.
    assert all(c.support_assessment != "verified" for c in report.citations)


def test_every_flag_is_grounded(report):
    # Any flag that asserts a contradiction must carry verbatim evidence;
    # could_not_verify flags must not smuggle evidence through.
    for f in report.flags:
        if f.status == "contradicted":
            assert f.evidence, f"contradicted flag without evidence: {f.msj_claim}"


def test_catches_the_ppe_contradiction(report):
    # The planted PPE flaw: the MSJ says no fall-arrest PPE; the record says he
    # wore a harness. The run should flag a contradiction citing a reference doc.
    ppe = [
        f
        for f in report.flags
        if "ppe" in f.msj_claim.lower()
        or "protective equipment" in f.msj_claim.lower()
        or "harness" in (f.explanation or "").lower()
    ]
    assert ppe, "expected a PPE contradiction flag in the captured run"


# --- Tier-3 deliverables: previously shipped with zero snapshot coverage ---


def test_every_flag_carries_a_confidence_band(report):
    # The deterministic confidence layer must score every flag with a valid band and a
    # reasoning string — not leave any flag unscored.
    for f in report.flags:
        assert f.confidence is not None, f"flag has no confidence: {f.msj_claim}"
        assert f.confidence.band in ("low", "medium", "high")
        assert f.confidence.reasoning


def test_confidence_band_matches_its_value(report):
    # The band must be consistent with the numeric value (>=0.8 HIGH, >=0.5 MEDIUM,
    # else LOW) — pins that the score and its label can't drift apart.
    for f in report.flags:
        c = f.confidence
        expected = "high" if c.value >= 0.8 else "medium" if c.value >= 0.5 else "low"
        assert c.band == expected, f"band {c.band} != {expected} for value {c.value}"


def test_judicial_memo_is_present_and_grounded(report):
    # The Tier-3 memo must exist on a run with findings, summarize in prose, and tie
    # back to the findings/citations it was built from (provenance, not free-floating).
    memo = report.judicial_memo
    assert memo is not None, "expected a judicial memo on a run with confirmed defects"
    assert memo.summary.strip()
    assert memo.grounded_in, "memo must record what it synthesized"


def test_judicial_memo_does_not_opine_on_the_merits(report):
    # Decision support, not displacement: the bench memo must not tell the judge how to
    # rule. A coarse guard against the most obvious merits language.
    text = report.judicial_memo.summary.lower()
    for banned in ("should grant", "should deny", "i recommend", "the court should",
                   "motion should be granted", "motion should be denied"):
        assert banned not in text, f"memo opines on the merits: '{banned}'"
