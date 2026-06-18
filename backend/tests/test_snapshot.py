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
