"""Endpoint test for POST /analyze — the primary deliverable.

The orchestrator is monkeypatched so we verify the HTTP contract and JSON
serialization (the enum-as-string regression in particular) without LLM calls.
"""

import pytest
from fastapi.testclient import TestClient

import main
from schemas import (
    Citation,
    EvidenceRef,
    Finding,
    FlagType,
    VerificationReport,
    VerificationStatus,
)


@pytest.fixture
def client(monkeypatch):
    async def fake_pipeline(docs):
        return VerificationReport(
            citations=[
                Citation(
                    authority="Privette v. Superior Court",
                    proposition="A hirer is never liable.",
                    is_direct_quote=True,
                    quoted_text="A hirer is never liable",
                    assessment_reasoning="Absolute claim, no internal support.",
                    support_assessment=VerificationStatus.COULD_NOT_VERIFY,
                    flag_type=FlagType.OVERSTATEMENT,
                    issue="Absolute claim.",
                )
            ],
            flags=[
                Finding(
                    flag_type=FlagType.CROSS_DOC_INCONSISTENCY,
                    msj_claim="The incident occurred on March 14, 2021.",
                    comparison_reasoning="MSJ March 14 vs police report March 12.",
                    status=VerificationStatus.CONTRADICTED,
                    evidence=[
                        EvidenceRef(source_doc="police_report", quote="March 12, 2021")
                    ],
                    explanation="Date conflict.",
                    raised_by="CrossDocConsistencyAgent",
                )
            ],
            degraded_agents=[],
        )

    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    return TestClient(main.app)


def test_analyze_returns_structured_report(client):
    response = client.post("/analyze")

    assert response.status_code == 200
    body = response.json()
    assert len(body["citations"]) == 1
    assert len(body["flags"]) == 1


def test_report_fields_are_top_level_not_wrapped(client):
    # Contract guard: the UI reads citations/flags from the top level. A
    # regression that re-nests them under a `report` key would leave the UI
    # blank despite a 200, which unit tests of either side miss in isolation.
    body = client.post("/analyze").json()

    assert "report" not in body
    assert {"citations", "flags", "degraded_agents"} <= body.keys()


def test_enums_serialize_as_plain_strings(client):
    # Guards the use_enum_values contract: no "VerificationStatus.X" leakage.
    body = client.post("/analyze").json()

    assert body["citations"][0]["support_assessment"] == "could_not_verify"
    assert body["citations"][0]["flag_type"] == "overstatement"
    assert body["flags"][0]["status"] == "contradicted"
