"""Structured contract shared across the BS Detector pipeline.

Every agent consumes and emits Pydantic models (never raw text blobs), and the
``/analyze`` endpoint returns a ``VerificationReport``. Keeping this contract in
one module lets the eval harness and the UI depend on a single source of truth.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FlagType(str, Enum):
    """Category of an issue raised against the Motion for Summary Judgment."""

    FACTUAL_CONTRADICTION = "factual_contradiction"
    CROSS_DOC_INCONSISTENCY = "cross_doc_inconsistency"
    CITATION_UNSUPPORTED = "citation_unsupported"
    QUOTE_ALTERED = "quote_altered"
    OVERSTATEMENT = "overstatement"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationStatus(str, Enum):
    """Outcome of checking a claim or citation against the source documents.

    ``COULD_NOT_VERIFY`` is a first-class outcome: the pipeline must prefer it
    over fabricating a finding when literal supporting evidence is absent.
    """

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    COULD_NOT_VERIFY = "could_not_verify"


class _EnumValueModel(BaseModel):
    """Base that serializes enum fields as their string values.

    Keeps ``model_dump()`` and the FastAPI response identical (e.g.
    ``"could_not_verify"`` rather than ``VerificationStatus.COULD_NOT_VERIFY``).
    Because our enums subclass ``str``, equality checks against the enum members
    still hold.
    """

    model_config = ConfigDict(use_enum_values=True)


class EvidenceRef(BaseModel):
    """A verbatim pointer back to a source document — the grounding primitive.

    ``quote`` must be copied literally from ``source_doc`` so it can be checked
    against the corpus; a quote that does not exist downgrades the finding to
    ``could_not_verify`` (see ``services.grounding``).
    """

    source_doc: str = Field(
        ..., description="Document stem, e.g. 'police_report', 'witness_statement'."
    )
    quote: str = Field(..., description="Text copied literally from the source document.")
    locator: str | None = Field(
        None, description="Optional human hint for where the quote sits, e.g. 'line ~51'."
    )


class Citation(_EnumValueModel):
    """A legal authority cited by the MSJ and our assessment of it."""

    authority: str = Field(..., description="e.g. 'Privette v. Superior Court'.")
    reporter: str | None = Field(
        None, description="Reporter/pin cite, e.g. '5 Cal.4th 689, 695 (1993)'."
    )
    proposition: str = Field(
        ..., description="What the MSJ claims this authority supports."
    )
    is_direct_quote: bool = Field(
        ..., description="True if the MSJ presents quoted text from the authority."
    )
    quoted_text: str | None = Field(
        None, description="The literal quoted text, when is_direct_quote is True."
    )
    # Declared BEFORE support_assessment on purpose: with structured outputs the
    # model emits keys in field order, so a reasoning field placed first is the
    # only way the prompt's "proposition -> evidence -> verdict" chain-of-thought
    # actually runs before the verdict is decided (rather than being narrated
    # after the fact).
    assessment_reasoning: str = Field(
        ...,
        description="Brief step-by-step analysis BEFORE the verdict: the "
        "proposition, then the internal evidence bearing on it.",
    )
    support_assessment: VerificationStatus = Field(
        ..., description="Whether the authority plausibly supports the proposition."
    )
    flag_type: FlagType | None = Field(
        None,
        description="Machine-readable category of the problem, if any: "
        "'overstatement' (absolute/overbroad claim), 'quote_altered' (the quoted "
        "text looks edited), or 'citation_unsupported' (authority does not support "
        "the proposition). Null when the citation is unproblematic.",
    )
    issue: str | None = Field(
        None,
        description="Human-readable explanation of the problem, paired with flag_type.",
    )

    @model_validator(mode="after")
    def _no_verified_without_a_quote(self) -> "Citation":
        # Mirror of Finding's evidence guard: a "verified" support assessment with
        # no quoted_text to stand on is an unfounded claim about an authority the
        # pipeline cannot look up. Fail safe to could_not_verify deterministically,
        # not just by prompt instruction.
        if self.support_assessment == VerificationStatus.VERIFIED and not self.quoted_text:
            self.support_assessment = VerificationStatus.COULD_NOT_VERIFY
        return self


class Finding(_EnumValueModel):
    """A grounded issue raised by an agent against the MSJ."""

    flag_type: FlagType
    msj_claim: str = Field(..., description="The assertion in the MSJ under scrutiny.")
    # Before `status` (see Citation.assessment_reasoning): forces the prompt's
    # numbered comparison method to run before the verdict under structured output.
    comparison_reasoning: str = Field(
        ...,
        description="Brief analysis BEFORE the verdict: the MSJ fact, what the "
        "reference documents say about it, and whether they conflict.",
    )
    status: VerificationStatus
    evidence: list[EvidenceRef] = Field(
        default_factory=list,
        description="Verbatim support; required when status is not could_not_verify.",
    )
    explanation: str
    # The orchestrating agent overwrites this after parsing, so its value from the
    # model is irrelevant. (Note: OpenAI strict structured outputs make every field
    # required regardless of a Python default, so the model still emits it — the
    # default just gives a clean value when a Finding is built in code/tests.)
    raised_by: str = Field(default="", description="Agent that produced this finding.")

    @model_validator(mode="after")
    def _require_evidence_for_assertions(self) -> "Finding":
        # A verified/contradicted claim with no evidence is internally
        # inconsistent. Fail safe to could_not_verify rather than asserting
        # something the finding cannot back up (mirrors the grounding layer), and
        # annotate the explanation so the downgrade isn't silently contradictory.
        if self.status != VerificationStatus.COULD_NOT_VERIFY and not self.evidence:
            self.status = VerificationStatus.COULD_NOT_VERIFY
            self.explanation += " [downgraded: no supporting evidence provided]"
        return self


class CitationAuditOutput(BaseModel):
    """Wrapper returned by the citation agent (structured outputs need a root object)."""

    citations: list[Citation] = Field(default_factory=list)


class CrossDocOutput(BaseModel):
    """Wrapper returned by the cross-document consistency agent."""

    findings: list[Finding] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Top-level response of ``POST /analyze``.

    ``degraded_agents`` records agents that failed gracefully so the report is
    transparent about partial coverage instead of silently dropping work.
    """

    citations: list[Citation] = Field(default_factory=list)
    flags: list[Finding] = Field(default_factory=list)
    degraded_agents: list[str] = Field(default_factory=list)
