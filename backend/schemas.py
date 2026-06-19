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


class ConfidenceBand(str, Enum):
    """Coarse confidence bucket for a flag.

    Bands (not just a raw float) because a judge reading the report needs a legible
    signal, and because our confidence is DETERMINISTIC — derived from verifiable
    signals, not a number the model invents — so a small set of bands is both honest
    about the granularity we can actually justify and easy to act on.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class _EnumValueModel(BaseModel):
    """Base that serializes enum fields as their string values.

    Keeps ``model_dump()`` and the FastAPI response identical (e.g.
    ``"could_not_verify"`` rather than ``VerificationStatus.COULD_NOT_VERIFY``).
    Because our enums subclass ``str``, equality checks against the enum members
    still hold.
    """

    model_config = ConfigDict(use_enum_values=True)


class ConfidenceScore(_EnumValueModel):
    """How certain the pipeline is about a flag, with the reasoning behind it.

    DETERMINISTIC by design: `value`/`band` are computed from verifiable signals on
    the grounded finding (is it asserted or abstained? how many reference documents
    corroborate it? how severe?), never self-reported by the model. `reasoning` is
    generated FROM those signals, so the number is auditable — you can reconstruct it
    by hand. This is the "traceable reasoning" the product is built on; a model-
    invented 0.87 would be exactly the unverifiable number it sets out to replace.
    """

    value: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence in [0,1].")
    band: ConfidenceBand
    reasoning: str = Field(..., description="Why this score, stated from the signals.")
    signals: dict = Field(
        default_factory=dict,
        description="The raw signals that produced the score, for auditability.",
    )


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
    def _no_unverifiable_verified(self) -> "Citation":
        # HONEST CEILING: this pipeline has NO case-law lookup — it judges citations on
        # the brief's internal text alone. "verified" means "the authority actually
        # supports the proposition," which you can only confirm by reading the
        # authority. Since we can't, a `verified` verdict is never confirmable here, so
        # we fail it safe to could_not_verify deterministically — abstention is the
        # honest top outcome, exactly as the prompt instructs ("verified is rare; never
        # assert it for something you cannot confirm"). This closes the seam the
        # snapshot exposed: a fabricated authority (Kellerman) was emitted `verified`
        # with an MSJ-sourced quote — but a quote merely existing in the brief does NOT
        # prove the cited case says it. (A real production upgrade — eyecite /
        # CourtListener existence check — is named in REFLECTION as the way to earn a
        # true `verified`; deliberately out of scope here.) Assign `.value`, not the
        # enum object: use_enum_values coerces input at construction but NOT a field
        # reassigned in an after-validator, so without .value model_dump() would emit
        # the enum here and a plain string elsewhere — a heterogeneous contract.
        if self.support_assessment == VerificationStatus.VERIFIED:
            self.support_assessment = VerificationStatus.COULD_NOT_VERIFY.value
        return self


class Finding(_EnumValueModel):
    """A grounded issue raised by an agent against the MSJ.

    Field ORDER is load-bearing: structured outputs emit keys in declaration order,
    so every verdict component (status, flag_type) is placed AFTER
    comparison_reasoning. That way the model commits to the reasoning before naming
    a category — flag_type is a conclusion ("this conflict is a quote_altered"), not
    an input, so emitting it first would let the label drive the analysis instead of
    the other way round. (Mirrors Citation's reasoning-before-verdict ordering.)
    """

    msj_claim: str = Field(..., description="The assertion in the MSJ under scrutiny.")
    comparison_reasoning: str = Field(
        ...,
        description="Brief analysis BEFORE the verdict: the MSJ fact, what the "
        "reference documents say about it, and whether they conflict.",
    )
    status: VerificationStatus
    flag_type: FlagType = Field(
        ..., description="Category of the issue — a CONCLUSION drawn after the "
        "comparison_reasoning, not an input to it.",
    )
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
    # Assigned by the orchestrator AFTER grounding, never by the agent: confidence is
    # deterministic (services.confidence) so the model must not self-report it. None
    # until scored. Excluded from the agents' structured-output schema for the same
    # reason — see how the agents build their response model.
    confidence: "ConfidenceScore | None" = Field(
        default=None, description="Deterministic post-grounding confidence; set by the orchestrator."
    )

    @model_validator(mode="after")
    def _require_evidence_for_assertions(self) -> "Finding":
        # A verified/contradicted claim with no evidence is internally
        # inconsistent. Fail safe to could_not_verify rather than asserting
        # something the finding cannot back up (mirrors the grounding layer), and
        # annotate the explanation so the downgrade isn't silently contradictory.
        # Assign `.value` for the same use_enum_values reason as Citation above:
        # a field reassigned in an after-validator is not re-coerced, so emitting the
        # string keeps model_dump()['status'] homogeneous across all findings.
        if self.status != VerificationStatus.COULD_NOT_VERIFY and not self.evidence:
            self.status = VerificationStatus.COULD_NOT_VERIFY.value
            self.explanation += " [downgraded: no supporting evidence provided]"
        return self


class FindingDraft(_EnumValueModel):
    """What an agent EMITS — a Finding without the orchestration-only fields.

    Critically excludes ``confidence``: that is deterministic and assigned post-gate by
    the orchestrator, never self-reported by the model. It MUST also stay out of the
    agent's structured-output schema for a hard technical reason: OpenAI structured
    outputs reject an open ``dict`` field (it demands additionalProperties=false), and
    ConfidenceScore.signals is exactly such a dict — so embedding confidence in the
    emitted schema makes the whole response_format invalid (a 400 from the API). The
    orchestrator promotes each draft into a full ``Finding`` (see _to_finding) before
    grounding and scoring. Same field order as Finding (reasoning before verdict).
    """

    msj_claim: str = Field(..., description="The assertion in the MSJ under scrutiny.")
    comparison_reasoning: str = Field(
        ...,
        description="Brief analysis BEFORE the verdict: the MSJ fact, what the "
        "reference documents say about it, and whether they conflict.",
    )
    status: VerificationStatus
    flag_type: FlagType = Field(
        ..., description="Category of the issue — a CONCLUSION drawn after the reasoning.",
    )
    evidence: list[EvidenceRef] = Field(default_factory=list)
    explanation: str

    def to_finding(self, raised_by: str) -> "Finding":
        """Promote an emitted draft to a full Finding (provenance stamped, unscored)."""
        return Finding(
            msj_claim=self.msj_claim,
            comparison_reasoning=self.comparison_reasoning,
            status=self.status,
            flag_type=self.flag_type,
            evidence=self.evidence,
            explanation=self.explanation,
            raised_by=raised_by,
        )


class CitationAuditOutput(BaseModel):
    """Wrapper returned by the citation agent (structured outputs need a root object)."""

    citations: list[Citation] = Field(default_factory=list)


class CrossDocOutput(BaseModel):
    """Wrapper the cross-document consistency agent EMITS (drafts, no confidence)."""

    findings: list[FindingDraft] = Field(default_factory=list)


class QuoteAccuracyOutput(BaseModel):
    """Wrapper the quote-accuracy agent EMITS (drafts, no confidence).

    Reuses ``FindingDraft`` (with flag_type=quote_altered) so quote-accuracy flaws flow
    through the SAME grounding gate, confidence scoring, and report shape as cross-doc
    findings — one finding contract, not a parallel type per agent.
    """

    findings: list[FindingDraft] = Field(default_factory=list)


class JudicialMemo(_EnumValueModel):
    """A one-paragraph synthesis of the top findings, written for a judge.

    Decision SUPPORT, not displacement: the memo summarizes what the audit found and
    how certain it is, and explicitly does NOT opine on the merits or how to rule —
    mirroring a bench-memo's role and the product's "help judges focus on judgment"
    framing. `grounded_in` lists the finding claims it drew from, so the prose stays
    traceable to the structured flags rather than free-floating.
    """

    summary: str = Field(..., description="One paragraph for the judge: what the audit found and how sure.")
    grounded_in: list[str] = Field(
        default_factory=list,
        description="The msj_claims of the findings this memo synthesizes, for traceability.",
    )


class VerificationReport(BaseModel):
    """Top-level response of ``POST /analyze``.

    ``degraded_agents`` records agents that failed gracefully so the report is
    transparent about partial coverage instead of silently dropping work.
    ``judicial_memo`` is None when no findings warranted a synthesis or the memo
    agent degraded — the structured flags remain the source of truth either way.
    """

    citations: list[Citation] = Field(default_factory=list)
    flags: list[Finding] = Field(default_factory=list)
    judicial_memo: JudicialMemo | None = Field(default=None)
    degraded_agents: list[str] = Field(default_factory=list)
