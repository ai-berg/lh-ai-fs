# Reflection — BS Detector

The pipeline's one job is to be trustworthy, so I spent the budget on making it
**abstain rather than guess**, and on an eval that **reports its own limits** instead
of a flattering number.

## Agent decomposition

Four agents with deliberately disjoint roles — **meaning vs. facts vs. words vs.
synthesis**:

- `CitationAuditAgent` — does an authority *support* its proposition? (legal-merits)
- `CrossDocConsistencyAgent` — does an MSJ fact *agree* with the record? (fact-vs-fact)
- `QuoteAccuracyAgent` — is a quotation *faithful* to its source? (textual fidelity)
- `JudicialMemoAgent` — synthesizes the confirmed findings into one paragraph

The first three fan out in parallel and none consumes another's output, so there is
no error-amplification path between them. I deliberately did **not** add a critique
or debate agent: self-critique is unreliable without an external signal, so I
replaced it with a deterministic verifier (`grounding.py`) that checks every cited
quote verbatim against its source and downgrades anything it can't confirm to
`could_not_verify`. For an auditability-driven legal tool, a check you can reproduce
by hand beats a second model's opinion. I split QuoteAccuracy *out* of the citation
agent — quote fidelity and legal support are different judgments — rather than adding
a fourth role for its own sake.

## Honesty enforced in code, not promised in a prompt

- The **grounding gate** rejects any quote not literally present in its source.
- A validator downgrades any `verified`/`contradicted` finding that arrives with **no
  evidence**.
- The **honest ceiling**: with no case-law lookup, the pipeline can never confirm an
  authority actually supports its proposition, so a citation validator downgrades
  **any** `verified` citation to `could_not_verify` (a quote existing in the brief
  doesn't prove the cited case says it). Fictional authorities are therefore never
  fabricated as `verified` — even when the model tries.

**Field-ordered chain-of-thought:** a `reasoning` field is declared *before* each
verdict field, because structured outputs emit keys in declaration order — so
reasoning-first is the only way the analysis runs before the verdict, not after it.

## Confidence is computed, not confessed

A model-emitted `confidence: 0.87` is exactly the unverifiable number this tool
exists to replace. So confidence is **deterministic** (`services/confidence.py`):
derived from how many *distinct* documents corroborate a finding and whether it's
assertive or an abstention. It's reproducible by hand; `HIGH` requires three
corroborating documents. The band reflects the evidence the agent *surfaced* on a run
(which varies), so read it off the artifact, not this prose:
`jq '.flags[].confidence.band' backend/tests/fixtures/analyze_snapshot.json`.
Trade-off: a deterministic score can't capture how *semantically* strong a
contradiction is — but "how corroborated is this, by how many independent documents"
is the signal that supports a judge's decision and that I can defend by hand.

## LangChain where it pays — the memo only

The fan-out agents are independent parallel calls with no chaining, retrieval, or
tool loop, so a framework there is pure overhead and would hide the decomposition.
The judicial memo is a single synthesis step (`prompt | llm | structured_output`) —
the canonical LCEL use case — so it's built on LangChain and the other three stay on
the OpenAI SDK directly. The memo is **decision support, not displacement**: it
summarizes confirmed findings and flagged citations, leads with the
highest-confidence ones, and never opines on the merits; `grounded_in` ties it back
to the structured flags.

## The eval, built to report bad news

The grounding primitive is carried over from prior production work on a legal-domain
RAG assistant; the eval *methodology* (labeled negatives, pending-adjudication
bucket, Wilson CIs, pre/post-gate ablation) is from grounded-generation research
(FActScore, SAFE, RAGAS, Stanford RegLab) — so Tier 2 is genuinely new, not carried
over. Defenses against an eval that lies:

- **Gold labeled from the source documents**, every `proof_span` hand-checkable —
  not from pipeline output (the easiest way to grade against what you already catch).
- **Labeled negatives** so precision is computable at all; a hard negative that
  actually trips the pipeline is reported as a false positive, not hidden.
- **A `pending_adjudication` bucket** for plausible-but-unplanted findings, scored
  neither right nor wrong.
- **A deliberately-planted flaw the pipeline misses** (the "362 days" arithmetic
  slip — there's no arithmetic agent), so recall honestly reports sub-100%.
- **Metric arithmetic unit-tested** independent of any model.

Numbers vary with the captured n=1 snapshot, so I don't hard-code them here — run
`python eval/run_evals.py` for the current recall, precision band, and pending count.

## Honest caveats

- **Small denominators** (two cases, 9 planted flaws/controls). Two cases prove the
  *method* generalizes past one fixture; the Wilson CIs say the *rate* isn't settled.
- **Post-gate hallucination is ~0 "by construction"** (the check reuses the grounding
  gate), which is why `--live` runs a pre/post-gate ablation against the raw model.
- **Precision is scoped to the cross-doc flag stream** — a spurious citation verdict
  isn't counted, since that needs a "citations that must not be flagged" label set
  the fixture lacks. The citation stream is reported as a separate diagnostic.
- **Prompt-injection defense is asymmetric**: the per-document sentinel + grounding
  gate stop forgery and fabricated evidence, but injection that *suppresses* a true
  finding has no structural backstop. A production answer is a recall-floor canary.

## What I'd do next

- An **arithmetic/temporal agent** for the date-math defect the pipeline misses.
- A **real citation existence check** (CourtListener / eyecite) to turn internal
  plausibility into a true verified/not-found signal — out of scope here because the
  authorities are fictional and it adds a live dependency.
- An **entailment/NLI faithfulness pass** (MiniCheck/RAGAS) for unsupported
  inferences in real words, which verbatim grounding can't catch.

## On scope discipline

The hardest calls were the ones *not* to build: no debate loop, no NLI gate, no
external citation DB, no multi-judge jury. Each is a real production upgrade and a
take-home over-engineering trap. I'd rather ship something I can fully defend and name
the frontier than bolt on capability I can't justify in the time given.
