# Reflection — BS Detector

A short, honest account of the design decisions, the trade-offs behind them, and
what I'd do with more time. The theme: **the pipeline's one job is to be
trustworthy, so I spent the budget on making it abstain rather than guess, and on
an eval that reports its own limits instead of a flattering number.**

## What I built and why

**Four agents with disjoint roles + a deterministic grounding gate, not N agents
for their own sake.** Three fan out in parallel — `CitationAuditAgent` (does an
authority *support* its proposition? a legal-merits judgment), `CrossDocConsistency
Agent` (does an MSJ fact *agree* with the record? a fact-vs-fact check), and
`QuoteAccuracyAgent` (is a quotation *faithful* to its source? a textual-fidelity
check) — and `JudicialMemoAgent` synthesizes their output last. The roles are
deliberately non-overlapping: meaning vs. facts vs. words vs. synthesis. None of the
fan-out three consumes another's output, so there is no error-amplification path
between them. I deliberately did **not** add a critique agent or a debate loop:
self-critique is unreliable without an external signal, so I replaced it with a
deterministic verifier (`grounding.py`) that checks every cited quote verbatim
against its source and downgrades anything it can't confirm to `could_not_verify`.
For an auditability-driven legal tool, a check you can reproduce by hand beats a
second model's opinion. (The "4 agents" line in the spec was a trap to satisfy
honestly: the test isn't a count, it's whether the roles are genuinely distinct —
so QuoteAccuracy was split *out* of the citation agent rather than bolted on.)

**Honesty enforced in code, not promised in a prompt.** Three independent
fail-safes: the grounding gate; a Pydantic validator that downgrades any
`verified`/`contradicted` finding arriving with no evidence; and grounding that
runs even on `could_not_verify` findings so an abstaining finding can't smuggle
an unverified quote through. The brief's fictional authorities are **never
fabricated as `verified`** — the cardinal sin. On the committed run Whitmore is
reported `could_not_verify`; Kellerman is `contradicted / citation_unsupported`
(the agent judged, on the brief's own facts, that the cited OSHA-compliance
proposition doesn't hold because Apex, not Harmon, was the employer). That's an
*internally-justified* contradiction with a `flag_type`, not a bare unfounded
ruling — the honesty axis allows it, and the eval's honesty check scopes to exactly
that distinction. So the honest claim is "never `verified`," not "always
`could_not_verify`"; a fabricated authority may be `contradicted` when the brief
internally undermines it.

**Field-ordered chain-of-thought.** A `reasoning` field is declared *before* each
verdict field, because with native structured outputs the model emits keys in
field order — so reasoning-first is the only way the chain-of-thought actually
runs before the verdict rather than rationalizing it after.

## Tier 3 decisions, and the trade-offs behind them

**Confidence is computed, not confessed.** The spec asks each flag to be "rated by
how certain the pipeline is, with reasoning." The obvious path — have the model emit
`confidence: 0.87` — is exactly the unverifiable number this whole tool exists to
replace, and it would contradict the honesty posture of the rest of the system. So
confidence is **deterministic** (`services/confidence.py`), derived from signals the
pipeline already checks: is the finding assertive or an abstention? how many *distinct*
reference documents corroborate it? It's reproducible by hand, and `HIGH` is reserved
for a contradiction corroborated by **three** distinct documents — so the band means
"independent documents agree," not "the model sounded sure." **Honest note on this
corpus:** on the committed Rivera snapshot *no flag actually reaches `HIGH`* — the
incident-date contradiction grounds in a single document (the police report) and so
lands `MEDIUM (0.55)`, and the PPE/responsibility flags ground in two documents and
land `MEDIUM (0.75)`. The three-document corroboration the `HIGH` band needs is
reachable for the date (the medical records and witness statement *do* state March 12),
but the cross-doc agent only quoted one source per finding on this run, so the band
stayed `MEDIUM`. That gap — the scorer can only count what the agent chose to cite — is
itself the honest finding: the band reflects the evidence actually surfaced, not the
evidence that exists. (An earlier draft of this section wrongly claimed the date flag
"lands HIGH, three sources"; that was the *unit-test fixture*, not the live run — a
reminder to quote the snapshot, never the test.) Trade-off: a deterministic score can't
capture nuance an LLM might (how *semantically* strong a contradiction is) — but for a
judge, "how corroborated is this, by how many independent documents" is the signal that
actually supports a decision, and it's one we can defend by hand.

**LangChain used where it pays — the memo, and only the memo.** The JD lists LangChain
as a core skill; the assessment penalizes a framework added without need. Both are
right at once, so the call was *where*, not *whether*. The fan-out agents are
independent parallel calls with no chaining, retrieval, or tool loop — a framework
there is pure overhead, and it would hide the agent decomposition the reviewer is
grading. The judicial memo, by contrast, is a single synthesis step (`prompt | llm |
structured_output`) — the canonical LCEL use case — so `JudicialMemoAgent` is built on
LangChain and the other three stay on the OpenAI SDK directly. Using the framework
exactly where it fits demonstrates the tool without distorting the architecture. (My
own production legal-RAG system runs without a framework for the same reason, so this
is a consistent judgment, not a one-off.)

**The memo is decision support, not displacement.** It synthesizes only the *confirmed*
(assertive), highest-confidence findings into one paragraph, leads with the most
material, qualifies lower-confidence ones explicitly, and never opines on the merits or
how to rule — mirroring a bench memo and the product's "help judges focus on judgment"
framing. Its selection logic is deterministic (which findings, in what order); only the
prose is the LLM's. `grounded_in` ties the paragraph back to the specific findings, so
the memo stays traceable to the structured flags rather than free-floating.

**A schema bug only the live run could catch.** Adding a `confidence` field to
`Finding` silently broke the agents' structured output: OpenAI's structured outputs
reject an open `dict` (it demands `additionalProperties: false`), and
`ConfidenceScore.signals` is exactly that — so the emitted schema 400'd at the API.
The unit tests passed (they mock the LLM) and the snapshot eval passed (it's static);
only a real `POST /analyze` surfaced it. The fix was also the cleaner design: agents
emit a `FindingDraft` *without* confidence (confidence is orchestration-only, assigned
post-gate), and the orchestrator promotes drafts to full `Finding`s. The lesson logged:
mock-backed green is not end-to-end green for anything touching the provider's schema.

**Resilience extended to the new agents.** All four agents — including the memo — run
through the same `run_agent` wrapper (timeout + one retry + `degraded_agents` tracking),
so a single agent failing degrades that slice of the report without sinking the whole
request. The memo runs sequentially after the fan-out (it consumes the others' output),
so its bounded timeout adds on top rather than in parallel; the per-agent timeout is env-
overridable for a larger corpus.

## The eval, and why it's built to report bad news

A note on where this came from: the *grounding primitive* the eval reuses to
detect ungrounded quotes is the same literal-source-citation check carried over
from prior production experience on a legal-domain LLM/RAG assistant. The *eval
methodology* itself — blind-frozen gold set, labeled negatives, the
pending-adjudication bucket, Wilson CIs, the pre/post-gate ablation — is from
grounded-generation research (FActScore, SAFE, RAGAS, the Stanford RegLab legal-
hallucination audits), not from that prior system, which had no output-quality
eval of this kind. So Tier 2 is the part that is genuinely new rather than
carried over.

An eval is easy to make lie. I tried to defend against the standard ways:

- **Gold set labeled from the source documents** (`backend/eval/gold_set.yaml`), every
  `proof_span` hand-checkable against the corpus. I labeled the flaws from the
  documents rather than from pipeline output (the easiest way for an eval to lie
  is to grade against what the system already catches) — but in fairness this is
  an *attestation*, not a git-provable blind freeze: the gold-set file was
  committed after the first snapshot, so trust the verbatim `proof_span`s, not my
  word on ordering.
- **Labeled negatives** (true MSJ statements that must not be flagged) so
  precision is computable at all — without them a "100% precision" headline is
  meaningless.
- **A `pending_adjudication` bucket** for plausible-but-unplanted findings, scored
  neither right nor wrong, so the number isn't gamed in either direction. The run
  actually produces one: the model's (correct) "Apex, not Harmon, controlled
  scaffolding" observation lands there instead of inflating precision.
- **A deliberately-planted flaw the pipeline misses** — the brief's "one year and
  362 days" arithmetic slip (it's ~361). There is no arithmetic-checking agent,
  so recall honestly reports a sub-100%, not a staged perfect score. An honest miss says more
  than a cherry-picked 100%.
- **Metric arithmetic is unit-tested independent of any model**
  (`backend/eval/test_metrics.py`): a perturbation that empties the findings drops recall,
  a flag on a negative raises FP, an ungrounded quote raises hallucination — the
  numbers genuinely move.

## Honest caveats (small fixture, real limits)

I'd rather state these than have them found:

- **Small denominators.** The gold set now spans two cases (the provided Rivera
  matter plus a synthetic contract case authored to test generalization), 9
  planted flaws/controls total — aggregate recall **8/9** (one deliberate
  expected-miss). I deliberately **do not hard-code the precision number here**: the
  pipeline scores against a captured snapshot, and re-capturing it (needed so the
  4th agent appears in the artifact) produces a different n=1 LLM sample each time —
  the false-positive count has swung between 0 and 1 across captures of the synthetic
  case. Quoting a single band in prose is exactly how a doc drifts out of sync with
  its own harness. **Run `python eval/run_evals.py` for the current precision band,
  TP/FP, and the pending bucket** — that command is the source of truth, the prose is
  not. (An earlier draft quoted a stale, flattering band; the lesson logged is to
  point at the command, never transcribe its output.) The one honest recall miss is Rivera's
  intra-document arithmetic slip (the "362 days" off-by-one), planted with
  `expected_to_be_missed` because there is no arithmetic-checking agent — so the
  sub-100% is real, not staged. The one false positive is the synthetic
  deadline-renegotiation over-flag described above — also real, also reported. Two
  cases prove the *method* carries past one fixture, but they don't make the *rate*
  statistically settled. The Wilson CIs say so; more cases is the obvious next
  step, not a claim made.
- **Post-gate hallucination is ~0 "by construction."** The hallucination check
  reuses the pipeline's own grounding check, so on the shipped report it's
  near-tautological. That's why `--live` runs a **pre-gate vs post-gate
  ablation**: it measures the raw model's fabrication rate before the gate and
  compares. On the runs I captured the model fabricated nothing, so the gate
  removed 0 — reported honestly rather than claimed as a win.
- **Precision is scoped to the cross-doc flag stream**, not the whole pipeline. A
  spurious citation verdict wouldn't count as a false positive, because scoring
  that would need its own "citations that must not be flagged" label set the
  fixture doesn't have. Documented in `metrics.py` as a deliberate scope, not an
  oversight.
- **The honesty axis is intentionally lenient about `contradicted`.** A fictional
  authority may be reported `contradicted` *if* it carries a `flag_type` that
  justifies it (e.g. the Privette "never liable" overstatement is internally
  detectable). A bare `contradicted` with no flag fails the axis — that's the
  line between a justified internal finding and an unfounded claim.
- **The honesty axis tests BOTH failure directions.** Beyond "never fabricate
  support for a fake case", a second control (`real_authorities_not_overflagged`)
  checks the opposite error: condemning a *genuine*, accurately-cited authority.
  The brief cites two real California Supreme Court Privette-doctrine cases
  (Privette itself and SeaBright v. US Airways) accurately; a precise pipeline must
  not mark them contradicted. Without this control a pipeline could "pass" honesty
  by reflexively distrusting everything — including real law. (An earlier draft of
  the gold set mislabeled SeaBright as fabricated; that was a factual error in the
  ground truth, since corrected — the gold is only as trustworthy as its labels,
  so the labels are verified against the documents and against real reporters.)
- **A planted hard-negative surfaced a real false positive — kept, not hidden.**
  The synthetic case's `deadline_true` negative (the contract's true Feb 28
  deadline) is flagged `contradicted` by the pipeline, which mistakes a later
  "revised schedule" note for a contradiction of what the deadline *was*. The eval
  reports this as **FP=1** (synthetic precision band drops to [75%, 75%]) rather
  than burying the spurious flag in the unscored pending bucket. It is exactly the
  over-flagging error a precision near-miss is meant to catch; reporting it is the
  point.
- **Prompt-injection defense is asymmetric — disclosed, not papered over.** The
  per-document sentinel + system header reduce *forgery* (a document forging a
  delimiter or issuing commands) and the grounding gate kills *fabricated evidence*.
  But an injection that *suppresses* a true contradiction — steering an agent to
  stay silent — produces an empty/short finding list, and every downstream gate
  only inspects findings that exist. So injection-driven **false negatives** have no
  structural backstop here; the mitigations are prompt-layer, not a guarantee. A
  production answer would add a recall floor / canary (a known planted conflict the
  pipeline must always surface) to detect suppression. Named, not built.

## What I'd do differently / next

- **An arithmetic/temporal-consistency agent** for the date-math class of defect
  the current pipeline misses (the planted, deliberately-missed arithmetic flaw).
- **A real existence check for citations** (CourtListener / eyecite) to turn
  "internal plausibility" into a true verified/not-found signal — the genuinely
  correct *production* upgrade, deliberately out of scope here because the
  authorities are fictional and it adds a live dependency.
- **An entailment/NLI faithfulness pass** (MiniCheck/RAGAS) to catch
  unsupported-inferences-in-real-words, which verbatim grounding can't. Named as
  the upgrade path in `grounding.py`; not built, because it's heavier than a
  take-home warrants and trades determinism for recall.
- **More cases + CIs**, and a confidence-scoring layer (Tier 3) calibrated from
  logprobs rather than a self-reported number, which would be fiction.

## On scope discipline

The hardest calls were the ones to *not* build: no framework (a 2-agent graph is
clearer with native structured outputs), no debate loop, no NLI gate, no external
citation DB, no multi-judge jury. Each is a real production upgrade and a
take-home over-engineering trap. I'd rather ship something I can fully defend and
name the frontier than bolt on capability I can't justify in the time given.
