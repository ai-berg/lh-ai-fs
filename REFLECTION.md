# Reflection — BS Detector

A short, honest account of the design decisions, the trade-offs behind them, and
what I'd do with more time. The theme: **the pipeline's one job is to be
trustworthy, so I spent the budget on making it abstain rather than guess, and on
an eval that reports its own limits instead of a flattering number.**

## What I built and why

**Two LLM agents + a deterministic grounding gate, not N agents.**
`CitationAuditAgent` reasons about authorities within the MSJ alone;
`CrossDocConsistencyAgent` contrasts MSJ facts against the reference bundle. They
fan out in parallel and neither consumes the other's output, so there is no
error-amplification path between them. I deliberately did **not** add a critique
agent or a debate loop: self-critique is unreliable without an external signal,
so I replaced it with a deterministic verifier (`grounding.py`) that checks every
cited quote verbatim against its source and downgrades anything it can't confirm
to `could_not_verify`. For an auditability-driven legal tool, a check you can
reproduce by hand beats a second model's opinion.

**Honesty enforced in code, not promised in a prompt.** Three independent
fail-safes: the grounding gate; a Pydantic validator that downgrades any
`verified`/`contradicted` finding arriving with no evidence; and grounding that
runs even on `could_not_verify` findings so an abstaining finding can't smuggle
an unverified quote through. The authors of the brief's fictional authorities
(Whitmore, Kellerman, …) are reported as `could_not_verify`, never fabricated as
supported.

**Field-ordered chain-of-thought.** A `reasoning` field is declared *before* each
verdict field, because with native structured outputs the model emits keys in
field order — so reasoning-first is the only way the chain-of-thought actually
runs before the verdict rather than rationalizing it after.

## The eval, and why it's built to report bad news

An eval is easy to make lie. I tried to defend against the standard ways:

- **Gold set labeled from the source documents** (`eval/gold_set.yaml`), every
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
  so recall honestly reports **4/5**, not a staged 4/4. An honest 80% says more
  than a cherry-picked 100%.
- **Metric arithmetic is unit-tested independent of any model**
  (`eval/test_metrics.py`): a perturbation that empties the findings drops recall,
  a flag on a negative raises FP, an ungrounded quote raises hallucination — the
  numbers genuinely move.

## Honest caveats (small fixture, real limits)

I'd rather state these than have them found:

- **Small denominators.** The gold set now spans two cases (the provided Rivera
  matter plus a synthetic contract case authored to test generalization), 8
  planted flaws total — aggregate recall 7/8. The one honest miss is Rivera's
  intra-document arithmetic slip (the "362 days" off-by-one), planted with
  `expected_to_be_missed` because there is no arithmetic-checking agent — so the
  sub-100% is real, not staged. Two cases prove the *method* carries past one
  fixture, but they don't make the *rate* statistically settled. The Wilson CIs
  say so; more cases is the obvious next step, not a claim made.
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

## What I'd do differently / next

- **An arithmetic/temporal-consistency agent** for the date-math class of defect
  the current pipeline misses (the planted 4/5 miss).
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
