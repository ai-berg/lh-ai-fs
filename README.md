# BS Detector

A multi-agent pipeline that audits a legal **Motion for Summary Judgment** (MSJ)
against the surrounding case file and reports where the brief misstates its
authorities or contradicts the record — the *epistemic work* of checking
citations and record cites, returned as structured, source-grounded JSON.

> **Status:** Tiers 1, 2, and 3 are implemented and tested — four distinct agents,
> a deterministic confidence-scoring layer, a judicial-memo synthesis agent, a
> single-command eval harness, graceful degradation, and a structured UI (judicial
> memo + verdict-colored finding/citation cards with confidence bands). The remaining
> stretch is depth, not coverage: a fuller design system and the future agents named
> in the [Roadmap](#roadmap).

## What it does

`POST /analyze` runs the case documents in `backend/documents/` through a
multi-agent pipeline and returns a `VerificationReport`:

Four agents with deliberately **distinct, non-overlapping** roles — meaning vs.
facts vs. words vs. synthesis:

- **Citation Audit Agent** — extracts every legal authority cited in the MSJ,
  assesses (on internal plausibility) whether it *supports* the proposition it is
  cited for, and flags direct-quote overstatements (e.g. an absolute "a hirer is
  *never* liable"). Authorities it cannot confirm are reported as
  `could_not_verify` — never fabricated as `verified`. *(a legal-merits judgment)*
- **Cross-Document Consistency Agent** — contrasts the MSJ's factual assertions
  against the police report, medical records, and witness statement, citing the
  **minimal verbatim span** that contains each conflict (e.g. the incident date,
  whether fall-arrest PPE was worn). *(a fact-vs-fact check)*
- **Quote Accuracy Agent** — checks whether passages the MSJ quotes from the
  case-file documents are *faithful* to their source (words quietly removed,
  inserted, or altered), distinct from whether an authority supports a proposition.
  *(a textual-fidelity check)*
- **Judicial Memo Agent** — synthesizes the confirmed, highest-confidence findings
  into a **one-paragraph bench memo** for a judge. Decision support, not
  displacement: it summarizes what the audit found and how certain it is, and never
  opines on the merits. *(the only LangChain-based agent — see below)*

Every finding is **grounded**: a quote that does not literally exist in its cited
source document is rejected and the finding collapses to `could_not_verify`. This
is the anti-hallucination guarantee — the pipeline points at real text or admits
uncertainty.

Each flag also carries a **deterministic confidence band** (`services/confidence.py`)
derived from verifiable signals — chiefly how many *distinct* reference documents
corroborate it — not a number the model self-reports. `HIGH` means independent
documents agree; the score is reproducible by hand and auditable on hover in the UI.

## Architecture

```
POST /analyze
   └─ orchestrator.run_pipeline(docs)
        ├─ CitationAuditAgent        ─┐
        ├─ CrossDocConsistencyAgent  ─┤  (3 agents fan out concurrently)
        └─ QuoteAccuracyAgent        ─┘
              └─ grounding.validate_grounding()   ← drops ungrounded findings
              └─ confidence.score_confidence()    ← deterministic band per flag
              └─ JudicialMemoAgent  (LangChain LCEL)  ← synthesizes confirmed findings
        → VerificationReport { citations, flags, judicial_memo, degraded_agents }
```

- **Routes → Services → Repositories.** `main.py` is a thin route; agents and
  orchestration live in `services/`; document loading in `repositories/`.
- **Structured data between agents** (Pydantic models in `schemas.py`), never raw
  text blobs. Agents emit a `FindingDraft`; the orchestrator promotes it to a scored
  `Finding` — confidence is assigned post-grounding, never self-reported.
- **OpenAI SDK directly for the fan-out; LangChain only for the memo.** The three
  parallel agents have no chaining/retrieval/tool-loop, so a framework there is pure
  overhead; the memo is a single `prompt | llm | structured_output` LCEL chain — the
  canonical use case. Framework used where it pays, not by default (see REFLECTION).
- **Resilient orchestration**: all four agents (memo included) run through one
  `run_agent` wrapper — timeout + one retry + `degraded_agents` tracking — so any
  single failure degrades that slice without sinking the report.
- **Prompt-injection defense**: untrusted document text is fenced with a
  **per-document** random sentinel before substitution (`prompts.py`) — each
  document gets its own marker, so a malicious document can't forge a *sibling*
  document's fence (it never sees the sibling's random marker). That sibling-forgery
  resistance is exactly what a single per-request sentinel would not provide.

## Run it

Requires **Docker Compose v2.24+** (the hardening override uses the `!override`
merge tag).

```bash
cp backend/.env.example .env      # add your OPENAI_API_KEY at the repo root
docker compose up --build
```

The API runs at `http://localhost:8002` (published on loopback only). Try it:

```bash
curl -X POST http://localhost:8002/analyze | jq
```

**Or use the UI.** Open **http://localhost:5175** and click **Run Analysis** — the
React app renders the report as structured cards (one per finding and citation,
color-coded by verdict: red = contradicted, amber = could-not-verify, green =
verified) with the verbatim evidence span and its source document under each, plus
the raw JSON behind a toggle. The verdict colors are intentional: an abstention
(`could_not_verify`) is amber, not red — "I can't confirm this" is a first-class
outcome here, not an error.

> **Security note.** This repo runs third-party scaffold code, so it is hardened
> via `docker-compose.override.yml` (non-root, dropped capabilities,
> loopback-only ports, resource limits) and dependencies are installed at image
> build time rather than at boot. Treat your `OPENAI_API_KEY` as exposed to the
> sandboxed code: use a dedicated key with a low billing cap.

## Tests

Run inside the container (the project is Docker-first):

```bash
docker compose run --rm backend python -m pytest -q
```

Most tests are deterministic and LLM-free: the grounding layer
(verbatim/normalized evidence checks), orchestration resilience (graceful agent
degradation), the schema invariants, and the prompt-injection fencing — the
safety-critical logic, verified without spending tokens (agents are tested with a
monkeypatched LLM).

End-to-end behavior is pinned by a **committed snapshot** of a real GPT-5.5 run
on the case file (`tests/fixtures/analyze_snapshot.json`); `test_snapshot.py`
asserts its load-bearing properties (all 11 authorities extracted, fictional
authorities never fabricated as "verified", every contradiction flag grounded,
the PPE contradiction caught). Regenerate it with
`docker compose exec backend python scripts/capture_snapshot.py`.

## Evals

```bash
# In the container (Docker-first):
docker compose exec backend python eval/run_evals.py          # score the committed snapshot (reproducible, no API)
docker compose exec backend python eval/run_evals.py --live   # run the real pipeline, then score (spends API)

# Or on the host, no Docker (a root shim delegates to backend/eval/run_evals.py):
pip install -r backend/requirements.txt && python run_evals.py
```

The harness scores the pipeline against hand-labeled gold sets across **two
cases** — the provided Rivera matter (`backend/eval/gold_set.yaml`) and a synthetic
contract case (`backend/eval/cases/synthetic_contract/`) authored to show the method
generalizes past one fixture — and reports per-case **and aggregate**, honestly:

- **Recall** — planted flaws caught, per flaw. The gold set includes a flaw the
  pipeline is *expected to miss* (an intra-document arithmetic slip with no
  checking agent), so the Rivera case reports an honest **5/6** and the
  cross-case **aggregate is 8/9** — never a staged 100%. The honesty axis runs in
  both directions: a control flaw checks that the two *real* authorities the brief
  cites accurately (Privette, SeaBright) are **not** over-flagged as fabricated.
- **Precision** — false flags landing on labeled *negatives* (true MSJ statements
  that must not be flagged); without negatives precision is meaningless, so each
  gold set ships hard negatives the model is tempted to flag but must not. The
  synthetic case's hard negative actually trips the pipeline — a true contract
  deadline it over-flags as contradicted by a later schedule revision — so the
  harness honestly reports **FP=1** (band `[71%, 83%]` aggregate) rather than hiding
  the spurious flag. Plausible-but-unplanted findings go to a `pending_adjudication`
  bucket — scored neither right nor wrong, so the number isn't gamed in either
  direction. **Scope:** precision is measured over the cross-doc *flag* stream; the
  citation-support stream has no labeled "citation that must not be flagged" set, so
  it is reported as a separate diagnostic (flag-type distribution + a check that an
  `overstatement` carries the quote it overstates) rather than folded into the band.
- **Grounding consistency** — cited quotes that don't literally exist in their
  source. This re-runs the pipeline's *own* grounding check, so it is a regression
  guard, **not** an independent hallucination oracle: it is ~0 post-gate by
  construction (the report says so), and `--live` runs the pre-gate vs post-gate
  ablation that shows what the gate actually removes from the raw model output.

The metric arithmetic has its own 14 unit tests (`backend/eval/test_metrics.py`) on
synthetic true-positive / false-positive / miss / hallucination cases, so the
scoring logic is proven independent of any model output. (These prove the
*method* is correct, not that the pipeline catches flaws at scale — that rests on
the planted flaws across the gold set's cases, not on the test count.)

## Roadmap

| Tier | Item | Status |
|------|------|--------|
| 1 | Citation extraction + support assessment + quote flags, JSON output | ✅ done |
| 1 | Grounding / anti-hallucination | ✅ done |
| 2 | Cross-document consistency | ✅ done |
| 2 | **Eval harness** (`python run_evals.py`): precision, recall, grounding-consistency + gate ablation | ✅ done |
| 3 | **4 distinct agents** (Citation, CrossDoc, QuoteAccuracy, JudicialMemo) | ✅ done |
| 3 | **Deterministic confidence scoring** (band + reasoning per flag) | ✅ done |
| 3 | **Judicial-memo agent** (LangChain LCEL, one-paragraph bench memo) | ✅ done |
| 3 | **Graceful degradation** (all agents through `run_agent`: timeout + retry + `degraded_agents`) | ✅ done |
| 3 | Structured UI (judicial memo + finding/citation cards + confidence bands) | ✅ done |
| 3 | Fuller design system / future agents (temporal-arithmetic, omission) | ⏳ planned |
| — | [Reflection document](REFLECTION.md) | ✅ done |

## Design influences

Some of the design choices here are grounded in prior production experience;
others come from recent literature. Attributing them explicitly:

**From building a production legal-domain LLM/RAG assistant (a court system's
AI assistant, ~600k users):**
- *Grounding by literal source citation* — forcing the model to point at exact
  source text and rejecting anything it can't, rather than trusting free-form
  output. In that system it was enforced via mandatory source-chunk tags; here
  it's the verbatim-quote check in `grounding.py`.
- *Treating document text as data, never instructions* — the `<security>` block
  and the per-document sentinel fencing in `prompts.py` mirror the prompt-injection
  defenses used there for untrusted case-file content.
- *Passing structured data between stages* and *expressing uncertainty instead of
  fabricating* (`could_not_verify`) — both are operating principles carried over
  from that project.

**From studying current research / best practices:**
- *Minimal-span evidence citation* and *groundedness as a first-class metric* —
  informed by recent grounded-generation work (e.g. Google's FACTS Grounding
  benchmark, MiniCheck). This shaped the "cite the shortest span containing the
  fact" instruction and the word-boundary grounding check.
- *Native structured outputs over a heavyweight agent framework* — a deliberate
  simplicity choice for a small agent graph, rather than reaching for LangChain.

## The assessment

This project implements the "BS Detector" technical assessment for the
Sr Fullstack Engineer role at Learned Hand.
