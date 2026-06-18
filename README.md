# BS Detector

A multi-agent pipeline that audits a legal **Motion for Summary Judgment** (MSJ)
against the surrounding case file and reports where the brief misstates its
authorities or contradicts the record — the *epistemic work* of checking
citations and record cites, returned as structured, source-grounded JSON.

> **Status:** Tier 1 (core pipeline) is implemented and tested. The eval harness
> (Tier 2) and the additional agents / UI (Tier 3) are in progress — see
> [Roadmap](#roadmap).

## What it does

`POST /analyze` runs the case documents in `backend/documents/` through a
multi-agent pipeline and returns a `VerificationReport`:

- **Citation Audit Agent** — extracts every legal authority cited in the MSJ,
  assesses (on internal plausibility) whether it supports the proposition it is
  cited for, and flags direct-quote overstatements (e.g. an absolute "a hirer is
  *never* liable"). Authorities it cannot confirm are reported as
  `could_not_verify` — never fabricated as `verified`.
- **Cross-Document Consistency Agent** — contrasts the MSJ's factual assertions
  against the police report, medical records, and witness statement, citing the
  **minimal verbatim span** that contains each conflict (e.g. the incident date,
  whether fall-arrest PPE was worn).

Every finding is **grounded**: a quote that does not literally exist in its cited
source document is rejected and the finding collapses to `could_not_verify`. This
is the anti-hallucination guarantee — the pipeline points at real text or admits
uncertainty.

## Architecture

```
POST /analyze
   └─ orchestrator.run_pipeline(docs)
        ├─ CitationAuditAgent        ─┐  (fan out concurrently)
        └─ CrossDocConsistencyAgent  ─┘
              └─ grounding.validate_grounding()   ← drops ungrounded findings
        → VerificationReport { citations, flags, degraded_agents }
```

- **Routes → Services → Repositories.** `main.py` is a thin route; agents and
  orchestration live in `services/`; document loading in `repositories/`.
- **Structured data between agents** (Pydantic models in `schemas.py`), never raw
  text blobs. Uses OpenAI **native structured outputs** — no heavyweight
  framework.
- **Resilient orchestration**: agents fan out concurrently; a failing agent is
  recorded in `degraded_agents` and the pipeline still returns a valid report.
- **Prompt-injection defense**: untrusted document text is fenced with a
  per-request random sentinel before substitution (`prompts.py`).

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

The harness scores the pipeline against a hand-frozen gold set
(`eval/gold_set.yaml`, labeled from the source documents *before* running the
pipeline) and reports, honestly:

- **Recall** — planted flaws caught, per flaw. The gold set includes a flaw the
  pipeline is *expected to miss* (an intra-document arithmetic slip with no
  checking agent), so recall reports an honest **4/5**, not a staged 100%.
- **Precision** — false flags landing on labeled *negatives* (true MSJ statements
  that must not be flagged); without negatives precision is meaningless, so the
  gold set ships three. Plausible-but-unplanted findings go to a
  `pending_adjudication` bucket — scored neither right nor wrong, so the number
  isn't gamed in either direction.
- **Grounding consistency** — cited quotes that don't literally exist in their
  source. This re-runs the pipeline's *own* grounding check, so it is a regression
  guard, **not** an independent hallucination oracle: it is ~0 post-gate by
  construction (the report says so), and `--live` runs the pre-gate vs post-gate
  ablation that shows what the gate actually removes from the raw model output.

The metric arithmetic has its own unit tests (`eval/test_metrics.py`) on synthetic
true-positive / false-positive / miss / hallucination cases, so the scoring logic
is proven independent of any model output.

## Roadmap

| Tier | Item | Status |
|------|------|--------|
| 1 | Citation extraction + support assessment + quote flags, JSON output | ✅ done |
| 1 | Grounding / anti-hallucination | ✅ done |
| 2 | Cross-document consistency | ✅ done |
| 2 | **Eval harness** (`python eval/run_evals.py`): precision, recall, hallucination | ✅ done |
| 3 | Confidence scoring, judicial-memo agent (≥4 agents) | ⏳ planned |
| 3 | Structured UI | ⏳ planned |
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
  and the per-request sentinel fencing in `prompts.py` mirror the prompt-injection
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
