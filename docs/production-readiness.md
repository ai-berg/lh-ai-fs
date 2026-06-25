# Part 2 — Production Readiness Plan: BS Detector as a Paid MVP

> **Stance up front (a design choice, not a fact).** The one load-bearing change from Part 1 to Part 2 is
> turning the synchronous `POST /analyze` — in `backend/main.py` a no-arg route that does
> `docs = load_documents(); return await run_pipeline(docs)` and **blocks for the whole multi-minute run** —
> into a **durable, status-reporting, idempotent async job**. Everything else hangs off that. The honest
> core (`run_pipeline`, `run_agent`, `grounding.py`, `confidence.py`, the four agents, `prompts.py`) is
> already correct, deterministic where it matters, and provider-portable. Part 2 **relocates** it onto a
> durable plane; it does not rewrite it.
>
> **Where I've lived this.** I operated a production legal-domain RAG assistant embedded in a national
> court system (~600k users, Anthropic Claude on AWS Bedrock). Throughout, I cite that experience as
> *"here is what I'd do better,"* never as a success story. AWS-first is authentic to it; I draw
> portability boundaries only where they pay.

Notation: every claim is grounded either in the challenge scenario (`docs/production-readiness-plan.md`)
or in the verified Part 1 code. Anything that is a judgment call is tagged **`(ours)`**.

---

## Q1 — Assumptions: users, scale, latency, reliability, risk

- **Users.** Paying law firms / legal teams. The operator is a lawyer or paralegal; the ultimate reader of
  a report is often a judge (the bench-memo framing the `JudicialMemoAgent` is built around). **`(ours)`**
  The natural access unit is therefore **`(tenant, matter, role)`**, not bare `tenant_id` — privilege
  attaches *per matter*, and an ethical wall inside a firm can wall same-firm users off a matter. I name
  the unit correctly now even though full intra-firm RBAC is deferred (Q8), so it stays additive, not a rewrite.
- **Scale.** Scenario: "hundreds of simultaneous users at launch," "path to tens of thousands," a matter is
  "dozens to hundreds of documents." **`(ours)`** I take the **matter (one verification run) as the unit of
  concurrency** and size the MVP for low hundreds of concurrent matters. Tens of thousands is the *stated
  path*, explicitly deferred (Q8) — designing for it now is the over-engineering I refuse.
- **Latency.** From the code: `AGENT_TIMEOUT_SECONDS` defaults to **120s** (`services/agents/base.py`),
  sized for a measured ~50–90s reasoning call. Three finding agents fan out concurrently via `asyncio.gather`
  in `run_agents`, then `JudicialMemoAgent` runs last. A single-document analysis is ~1–3 min; a real matter
  is minutes-to-tens-of-minutes — exactly the scenario's "minutes." **`(ours)`** This is an **async-job
  product, not a chat product**: firms tolerate minutes *if* status is visible and the result is trustworthy.
- **Reliability.** **`(ours)`** The costly failure here is **not a 500 — it's a report that silently checked
  3 of 4 documents and didn't say so.** The bar is **no silent under-coverage**, which
  `VerificationReport.degraded_agents` (`schemas.py`) already encodes per request and which Part 2 must carry
  end-to-end into the durable report and a production metric. I assume **at-least-once execution with
  idempotency**, not exactly-once.
- **Risk.** Data is "confidential, privileged, or sensitive." **`(ours)`** I assume the worst — privileged
  work product — so the dominant risk is **confidentiality / privilege loss** (cross-tenant leak, a document
  reaching a provider training corpus, a privilege waiver via careless retention), not downtime.
  **`(ours, go/no-go)`** I assume we can contractually get **zero-retention / no-training** terms from the
  model provider (OpenAI today, since Part 1 is on the OpenAI SDK), attested in the data-processing agreement
  and surfaced to the customer. **Plan B if a firm won't accept OpenAI as a sub-processor:** the `llm.py`
  seam swaps to **AWS Bedrock or Azure OpenAI under their zero-retention terms** — the validated portability
  target, not a vague hope — so this is a contained provider swap, not a dead end.

**Explicitly NOT assuming:** sub-second latency; that documents arrive clean (real matters are PDFs — OCR is
in the eventual path, not the MVP); any throughput beyond the scenario's "hundreds → tens of thousands."

---

## Q2 — Major components, and why these boundaries

Boundaries are drawn by **failure domain and duration** — each maps to one AWS-managed primitive and one
recovery story.

```
                          BS Detector — MVP system context
   ┌──────────┐  upload (presigned PUT)   ┌──────────────────────────────────┐
   │ Law-firm │ ────────────────────────► │ S3  documents/{tenant}/{matter}/ │ (SSE-KMS, per-tenant prefix)
   │  user    │                           └──────────────────────────────────┘
   │ (tenant, │  POST /analyze (202+job)  ┌──────────────┐  enqueue   ┌──────────────┐
   │  matter, │ ────────────────────────► │  API /control│ ─────────► │ SQS  + DLQ   │
   │   role)  │  GET /jobs/{id}  (status) │  plane       │ ◄────────  └──────┬───────┘
   │          │  GET /jobs/{id}/report    │ (FastAPI)    │  read job         │ trigger
   └──────────┘ ◄──────────────────────── └──────┬───────┘                  ▼
                                                  │ r/w        ┌──────────────────────────┐
                                          ┌───────▼────────┐   │  Worker (SQS→Lambda)      │
                                          │ DynamoDB        │   │  run_pipeline (UNCHANGED) │
                                          │ jobs (tenant,   │◄──┤  fan-out 3 agents + memo  │
                                          │  job): status,  │   │  → grounding → confidence │
                                          │  manifest, pins,│   └─────────┬─────────────────┘
                                          │  degraded_agents│             │ structured calls
                                          └─────────────────┘             ▼
                                                                  ┌────────────────┐
                                                                  │ LLM provider   │ (zero-retention)
                                                                  │ OpenAI / (Bedrock later)
                                                                  └────────────────┘
```

1. **API / control plane** — API Gateway + a small FastAPI (Lambda via Mangum, or Fargate behind an ALB).
   Owns auth, tenant resolution, presigned uploads, job creation, and the **millisecond** status/report
   reads. It does **not** run analysis. *Rationale:* the thing that answers in milliseconds (status) must
   never share a process with the thing that takes minutes (analysis).
2. **Document store** — S3, one prefix per tenant (`s3://bucket/{tenant_id}/{matter_id}/...`), SSE-KMS (Q6).
   *Rationale:* confidential docs are the highest-risk asset; isolating at the storage+key layer is the
   cheapest strong tenant boundary.
3. **Job/state store** — DynamoDB. **`(ours, justified — not familiarity)`** PK `tenant_id`, SK `job_id`
   serves every build-first flow (`POST /analyze`, `GET /jobs/{id}`, `GET .../report`); the idempotency key
   is its own item guarding the conditional create. The workload is **key-value, high-write-concurrency,
   atomic compare-and-swap** (job status churn), not relational reporting — DynamoDB's conditional writes
   give the single-item CAS idempotency needs, and **on-demand billing soaks write bursts without
   provisioning** (I've driven this pattern to hundreds of thousands of updates in minutes with zero
   throttle). A GSI for *list-jobs-by-matter* is purely additive — added the day a list view exists, not now.
   I'd reach for Postgres only if the audit/provenance surface turns genuinely relational — a possible
   *second* store later, not a reason to start relational.
4. **Queue** — SQS standard, one queue + DLQ. Decouples accept from execute, absorbs the launch spike, and is
   the retry/redrive substrate. *Rationale:* the spike must hit a buffer, not the workers.
5. **Execution plane** — the Part 1 orchestrator (`run_pipeline` + `run_agents` + `run_agent` +
   `apply_grounding` + `_dedupe_findings` + `score_confidence` + the memo) lifted **verbatim** onto
   **SQS→Lambda** (Fargate/ECS only for the irreducible large matter — Q3). *Rationale:* the honest core is
   the product; the MVP hosts it, it doesn't rewrite it. **`(ours)`** The orchestration the product needs
   already lives in `run_pipeline`'s `asyncio.gather` fan-out — **there is no external workflow engine: no
   Step Functions, no Temporal.** A bounded 3-agents-plus-memo fan-out does not warrant a state machine;
   adding one is ceremony over a single process. *Why Lambda over a standing Fargate worker as the default,*
   given multi-minute I/O-bound LLM calls: the common matter finishes well under the ceiling, and
   **scale-to-zero + per-message concurrency** fits a spiky, low-baseline launch better than paying for
   idle warm workers; the moment a matter is genuinely irreducible, the *same* code runs on Fargate (no
   rewrite), so I don't pre-pay for the tail.
6. **Grounding + scoring trust kernel** — `services/grounding.py` (`is_grounded`, `validate_grounding`,
   `ground_citation_quotes`) and `services/confidence.py` (`score_confidence`) are a **separate boundary on
   purpose**: deterministic, LLM-free, unit-tested independent of any model. Every finding crosses this gate,
   and the *same* `is_grounded` is what the eval harness re-runs. This is the trust boundary the whole
   legal-correctness story rests on.
7. **LLM access** — the `llm.py` seam: `call_llm_structured` with `STRUCTURED_MODEL` env-pinned, native
   structured outputs, typed `LLMOutputError` on refusal/truncation. Called from inside the worker.
8. **Eval / quality plane** — `eval/run_evals.py` + `eval/metrics.py`, promoted to a CI gate (Q7).

**`(ours)` Deliberately absent at MVP:** no workflow engine, no standing Fargate fleet (it's the
irreducible-matter exception, not a default plane), no separate vector DB, no microservice split of the four
agents, no second state store. Their absence is the design.

---

## Q3 — How an analysis moves through the system (duration-routed)

```
   client          API / control plane         SQS        Worker (Lambda)        S3 / DynamoDB
     │  presigned PUT URLs   │                   │              │                      │
     │ ─────────────────────►│                   │              │                      │
     │  (docs land in S3 directly, never transit our compute)   │                      │
     │  PUT docs ───────────────────────────────────────────────────────────────────► │ S3
     │  POST /matters/{id}/analyze               │              │                      │
     │ ─────────────────────►│ write job QUEUED  │              │                      │
     │                       │ (idempotency key) ───────────────────────────────────► │ DynamoDB
     │                       │ enqueue ─────────►│              │                      │
     │ ◄── 202 + job_id ─────│                   │ ── trigger ─►│ load manifest docs ◄─│ S3
     │                       │                   │              │ run_pipeline:        │
     │  GET /jobs/{id} ─────►│ read status ──────────────────── │  3 agents ‖ → ground │
     │ ◄── RUNNING +degraded │                   │              │  → confidence → memo │
     │                       │                   │              │ write report ──────► │ S3 (immutable)
     │                       │                   │              │ status SUCCEEDED ──► │ DynamoDB
     │  GET /jobs/{id}/report ──────────────────────────────────────────────────────► │ stream artifact
```

1. **Upload.** Client requests **presigned S3 PUT URLs** scoped to `{tenant_id}/{matter_id}`; documents land
   in S3 directly and **`(ours)` never transit our compute on upload**. The API records a manifest in
   DynamoDB. *(OCR/PDF ingest is deferred — Q8. The MVP ingests the `.txt` the pipeline already eats:
   `load_documents` produces a `{stem: content}` dict, reproduced from S3 instead of disk.)*
2. **Request.** `POST /matters/{matter_id}/analyze` (authenticated — see Q6: OIDC/JWT at the API Gateway,
   the principal resolves the tenant) validates the manifest and writes a job row with an **atomic
   conditional create** — DynamoDB `PutItem` with `ConditionExpression: attribute_not_exists(pk)`. The key is
   `hash(tenant_id, matter_id, document-content-hashes)`. **`(ours)`** The hashes are **server-authoritative**
   — taken from the S3 checksum-on-PUT (or an upload-complete ingest step writes them to the manifest), never
   client-supplied and never a synchronous S3 read-back at request time. **Idempotency is a correctness
   feature, not perf tuning** — a non-idempotent retry on a *paid legal analysis* double-bills and
   double-flags, so concurrent duplicate submits **collapse to one job** (the loser's conditional write fails
   and returns the existing `job_id`). **Stated invariant:** idempotency holds *within a deploy's model/prompt
   generation* — an identical resubmit after a model/prompt upgrade returns the prior report, by design, since
   the deployed pin only changes on deploy and there is no re-analyze endpoint yet. The day a deliberate
   re-analyze path ships, the (server-authoritative) model/prompt pin folds into the key so a re-run under a
   new pin is a new billable job — deferred until that path exists. State transitions are likewise guarded by
   `ConditionExpression` (e.g. `QUEUED→RUNNING` compare-and-set), so a **redelivered SQS message can't
   stomp a terminal status** — the exact failure I fixed in production, where a zombie redelivery overwrote
   a succeeded record until a state-transition guard ("a success may never be downgraded by a late retry")
   closed it. Then it enqueues one SQS message and returns **202 + job_id**.
3. **Execution plane — Lambda, one live plane.** **`(ours, and the lesson I lived)`** Every matter runs
   `SQS→Lambda`. The typical matter finishes well under the ~15-min ceiling; I keep **one live plane on
   purpose** — a half-finished Batch↔Lambda migration once left a feature flag that was a **no-op in prod**
   while the old plane silently carried all the load, and a dual path nobody trusts is worse than one honest
   plane. So I don't ship a second compute arm as designed-but-unbuilt machinery. **If a matter is so large
   it can't finish under the ceiling even on the largest-context model, the MVP fails it loudly with a typed
   reason** (`matter_too_large_for_v1`) rather than pretending a dark Fargate path will catch it. Lifting the
   ceiling for those rare matters — the *same* image on Fargate/ECS, or map-reduce sharding — is named and
   deferred (below), built when a real matter forces it. That trajectory (start heavier, refactor to fit
   serverless by sharding) is exactly the one I went through operating a legal-document pipeline at scale.
4. **Execute.** The Lambda materializes the manifest's docs into the `{stem: content}` dict and calls
   **`run_pipeline` unchanged**: `run_agents` fans the three agents out under `run_agent` (timeout + one
   retry + `degraded_agents`); `apply_grounding` clears every ungrounded quote to `could_not_verify`;
   `_dedupe_findings` collapses CrossDoc/QuoteAccuracy overlap; `score_confidence` scores deterministically;
   `_run_memo` runs the memo last under the same contract.
5. **Persist + report.** The worker writes the immutable `VerificationReport` JSON to
   `{tenant}/{matter}/reports/{job}.json`, flips DynamoDB to `SUCCEEDED` with the artifact pointer +
   `degraded_agents`. `GET /jobs/{id}` returns coarse status **+ `degraded_agents` as a partial-coverage
   signal**; `GET /jobs/{id}/report` streams the artifact, shape-identical to Part 1's `VerificationReport`.

**`(ours)` The first real limit is context size, not the 15-min clock.** Each finding agent is prompted over
the *whole corpus*, so a big matter exhausts the **model context window** before it exhausts Lambda's
wall-clock. The MVP answer is a **one-line `STRUCTURED_MODEL` bump** (`llm.py` seam) to a larger-context model
for an unusually big matter — config, not architecture, and for a low-hundreds-of-`.txt` MVP it rarely fires.
A matter that exceeds even the largest context is **deferred** ([below](#what-im-intentionally-not-solving-yet)):
map-reduce over shards, whose real subtlety — a cross-shard reduce pass so `CrossDocConsistencyAgent` still
sees doc-3-vs-doc-47 — I'd design when a real matter forces it, not now.

---

## Q4 — What's durable vs recomputable

**`(ours)`** Discriminator: **expensive + non-deterministic = durable; cheap + deterministic = recompute**,
with **auditability** as the tie-breaker for a legal product.

| Data | Disposition | Why |
|---|---|---|
| Uploaded documents (S3) | **Durable** | System of record; irreplaceable customer data. SSE-KMS, versioned, per-tenant prefix. |
| Job row (DynamoDB) | **Durable** | The **auditability spine** — what was asked, when, by whom, what happened, `degraded_agents`. Customers expect it. |
| Final `VerificationReport` (S3, immutable) | **Durable** | A legal deliverable a firm may cite — must be byte-reproducible and **never silently regenerated against a newer model**. Re-running spends real money and is non-deterministic at the model layer. |
| Provenance bundle (with the report) | **Durable** | `STRUCTURED_MODEL` snapshot id, prompt-bundle hash, grounding-module version, gating eval-suite version, input-doc hashes. Answers *"why did this report differ from last month?"* — the first question a firm asks. |
| Audit / access history | **Durable** | **At MVP the durable DynamoDB job row *is* the audit spine** — who requested which analysis over which matter, when, which model+version. Schema `{who, tenant_id, matter_id, action, model+version, ts}`. Tamper-evident WORM (S3 Object Lock *or* hash-chaining — one mechanism, picked when needed), a court-producible query path, and **legal-hold overriding the retention TTL** (deleting under hold = spoliation) are **deferred** until a customer's security review demands them (Q8). |
| Quality metric stream (counts/rates only) | **Durable** | Drift detection needs history. |
| Grounding result | **Recompute** | `is_grounded` is a *pure* function of (quote, source); kept inside the artifact for provenance, treated as derived. |
| Confidence band | **Recompute** | `score_confidence` is deterministic (`confidence.py`: 1 src→0.55 MED, 2→0.75 MED, 3+→0.90 HIGH; abstentions short-circuit to 0.25 LOW). Same finding → same score. |
| Judicial memo | **Recompute** | Synthesis over findings; regenerable, explicitly nullable on memo degrade. |

**Not stored at all:** document text in logs/metrics (privilege); any LLM raw chain-of-thought beyond the
structured schema. *Caveat named honestly:* "recompute" assumes the model is pinned and reproducible enough;
LLM nondeterminism means a re-run may differ, so for the artifact a firm relied on we store **the report
itself** durably and treat recompute as a recovery/debug path, never a substitute.

---

## Q5 — Where it fails first, and how to recover

**It fails first at the LLM provider.** Three reasoning calls per analysis, fanned out, under a launch spike:
the first thing to break is provider rate-limit / 429 / latency. This is the chronic throttling I lived,
where an embeddings TPM quota was a *hard ceiling* and load ran far above it. The MVP must treat the LLM as
the scarce, flaky resource.

**Recovery, layer by layer:**

```
   provider 429 / timeout
        │
        ▼  (1) run_agent: timeout + ONE retry → else neutral fallback + append degraded_agents
   one agent degrades  ──► report ships with fewer findings, HONESTLY labeled (never fabricated — gate)
        │
        ▼  (2) SQS visibility timeout + DLQ: worker dies → message redelivers → idempotency key re-runs clean
   poison job ──► DLQ after N attempts → DynamoDB FAILED + reason (never silently vanishes)
        │
        ▼  (3) spike control (DESIGN, not discovered in an incident):
   capped Lambda reserved concurrency, sized from min(RPM, TPM) provider limit
        │
        ▼  provider-wide throttle becomes a BOUNDED, graceful queue drain — not a fleet-wide meltdown
```

- **(1) Per-agent (already built).** `run_agent` gives each agent a timeout + one retry, then returns the
  neutral fallback and appends to `degraded_agents` — one throttled agent degrades coverage *transparently*.
  The grounding gate means a degraded run can never emit a fabricated finding; worst case is **fewer
  findings, honestly labeled**. `run_agent` re-raises `CancelledError` (verified) so shutdown is never
  swallowed or retried.
- **(2) Per-job.** SQS visibility timeout + DLQ. A worker that dies mid-analysis doesn't delete the message;
  SQS redelivers, and the idempotency key makes redelivery re-run cleanly without double-billing. Poison jobs
  hit the DLQ and flip DynamoDB to `FAILED` with a reason.
- **(3) Spike control — the SQS buffer + a deliberately capped Lambda reserved concurrency is the throttle.**
  I cap worker concurrency *under* the provider's published limit **on purpose**, so a launch spike becomes a
  bounded queue drain, not a fleet-wide 429 meltdown. The cap is sized from the **binding** provider limit —
  `min(RPM-bound, TPM-bound)`, and for these token-heavy multi-minute calls **TPM is usually the real
  limiter** — using the actual published numbers at launch, not a guess. Per-tenant fairness (so one firm's
  bulk upload can't starve others) is deferred until a noisy-neighbor problem actually exists — see Q8;
  the concurrency cap alone is the launch throttle.
- **The memo** (LangChain LCEL) runs through the same `run_agent` contract, so a hung memo degrades to a
  `None` memo with the structured flags still the source of truth.

**`(ours)` Not defended at MVP, named on purpose:** a full provider outage (no multi-provider failover yet —
Q8), and single-region S3/DynamoDB failure (single-region MVP — deliberate, to avoid repeating a
multi-region split that made quota/metric debugging error-prone). `run_agent` retries **exactly once** by
design — a persistently failing agent should degrade and be *visible*, not hammer the provider.

---

## Q6 — Protecting confidential documents, separating tenants

Opposing parties may be different tenants on the same plane, so a cross-tenant leak is *existential*. Tenancy
is enforced at the **cheapest strongest layer**, not in app code alone.

```
  authenticated principal ──► API edge resolves tenant context (tenant_id, matter_id) — ONCE
        │  (never re-derived from a request body downstream)
        ├──► S3 key:        {tenant_id}/{matter_id}/...        (per-tenant prefix)
        ├──► DynamoDB key:  (tenant_id, job_id)                (per-tenant partition)
        ├──► SQS message:   {tenant_id, matter_id, job_id}
        └──► Worker role:   tenant-scoped — a job for A CANNOT construct a path into B
                                   │
                            SSE-KMS + tenant-scoped role ─► a job for A cannot path into B
                                                    (per-tenant keys/crypto-shred = deferred)
```

- **Tenant identity resolved once, at the edge — never re-derived from a request body.** **OIDC/JWT at the
  API Gateway** authenticates the principal and the gateway resolves the `tenant_id`; from there it rides
  **trusted first-party infra the caller can't forge** — the API *writes* it into the SQS message and the
  DynamoDB partition key, and the worker reads it from there, never from input. *The most common multi-tenant
  breach is a downstream service trusting a `tenant_id` it got from the caller* — closed structurally by the
  tenant-scoped IAM role + partition key + prefix below, not by re-verifying a signature at each hop.
- **Storage isolation.** S3 **per-tenant prefix + bucket-level SSE-KMS**, with a **tenant-scoped IAM worker
  role**. **`(ours)`** That trio already delivers encryption-at-rest *and* the existential cross-tenant
  boundary (a job for tenant A literally cannot construct a path into B). **Per-tenant keys
  (envelope/DEK → crypto-shred, and a dedicated CMK) are deferred together** — they buy *delete-the-key =
  delete-the-tenant*, which is a security-review feature, not a launch necessity, and per-key request-rate/cost
  only bites at hundreds of busy tenants. I promote per-tenant keying when a firm's review asks; shipping it
  at MVP would be complexity ahead of the threat the prefix+role boundary already closes.
- **Request isolation.** Workers assume a tenant-scoped role; the blast radius of an agent logic bug is **one
  tenant, structurally** — not a runtime `if`-check.
- **Intra-firm ethical walls — stated, not silently skipped `(legal-domain, named)`.** Tenant isolation
  separates *firms*; it does **not** by itself wall one matter off from another *inside* a firm (an ethical
  screen). The MVP ships the `(tenant, matter, role)` unit end-to-end so matter-level ACLs are *additive*,
  but **I state plainly to a launch customer that v1 assumes all of a firm's authorized users may read that
  firm's matters** — per-user/per-matter screening (and SSO/SCIM) is deferred (Q8). The generated
  `VerificationReport` is itself privileged work product, so its read/export is access-controlled and
  audit-logged like any matter document — not a freely shareable artifact.
- **Data-in-motion to the LLM `(honest phrasing)`.** The worker **must** load plaintext into memory to call
  the model (`run_pipeline` takes a `{stem: content}` dict), so the enforceable claim is **plaintext is
  process-ephemeral — never persisted to disk or logs, isolated by a tenant-scoped role in a short-lived
  execution env with no warm-container reuse across tenants** — *not* the absolutist "never transits our
  compute" (nor a promise to scrub CPython's heap, which the runtime doesn't guarantee). All model calls go
  to the provider only under the no-training/zero-retention agreement (Q1). Part 1's per-document `uuid4`
  sentinel fencing (`prompts.py`) already stops one tenant's
  malicious document from closing a *sibling's* fence to inject instructions.
- **Logs carry counts and IDs, never content** (`tenant_id + job_id + agent + status`, matching `base.py`'s
  `agent_failed`/`agent_degraded` extras). Privilege means the observability layer must be safe to read.
- **"Can your engineers read my filing?" — internal access is a first-class answer `(legal-domain)`.**
  Operator access to tenant data is **default-deny, break-glass only** (time-boxed, approved, and written to
  the same audit log a customer can see), and the LLM provider is disclosed as a **named sub-processor** with
  its zero-retention term — so the privilege story covers *insiders and the vendor*, not just other tenants.
- **Retention `(ours)`.** Per-tenant configurable TTL with hard delete on matter close — holding a privileged
  document past need is itself a waiver risk. (Crypto-shred via per-tenant keys is the *stronger* delete,
  deferred with per-tenant keying above; TTL hard-delete is the launch mechanism.)

**Logical isolation by default; physical (dedicated account/VPC) as a paid enterprise tier** — offered to the
few firms whose security review demands a visibly separate blast radius, *not* defaulted, because per-tenant
infra doesn't scale to tens of thousands and well-built logical isolation defends the same threat.

---

## Q7 — Knowing it's correct, healthy, improving

Three different questions, three instruments — and the hard-won rule: **observability AS CODE.**

- **CORRECT (is the output trustworthy?).** The Part 1 eval harness *is* the MLOps face — `eval/run_evals.py`
  scores precision/recall/grounding against hand-frozen gold sets, with an honest **sub-100% recall** (it
  ships a deliberately-uncaught arithmetic flaw), a **pending-adjudication** bucket, **Wilson CIs**, and a
  **pre/post-gate fabrication ablation**. **`(ours)`** Wire it as **eval-as-CI with the authority of a
  failing unit test**: it runs on every change to a prompt, the model pin (`STRUCTURED_MODEL`), or the
  grounding/confidence code, and a recall regression / a rise in pre-gate fabrication / any ungrounded quote
  on the shipped report **blocks the deploy**. We read the **k/n fractions and CIs, not a point %** — the
  harness itself says so and discloses its grounding-consistency rate is a **regression guard, not an
  independent hallucination oracle**. It runs offline against committed snapshots with **zero API spend**.
- **HEALTHY (is the system up?).** CloudWatch on managed primitives: SQS **age-of-oldest-message** (the real
  "falling behind" signal), **DLQ depth** (must alarm — failed customer work), Lambda errors/throttles/
  duration, DynamoDB throttles, and the **LLM 429/timeout rate from
  `run_agent`'s existing degraded logging**. `degraded_agents` flowing into a metric gives **coverage health
  in production**. I do **not** ship an online "grounding probe" as a
  health signal — re-running the deterministic `is_grounded` over already-gated output is tautologically
  ~100%, so it can never fire. `is_grounded` is shared with the eval **only** for the CI gate.
- **IMPROVING.** Track recall/precision over time from the eval; track the real-world `degraded_agents` and
  DLQ rates release-over-release; track the **`could_not_verify` (abstention) rate as a headline metric, not
  an embarrassment** — a rising abstention rate signals a more cautious model or a corpus the gate can't
  handle. New pending-adjudication findings feed gold-set growth, so the eval gets stronger from production.

**The lesson baked in (what I'd do better).** A dashboard built by hand in the console — *not* in Terraform —
broke silently when a deploy renamed log phrases: metric filters went blind without telling anyone. So for
the MVP **every alarm, filter, and dashboard is Terraform from day one, and I alarm on the ABSENCE of
expected datapoints**: *"no successful analyses in N minutes during business hours"* fires; *"the
grounding-check-ran metric went to zero"* pages — because a metric that silently goes to zero looks identical
to "healthy," and that silent silence is the failure mode that actually burned me.

**Privilege-preserving by construction `(ours)`:** the metric stream stores **counts and rates only**, never
document text or quote bodies; gold sets stay **synthetic/curated**, so no privileged matter is ever baked
into a CI artifact or dashboard.

### Cost is a first-class metric, because the LLM dominates it

**`(ours)`** For a paid product, **$/matter** belongs on the same dashboard as latency and recall. Model
tokens dwarf S3/DynamoDB/Lambda here, so cost is essentially **(corpus size + 3 agents + memo) × per-token
price** for the typical matter; the rare larger-context fallback costs more per token *and* more tokens —
which is exactly why it isn't the default. I track the **ranking and its driver**, not a fabricated dollar
figure (the absolute number is the provider's per-token price, which I won't invent). I wire the metric now —
the `llm.py` seam sums per-call token usage onto the job row — and **measure** turnaround and $/matter from
day one; a *committed* SLO and any usage-based billing ledger wait for a real design-partner baseline.
**`(ours)` Defensible starting hypothesis from the verified numbers:** an agent call is ~50–90s and three
fan out in parallel under a 120s timeout, so a typical matter should land **p95 ≈ a few minutes** (the memo
adds one more call); that's the target I'd validate against real traffic, not a number I'd commit blind.

---

## Q8 — What to build first, what to defer

**BUILD FIRST — the smallest honest increment (one matter, async, isolated, observable):**

1. **Split the synchronous `POST /analyze`** into accept (`202 + job_id`) + `GET /jobs/{id}` status +
   `GET /jobs/{id}/report`. The single load-bearing change — it makes long jobs survivable.
2. **S3 (per-tenant prefix + SSE-KMS + tenant-scoped role) + presigned upload; DynamoDB job table with atomic
   conditional idempotency; SQS + DLQ; the Part 1 orchestrator hosted unchanged on `SQS→Lambda`** with a
   **capped reserved concurrency** as the spike throttle, and a **one-line larger-context model fallback** on
   the `llm.py` seam for an unusually big matter. (Per-tenant token budgets, map-reduce, and a second compute
   plane are deliberately *not* here — see deferred.)
3. **Tenant isolation** at the storage/key/partition layer + `(tenant_id, matter_id, role)` minted once at
   the edge and threaded end-to-end.
4. **Terraform for all of it, INCLUDING the CloudWatch alarms** — observability-as-code from line one.
5. **eval-as-CI** gating any prompt/model-pin/grounding change.

This is shippable to design-partner firms: upload a matter, poll status, get a reproducible, grounded,
confidence-scored report, data isolated.

**DEFER, with reasons** — full list in [What I'm intentionally not solving yet](#what-im-intentionally-not-solving-yet). The headlines: per-tenant token budgets + map-reduce sharding + a second compute plane (all
deferred to "a real big/noisy matter forces it"); OCR/PDF ingest; tamper-evident WORM audit + legal-hold;
multi-provider failover; a vector store; tens-of-thousands scale, intra-tenant RBAC/SSO/ethical-wall ACLs, BYOK.

---

## What I'm intentionally NOT solving yet

Named on purpose, so the plan survives a skeptical reviewer — and to keep the MVP lean rather than gold-plated.

- **A matter too large to finish under the Lambda ceiling fails loudly at MVP, and is solved later — never
  silently capped.** The MVP returns a typed `matter_too_large_for_v1` rather than carrying an unbuilt
  Fargate arm (a dark dual path would repeat the half-migrated-no-op-flag scar). Lifting the ceiling — the
  *same* image on Fargate/ECS, and/or **map-reduce over document shards** (whose cross-shard reduce pass for
  `CrossDocConsistencyAgent` needs a real recall-gated design) — is built when a real matter forces it, not
  speculatively.
- **Per-tenant fairness (the atomic token bucket) is deferred.** The capped concurrency throttle protects the
  provider at launch; a per-tenant CAS budget (and any billing meter built on it) waits for an actual
  noisy-neighbor or usage-pricing need. Building it now is complexity ahead of the problem.
- **OCR / PDF ingest is deferred.** The MVP eats `.txt` exactly as `load_documents` produces. When added:
  Textract **native async** is the routing target (not a custom poller), the **sync path ships first for the
  ~73% small docs** (the async path's high latency was the *tail*, not the common case), and normalized text
  is **content-hash-keyed** so re-ingest is a cache hit.
- **No multi-provider / multi-region.** Single LLM provider (under zero-retention), single AWS region.
  Provider failover is a later change on the `llm.py` seam — scoped honestly above. One region on purpose, to
  avoid repeating a multi-region split that made quota/metric debugging error-prone.
- **No entailment / NLI faithfulness checking.** Grounding stays the deterministic **verbatim-attribution**
  check it is today (`grounding.py` documents this and names MiniCheck/RAGAS as the upgrade path). It catches
  *fabricated quotes*, not *unsupported inferences built from real words* — acknowledged future work, out of
  the launch budget because it trades determinism/auditability (which legal review wants) for coverage we
  don't need to launch.
- **No case-law existence lookup.** The tool verifies internal consistency and quote fidelity; it does not
  confirm a real authority says what it's cited for. `could_not_verify` is the honest verdict, not
  "verified." Real existence checking (CourtListener/eyecite) is deferred.
- **No intra-tenant authorization sophistication at launch.** Tenant-level isolation ships; `(tenant, matter,
  role)` is *named* so it's additive, but per-user RBAC, SSO/SCIM, matter-level ethical-wall ACLs, and BYOK
  are deferred. Adding Postgres row-level security alongside DynamoDB at MVP is a second state store nobody
  asked for — the partition-key + per-tenant-KMS boundary defends the same threat.
- **Per-tenant keying (envelope/DEK *and* a dedicated CMK) is deferred together.** Launch ships bucket-level
  SSE-KMS + per-tenant prefix + tenant-scoped role — encryption-at-rest and the cross-tenant boundary. The
  per-tenant key (which buys *delete-the-key = crypto-shred a tenant*) is a security-review feature, not a
  launch necessity, and per-key request-rate/cost only bites at hundreds of busy tenants — so it is promoted
  when a firm's review asks, not built speculatively.
- **No streaming / partial-results UX.** Status is coarse (`QUEUED/RUNNING/SUCCEEDED/FAILED` +
  `degraded_agents`); per-agent live streaming is deferred.
- **No model training / fine-tuning for this MVP.** It's an LLM-API product; the eval harness is the MLOps
  surface. Correctness comes from a deterministic grounding gate, not model weights — training would be
  over-engineering here.
- **The one honesty gap I will NOT pretend to have closed:** prompt injection that **suppresses a true
  finding** has **no structural backstop.** The per-document `uuid4` sentinel fencing in `prompts.py` is an
  **asymmetric defense** — it stops a malicious document forging or corrupting a *sibling's* finding, but it
  cannot stop a document that talks the model *out of* raising a true flag. The honest production answer is
  **detection, not prevention**: a **recall-floor canary** — fold a synthetic known-flaw document into the
  *scheduled* eval run and alert if the pipeline stops catching it (the eval's own recall metric on a
  schedule, not new infra). Knowing the limit of my own verbatim-grounding gate is the point.
