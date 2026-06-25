# Production Readiness Plan

This is Part 2 of the BS Detector challenge. Treat it as a standalone system design challenge, not as a short reflection or README appendix.

Part 1 asks you to build the first working version of BS Detector. Part 2 asks you to describe how that prototype should become an MVP that can support real customers at startup scale.

We do not expect you to build this production system during the take-home. We do expect a clear, opinionated plan that shows how you think about architecture, AI workflow orchestration, database and storage design, infrastructure, security, sequencing, operational risk, and tradeoffs.

## Scenario

Assume BS Detector is moving from a prototype to a paid MVP for law firms and legal teams.

- Users upload litigation documents and request verification reports.
- A single matter may contain dozens to hundreds of documents.
- Analyses may take minutes and may involve multiple model calls, retrieval steps, citation checks, and cross-document consistency checks.
- Multiple tenants must be isolated from one another.
- Customers expect status updates, auditability, and reliable handling of long-running jobs.
- Usage can spike: assume hundreds of simultaneous users at launch, with a path to tens of thousands of users and high-volume document ingestion.
- Legal data may be confidential, privileged, or otherwise sensitive.

If you choose different scale assumptions, state them and explain why.

## What To Submit

Create `docs/production-readiness.md` or an equivalent document in your submission.

Your plan should be specific to BS Detector. It should not read like a generic SaaS scaling template. We are intentionally leaving room for you to choose assumptions, architecture, sequencing, and tradeoffs.

Your document should answer the questions you believe are most important. Strong plans will usually address questions like:

- What assumptions are you making about users, scale, latency, reliability, and risk?
- What are the major system components, and why do those boundaries make sense?
- How does an analysis move through the system from document upload to final report?
- What data must be stored durably, and what data can be recomputed?
- Where do you expect the system to fail first, and how would you recover?
- How would you protect confidential customer documents and separate tenants?
- How would you know whether the system is correct, healthy, and improving?
- What would you build first, what would you defer, and why?

Use diagrams, tables, or sketches if they clarify your thinking. Do not try to cover every possible production concern. Choose the concerns that matter most for this product and defend those choices.

Strong submissions usually have a clear opinion about the first production increment, the biggest risks, and the parts of the architecture that should remain flexible because the product is still early.

## Follow-Up Interview

In the follow-up interview, be prepared to defend your plan. We will ask why you made specific choices, what assumptions would change your design, and how you would move from your prototype to the first production-ready version.

We are not looking for a perfect enterprise architecture. We are looking for crisp judgment, explicit assumptions, and a plan that could survive contact with real users.
