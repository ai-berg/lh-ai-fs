"""Judicial-Memo Agent (Tier 3) — the ONLY LangChain-based agent.

WHY LangChain here and SDK everywhere else: the memo is a single synthesis step —
take the confirmed findings, produce one paragraph — which is the canonical use case
for an LCEL chain (`prompt | llm | structured_output`). The fan-out agents, by
contrast, are independent parallel calls with no chaining, retrieval, or tool loop,
so a framework there would be pure overhead. Using LangChain exactly where it pays
(and saying so) demonstrates the tool without distorting the architecture.

The memo is decision SUPPORT, not displacement: it summarizes what the audit found
and how certain it is, and never opines on how to rule — mirroring a bench memo and
the product's "help judges focus on judgment" framing. `grounded_in` ties the prose
back to the specific findings, so the memo stays traceable to the structured flags.
"""

import logging

from schemas import Finding, JudicialMemo, VerificationStatus

logger = logging.getLogger(__name__)

AGENT_NAME = "JudicialMemoAgent"

_MEMO_SYSTEM = """You are a judicial law clerk preparing a neutral bench memo for a judge.
You are given the CONFIRMED findings of an automated audit of a Motion for Summary
Judgment — contradictions and quote/citation defects, each already grounded in the
record and assigned a confidence band.

Write a SINGLE concise paragraph (4-6 sentences) for the judge that:
- States, in plain language, what the audit found wrong with the brief.
- Leads with the highest-confidence, most material findings; mentions lower-confidence
  ones as such ("the audit less confidently flags ...").
- Is faithful to the findings: do not introduce any defect not in the list, and do not
  overstate certainty beyond the confidence bands given.
- Does NOT opine on the merits, who should win, or how to rule. You summarize the
  audit's findings to support the judge's own judgment; you do not make it.

Return only the memo paragraph in `summary`."""

_MEMO_HUMAN = """Confirmed findings (claim — flag_type — confidence band — explanation):
{findings_block}"""


def _select(findings: list[Finding]) -> list[Finding]:
    """Findings worth putting before a judge: confirmed (assertive) ones, strongest
    first. An abstention (could_not_verify) asserts no defect, so it never feeds the
    memo — a memo of "things we couldn't confirm" would mislead, not inform."""
    confirmed = [f for f in findings if f.status == VerificationStatus.CONTRADICTED]
    # Strongest confidence first so the prompt's "lead with the most material" lands;
    # findings without a score sort last (None treated as 0).
    confirmed.sort(key=lambda f: (f.confidence.value if f.confidence else 0.0), reverse=True)
    return confirmed


def _findings_block(findings: list[Finding]) -> str:
    lines = []
    for f in findings:
        band = f.confidence.band if f.confidence else "unscored"
        lines.append(f"- {f.msj_claim} — {f.flag_type} — {band} — {f.explanation}")
    return "\n".join(lines)


def _build_chain():
    """Construct the LCEL synthesis chain. Imported lazily so tests that monkeypatch
    `_run_chain` never need LangChain or an API client."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    from llm import STRUCTURED_MODEL

    prompt = ChatPromptTemplate.from_messages([("system", _MEMO_SYSTEM), ("human", _MEMO_HUMAN)])
    # with_structured_output binds JudicialMemo as the response schema — the LangChain
    # equivalent of the SDK's structured parse, so the memo comes back typed, not as
    # free text we'd have to parse. No temperature pin (reasoning model rejects it).
    llm = ChatOpenAI(model=STRUCTURED_MODEL).with_structured_output(JudicialMemo)
    return prompt | llm


async def _run_chain(payload: dict) -> JudicialMemo:
    """Invoke the LCEL chain. Isolated so tests can replace just this seam."""
    chain = _build_chain()
    return await chain.ainvoke(payload)


async def write_judicial_memo(findings: list[Finding]) -> JudicialMemo | None:
    """Synthesize the confirmed findings into a one-paragraph memo, or None.

    Returns None when there is nothing confirmed to report — a judge should get a memo
    only when the audit actually found defects, never an empty or speculative one.
    """
    selected = _select(findings)
    if not selected:
        logger.info("judicial_memo_skipped_no_confirmed_findings")
        return None

    memo = await _run_chain({"findings_block": _findings_block(selected)})

    # Fill grounded_in ourselves from the selected findings rather than trusting the
    # model to echo them — keeps the memo provably tied to the structured flags.
    return memo.model_copy(update={"grounded_in": [f.msj_claim for f in selected]})
