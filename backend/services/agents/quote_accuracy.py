"""Quote-Accuracy Agent (Tier 3).

The fourth agent, with a role DISJOINT from the other three: it checks whether the
passages the MSJ quotes from the case-file documents are faithful to their source —
words quietly removed, inserted, or altered — which is a textual-fidelity judgment,
not the legal-merits judgment CitationAuditAgent makes nor the fact-vs-fact check
CrossDocConsistencyAgent makes. It emits Findings (flag_type=quote_altered) so its
output flows through the SAME grounding gate, confidence scoring, and report shape as
every other finding — one contract, four agents.

Scope: only quotes the MSJ draws from documents we POSSESS (the case file), because
accuracy requires the original to compare against. Quotes of the fictional case-law
authorities have no source text here, so they stay with the citation stream.
"""

import logging

from llm import call_llm_structured
from prompts import QUOTE_ACCURACY_SYSTEM, build_messages
from repositories.document_repository import MSJ_DOC
from schemas import Finding, QuoteAccuracyOutput

logger = logging.getLogger(__name__)

AGENT_NAME = "QuoteAccuracyAgent"


async def check_quote_accuracy(docs: dict[str, str]) -> list[Finding]:
    """Return quote-fidelity flaws: MSJ quotations that misrepresent their source."""
    msj = docs.get(MSJ_DOC, "")
    references = {name: text for name, text in docs.items() if name != MSJ_DOC}
    # Need both the MSJ (the quotes) and at least one source (to compare against);
    # without either there is nothing to check, so don't spend a model call.
    if not msj.strip() or not references:
        logger.warning("quote_accuracy_missing_inputs")
        return []

    # Same fencing as the cross-doc agent: each source document is its own fenced unit
    # so an injection in one can't forge another's boundary.
    messages = build_messages(QUOTE_ACCURACY_SYSTEM, msj=msj, **references)
    result = await call_llm_structured(messages=messages, schema=QuoteAccuracyOutput)

    # Promote drafts to Findings with provenance (drafts carry no confidence — scored
    # downstream; see schemas.FindingDraft).
    findings = [draft.to_finding(AGENT_NAME) for draft in result.findings]

    logger.info("quote_accuracy_done", extra={"count": len(findings)})
    return findings
