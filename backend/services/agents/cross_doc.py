"""Cross-Document Consistency Agent (Tier 2).

Compares factual assertions in the MSJ against the reference documents (police
report, medical records, witness statement) and reports contradictions, each
backed by a verbatim quote from the source. Findings are grounded downstream:
a quote that does not literally exist collapses the finding to could_not_verify.
"""

import logging

from llm import call_llm_structured
from prompts import CROSS_DOC_SYSTEM, build_messages
from repositories.document_repository import MSJ_DOC
from schemas import CrossDocOutput, Finding

logger = logging.getLogger(__name__)

AGENT_NAME = "CrossDocConsistencyAgent"


async def check_cross_doc_consistency(docs: dict[str, str]) -> list[Finding]:
    """Return contradictions between the MSJ and the reference documents."""
    msj = docs.get(MSJ_DOC, "")
    references = {name: text for name, text in docs.items() if name != MSJ_DOC}
    if not msj or not references:
        logger.warning("cross_doc_missing_inputs")
        return []

    # Pass each reference document as its own fenced unit (own sentinel), so a
    # malicious reference can't forge a sibling document's boundary.
    messages = build_messages(CROSS_DOC_SYSTEM, msj=msj, **references)
    result = await call_llm_structured(messages=messages, schema=CrossDocOutput)

    # Stamp provenance ourselves rather than trusting the model to fill it.
    for finding in result.findings:
        finding.raised_by = AGENT_NAME

    logger.info("cross_doc_done", extra={"count": len(result.findings)})
    return result.findings
