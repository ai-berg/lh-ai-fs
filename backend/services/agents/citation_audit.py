"""Citation Audit Agent (Tier 1).

Extracts every legal authority cited in the MSJ and assesses, on internal
consistency and plausibility alone, whether each authority supports the
proposition the brief attributes to it, flagging overstatements, altered
quotes, and unsupported citations.
It reads only the MSJ; it does not touch the reference documents.
"""

import logging

from llm import call_llm_structured
from prompts import CITATION_AUDIT_SYSTEM, build_messages
from repositories.document_repository import MSJ_DOC
from schemas import Citation, CitationAuditOutput

logger = logging.getLogger(__name__)


async def audit_citations(docs: dict[str, str]) -> list[Citation]:
    """Return all citations found in the MSJ with a support assessment each."""
    msj = docs.get(MSJ_DOC, "")
    if not msj:
        logger.warning("citation_audit_no_msj")
        return []

    messages = build_messages(CITATION_AUDIT_SYSTEM, msj=msj)
    result = await call_llm_structured(messages=messages, schema=CitationAuditOutput)
    logger.info("citation_audit_done", extra={"count": len(result.citations)})
    return result.citations
