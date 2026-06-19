"""Agent protocol and a resilient runner shared by every pipeline agent.

Each agent is an async callable that takes the document corpus and returns a
typed result. ``run_agent`` wraps a single agent with a timeout, one retry on
transient failure, and a neutral fallback so a misbehaving agent degrades the
report (recorded in ``degraded_agents``) instead of crashing the pipeline.
"""

import asyncio
import logging
import os
from typing import Awaitable, Callable, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Per-agent timeout. 120s is sized for a GPT-5.5 reasoning call with field-ordered
# chain-of-thought over the full case file (measured ~50-90s); overridable via
# AGENT_TIMEOUT_SECONDS for a larger corpus. It bounds EACH agent, and the fan-out
# agents run concurrently, so the fan-out wall-clock is ~one agent, not the sum — the
# sequential memo step adds its own bounded slice on top.
def _timeout_seconds() -> float:
    # Read at the CALL SITE (not frozen at import) so the value reflects the env AFTER
    # any lazy load_dotenv() — on a host run the .env is loaded when llm.py first runs,
    # which is later than this module's import. Parse defensively so a malformed
    # override falls back to the default instead of crashing.
    try:
        return float(os.getenv("AGENT_TIMEOUT_SECONDS", "120") or "120")
    except ValueError:
        logger.warning("invalid AGENT_TIMEOUT_SECONDS override; using default 120")
        return 120.0


class Agent(Protocol):
    """An agent: a named async function over the document corpus."""

    name: str

    async def __call__(self, docs: dict[str, str]) -> object: ...


async def run_agent(
    name: str,
    call: Callable[[], Awaitable[T]],
    *,
    fallback: T,
    degraded: list[str],
    timeout: float | None = None,
    retries: int = 1,
) -> T:
    """Run one agent resiliently.

    Retries once on timeout/exception, then returns ``fallback`` and appends
    ``name`` to ``degraded`` so the report stays transparent about coverage.
    """
    if timeout is None:
        timeout = _timeout_seconds()  # read here so a host .env override takes effect
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return await asyncio.wait_for(call(), timeout=timeout)
        except asyncio.CancelledError:
            # Request/shutdown cancellation must propagate, never be swallowed or
            # retried. (On Python 3.8+ CancelledError is a BaseException and would
            # already bypass the handler below; re-raising makes that explicit.)
            raise
        except Exception as exc:  # noqa: BLE001 — any agent failure must be contained
            logger.warning(
                "agent_failed", extra={"agent": name, "attempt": attempt, "error": str(exc)}
            )

    logger.error("agent_degraded", extra={"agent": name})
    degraded.append(name)
    return fallback
