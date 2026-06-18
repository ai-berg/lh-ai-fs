"""Agent protocol and a resilient runner shared by every pipeline agent.

Each agent is an async callable that takes the document corpus and returns a
typed result. ``run_agent`` wraps a single agent with a timeout, one retry on
transient failure, and a neutral fallback so a misbehaving agent degrades the
report (recorded in ``degraded_agents``) instead of crashing the pipeline.
"""

import asyncio
import logging
from typing import Awaitable, Callable, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Generous enough for a reasoning model (GPT-5.5) running a chain-of-thought
# rubric over the full case file: a single citation-audit pass measured ~50s, so
# 45s was too tight. This bounds a stuck call without cutting off normal work.
DEFAULT_TIMEOUT_SECONDS = 120.0


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
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    retries: int = 1,
) -> T:
    """Run one agent resiliently.

    Retries once on timeout/exception, then returns ``fallback`` and appends
    ``name`` to ``degraded`` so the report stays transparent about coverage.
    """
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
