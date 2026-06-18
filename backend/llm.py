import os
from functools import lru_cache
from typing import TypeVar

from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Flagship model for the structured-output pipeline. GPT-5.5 has strong legal
# reasoning and excellent structured-output support. Overridable via env so an
# operator can pin a snapshot or fall back without a code change. Reasoning
# models reject temperature != 1, so we don't pin temperature and rely on
# structured outputs + the factual task for stability.
STRUCTURED_MODEL = os.getenv("STRUCTURED_MODEL", "gpt-5.5")

# Cap the structured output so a long brief can't silently truncate the citation
# list. Sized for the worst-case extraction (every authority + every cross-doc
# finding with reasoning) well above the observed run, and configurable so an
# operator can raise it for a larger corpus without a code change. Without a cap, a
# truncated parse raises LengthFinishReasonError and the WHOLE agent degrades to []
# — better to fail loudly with a typed error the orchestrator can record.
MAX_COMPLETION_TOKENS = int(os.getenv("STRUCTURED_MAX_TOKENS", "8000"))

T = TypeVar("T", bound=BaseModel)


class LLMOutputError(RuntimeError):
    """The model returned no usable structured output (refusal or truncation).

    Raised so the orchestrator's per-agent resilience records a DEGRADED agent
    instead of an AttributeError on a None `.parsed` masquerading as a clean run.
    """


@lru_cache(maxsize=1)
def _async_client() -> AsyncOpenAI:
    """Lazily build the async client.

    Built on first use (not at import) so a missing OPENAI_API_KEY surfaces
    inside the request — where the orchestrator's per-agent resilience can catch
    it and degrade gracefully — instead of crashing the app at import.
    """
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def call_llm_structured(
    messages: list[dict],
    schema: type[T],
    model: str = STRUCTURED_MODEL,
) -> T:
    """Call the OpenAI API and parse the reply into ``schema`` (Pydantic).

    Uses native structured outputs so the provider guarantees the response
    matches the schema — no brittle JSON parsing. We don't pin ``temperature``
    because reasoning models reject non-default values; the schema constraint and
    the factual task keep outputs stable.

    Raises ``LLMOutputError`` on a refusal or a length-truncated reply, so the
    caller degrades the agent explicitly rather than dereferencing a ``None``
    ``.parsed`` (which the ``-> T`` annotation would otherwise quietly violate).
    """
    response = await _async_client().beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=schema,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    choice = response.choices[0]
    message = choice.message

    # A model refusal comes back as a populated `.refusal` with `.parsed` None —
    # surface it as a typed error, don't treat it as an empty result.
    if getattr(message, "refusal", None):
        raise LLMOutputError(f"model refused to answer: {message.refusal}")
    # finish_reason == "length" means the structured output was truncated mid-object;
    # `.parsed` is None and any downstream `.citations`/`.findings` access would crash.
    if choice.finish_reason == "length":
        raise LLMOutputError(
            "structured output truncated (finish_reason=length); raise "
            "STRUCTURED_MAX_TOKENS for this corpus"
        )
    if message.parsed is None:
        raise LLMOutputError("no parsed structured output returned")
    return message.parsed
