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

T = TypeVar("T", bound=BaseModel)


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
    """
    response = await _async_client().beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=schema,
    )
    return response.choices[0].message.parsed
