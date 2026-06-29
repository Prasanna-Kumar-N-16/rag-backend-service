"""Anthropic Claude streaming client with robust error handling and retry.

Design decisions
----------------
* Uses ``client.messages.stream()`` + ``get_final_message()`` for every call
  so large ``max_tokens`` values never hit SDK HTTP timeouts.
* Retry strategy: tenacity exponential backoff on transient errors
  (429 RateLimitError, ≥500 InternalServerError, APIConnectionError).
  Non-retryable 4xx errors (BadRequest, Auth, PermissionDenied, NotFound)
  are re-raised immediately.
* Model and max_tokens are config-driven so they are swappable via env vars.
"""

from __future__ import annotations

from typing import cast

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.logging import get_logger

logger = get_logger(__name__)

# Errors that are safe to retry
_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)


def _make_retry_decorator() -> retry:  # type: ignore[valid-type]
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )


class ClaudeClient:
    """Thin async wrapper around the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for generation.")
        self._model = settings.generation_model
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> str:
        """Stream a response from Claude and return the complete text.

        Args:
            system_prompt: Operator-level instructions (grounding, citation format).
            user_prompt: The user's question with injected context chunks.
            max_tokens: Maximum output tokens (streamed to avoid HTTP timeouts).

        Returns:
            The generated answer text.

        Raises:
            anthropic.BadRequestError: Invalid request (not retried).
            anthropic.AuthenticationError: Bad API key (not retried).
            anthropic.RateLimitError: After all retries exhausted.
            anthropic.InternalServerError: After all retries exhausted.
        """
        return cast(str, await self._generate_with_retry(system_prompt, user_prompt, max_tokens))

    @_make_retry_decorator()  # type: ignore[misc]
    async def _generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> str:
        logger.debug("claude_generate_start", model=self._model, max_tokens=max_tokens)
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                message = await stream.get_final_message()

            text = next(
                (block.text for block in message.content if block.type == "text"),
                "",
            )
            logger.info(
                "claude_generate_done",
                model=self._model,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                stop_reason=message.stop_reason,
            )
            return text

        except (
            anthropic.BadRequestError,
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
        ):
            # Non-retryable — propagate immediately without tenacity retry.
            raise
