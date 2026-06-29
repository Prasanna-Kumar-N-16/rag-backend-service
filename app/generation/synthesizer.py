"""RAG answer synthesizer.

Assembles a grounded prompt from reranked context chunks, calls Claude, and
returns the answer with structured source citations.

Citation format
---------------
Context chunks are injected as numbered passages [1]...[N]. The system prompt
instructs Claude to cite inline using those numbers. The synthesizer returns
both the answer text and the source metadata so the caller can render citations.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.claude_client import ClaudeClient
from app.logging import get_logger
from app.retrieval.reranker import RankedResult

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are a precise, grounded question-answering assistant.
Answer the user's question using ONLY the numbered context passages provided.
Cite relevant passages inline with their number, e.g. [1] or [2].
If the answer cannot be found in the provided context, say so clearly.
Do not add information beyond what the passages contain.
Keep your answer concise and well-structured."""


@dataclass(frozen=True)
class Source:
    """Metadata for a single cited source chunk."""

    citation_number: int
    source_key: str
    chunk_index: int
    content_preview: str  # first 200 chars for display


@dataclass(frozen=True)
class SynthesisResult:
    """The complete result of the RAG generation step."""

    answer: str
    sources: list[Source]
    model: str


class Synthesizer:
    """Orchestrates prompt construction → Claude generation → citation extraction."""

    def __init__(self, client: ClaudeClient, model: str) -> None:
        self._client = client
        self._model = model

    async def synthesize(
        self,
        query: str,
        ranked_results: list[RankedResult],
        max_tokens: int = 2048,
    ) -> SynthesisResult:
        """Generate a grounded answer from *query* and *ranked_results*.

        Args:
            query: The user's question.
            ranked_results: Reranked context chunks (top-k from Day 3 pipeline).
            max_tokens: Max tokens for the Claude response.

        Returns:
            :class:`SynthesisResult` with answer text and source list.
        """
        if not ranked_results:
            return SynthesisResult(
                answer="I could not find relevant information to answer your question.",
                sources=[],
                model=self._model,
            )

        context_block = "\n\n".join(
            f"[{i + 1}] {result.content}" for i, result in enumerate(ranked_results)
        )
        user_prompt = f"Context passages:\n\n{context_block}\n\nQuestion: {query}"

        logger.info("synthesize_start", query_len=len(query), context_chunks=len(ranked_results))

        answer = await self._client.generate(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )

        sources = [
            Source(
                citation_number=i + 1,
                source_key=result.source_key,
                chunk_index=result.chunk_index,
                content_preview=result.content[:200],
            )
            for i, result in enumerate(ranked_results)
        ]

        logger.info("synthesize_done", answer_len=len(answer), sources=len(sources))
        return SynthesisResult(answer=answer, sources=sources, model=self._model)
