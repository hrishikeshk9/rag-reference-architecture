"""
LLM generation with citation extraction.

Uses structured output (tool_use) to separate the prose answer from explicit
citations — avoids brittle regex parsing of inline [chunk_id] markers.
The model fills the `answer` and `citations` fields in the tool schema.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import anthropic

from src.generate.prompt import GenerationPrompt

logger = logging.getLogger(__name__)

ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your answer and the chunk IDs that support each claim.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The complete answer to the user's question.",
            },
            "citations": {
                "type": "array",
                "description": "List of chunk IDs that directly support claims in the answer.",
                "items": {"type": "string"},
            },
            "answer_found": {
                "type": "boolean",
                "description": "False if the context lacked enough information to answer fully.",
            },
        },
        "required": ["answer", "citations", "answer_found"],
    },
}


@dataclass
class GenerationResult:
    answer: str
    citations: list[str]
    answer_found: bool
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    prompt_version: str
    raw_chunks: list[dict] = field(default_factory=list)


class RAGGenerator:
    """
    Calls Anthropic Claude with structured tool_use for citation enforcement.

    Structured output via tool_use guarantees a parseable response: the LLM
    fills `answer` and `citations` fields rather than embedding markers in prose.
    This eliminates the class of "hallucinated citation" where the model invents
    a chunk_id that doesn't exist.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 1024,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    async def generate(
        self, prompt: GenerationPrompt, chunk_metadata: list[dict]
    ) -> GenerationResult:
        """Generate a grounded answer with citations from the assembled prompt."""
        import time
        import asyncio

        t0 = time.time()
        loop = asyncio.get_event_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
                tools=[ANSWER_TOOL],
                tool_choice={"type": "tool", "name": "submit_answer"},
            ),
        )

        latency_ms = (time.time() - t0) * 1000

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            logger.error(f"No tool_use block in response: {response.content}")
            return GenerationResult(
                answer="Error: model did not return a structured answer.",
                citations=[],
                answer_found=False,
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                prompt_version=prompt.prompt_version,
            )

        data = tool_block.input
        return GenerationResult(
            answer=data["answer"],
            citations=data.get("citations", []),
            answer_found=data.get("answer_found", True),
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            prompt_version=prompt.prompt_version,
            raw_chunks=chunk_metadata,
        )
