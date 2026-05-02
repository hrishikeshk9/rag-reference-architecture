"""
LLM-as-judge for faithfulness and relevance scoring.

Uses structured output (tool_use) to force scores into a typed schema.
Prompt version is tracked by SHA-256 of the prompt file content — any prompt
edit creates a new version, so eval results are always paired with the exact
prompt that generated them.

See ADR 0003 for eval strategy rationale.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

JUDGE_TOOL = {
    "name": "submit_score",
    "description": "Submit your evaluation scores for this answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithfulness_score": {
                "type": "number",
                "description": "0.0–1.0. Does every claim in the answer appear in the context? 1.0 = fully grounded, 0.0 = entirely hallucinated.",
            },
            "relevance_score": {
                "type": "number",
                "description": "0.0–1.0. Does the answer address the question? 1.0 = fully answers the question.",
            },
            "faithfulness_reasoning": {"type": "string"},
            "relevance_reasoning": {"type": "string"},
        },
        "required": ["faithfulness_score", "relevance_score", "faithfulness_reasoning", "relevance_reasoning"],
    },
}


@dataclass
class JudgeResult:
    faithfulness: float
    relevance: float
    faithfulness_reasoning: str
    relevance_reasoning: str
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int


class LLMJudge:
    """
    Claude-based judge for RAG evaluation.

    Uses `tool_choice={"type": "tool", "name": "submit_score"}` to guarantee
    structured output. Random sampling (temperature > 0) would introduce variance
    in scores; temperature=0 makes scores deterministic and comparable across runs.
    """

    FAITHFULNESS_PROMPT_PATH = Path(__file__).parent / "prompts" / "faithfulness.txt"
    RELEVANCE_PROMPT_PATH = Path(__file__).parent / "prompts" / "relevance.txt"

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self._client = anthropic.Anthropic()
        self._faithfulness_prompt = self.FAITHFULNESS_PROMPT_PATH.read_text()
        self._relevance_prompt = self.RELEVANCE_PROMPT_PATH.read_text()
        self._prompt_version = hashlib.sha256(
            (self._faithfulness_prompt + self._relevance_prompt).encode()
        ).hexdigest()[:12]

    async def score(
        self,
        question: str,
        context_chunks: list[str],
        answer: str,
    ) -> JudgeResult:
        """Score a (question, context, answer) triple for faithfulness and relevance."""
        import asyncio

        context_text = "\n\n---\n\n".join(context_chunks)
        user_message = (
            f"Question: {question}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Answer: {answer}"
        )

        system_prompt = f"{self._faithfulness_prompt}\n\n{self._relevance_prompt}"

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=self.model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "submit_score"},
            ),
        )

        tool_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_block is None:
            logger.error("LLM judge did not return tool_use block")
            return JudgeResult(
                faithfulness=0.0,
                relevance=0.0,
                faithfulness_reasoning="Error: no structured output",
                relevance_reasoning="Error: no structured output",
                prompt_version=self._prompt_version,
                model=self.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        data = tool_block.input
        return JudgeResult(
            faithfulness=float(data["faithfulness_score"]),
            relevance=float(data["relevance_score"]),
            faithfulness_reasoning=data["faithfulness_reasoning"],
            relevance_reasoning=data["relevance_reasoning"],
            prompt_version=self._prompt_version,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
