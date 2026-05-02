"""
Prompt construction for the RAG generation step.

Separates prompt assembly from LLM calls so prompts are independently testable
and version-controlled. Prompt version is hashed to track which version produced
a given LLM response in eval logs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.retrieve.reranker import RankedResult


SYSTEM_PROMPT = """\
You are a precise technical assistant. Answer questions using ONLY the provided context chunks.

Rules:
1. Every factual claim must be grounded in a context chunk. Cite it as [chunk_id].
2. If the context does not contain enough information to fully answer, say so explicitly.
3. Do not speculate or add information from general knowledge.
4. Quote exact figures, version numbers, and commands from the context — do not paraphrase them.
5. If chunks contradict each other, acknowledge the contradiction and cite both.
"""

CONTEXT_TEMPLATE = "[{chunk_id}] {text}"

USER_TEMPLATE = """\
Context:
{context_block}

Question: {question}

Answer (cite each claim with [chunk_id]):"""


@dataclass
class GenerationPrompt:
    system: str
    user: str
    prompt_version: str     # SHA-256 of SYSTEM_PROMPT for eval tracking
    chunk_ids_used: list[str]


def build_prompt(question: str, chunks: list[RankedResult]) -> GenerationPrompt:
    """Assemble system + user messages from ranked retrieval results."""
    context_block = "\n\n".join(
        CONTEXT_TEMPLATE.format(chunk_id=c.chunk_id, text=c.text.strip())
        for c in chunks
    )
    user_msg = USER_TEMPLATE.format(
        context_block=context_block,
        question=question,
    )
    version = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]

    return GenerationPrompt(
        system=SYSTEM_PROMPT,
        user=user_msg,
        prompt_version=version,
        chunk_ids_used=[c.chunk_id for c in chunks],
    )
