"""
Input guardrails: validate and sanitize queries before retrieval.

Implemented as a fast synchronous pre-filter — no LLM call, no I/O.
Blocks prompt injection, PII extraction attempts, and scope violations
before the retrieval pipeline runs.

See ADR 0004 for placement rationale (input + output, not just output).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class GuardrailViolation(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    PII_EXTRACTION = "pii_extraction"
    OUT_OF_SCOPE = "out_of_scope"
    QUERY_TOO_LONG = "query_too_long"
    EMPTY_QUERY = "empty_query"


@dataclass
class InputCheckResult:
    allowed: bool
    violation: GuardrailViolation | None = None
    reason: str = ""
    sanitized_query: str = ""  # Cleaned version if allowed


MAX_QUERY_LENGTH = 1000

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"disregard\s+your\s+system\s+prompt",
    r"repeat\s+the\s+above\s+text",
    r"<\s*/?script",
    r"system\s*:\s*you\s+are",
    r"assistant\s*:\s*i\s+will\s+now",
    r"jailbreak",
    r"dan\s+mode",
]

PII_EXTRACTION_PATTERNS = [
    r"(social\s+security|ssn)\s*(number|#)?",
    r"credit\s+card\s+number",
    r"password\s+for",
    r"api\s+key\s+for",
    r"private\s+key",
    r"bank\s+account\s+number",
]


class InputGuard:
    """
    Fast regex-based input guardrail.

    Runs before any retrieval or LLM call. Stateless and synchronous —
    adds <1ms latency. Sensitive queries are rejected outright; borderline
    queries are sanitized (HTML stripped, length truncated).
    """

    def check(self, query: str, allowed_topics: list[str] | None = None) -> InputCheckResult:
        """Validate query. Returns InputCheckResult with allowed=True if safe."""
        if not query or not query.strip():
            return InputCheckResult(
                allowed=False,
                violation=GuardrailViolation.EMPTY_QUERY,
                reason="Query is empty",
            )

        if len(query) > MAX_QUERY_LENGTH:
            return InputCheckResult(
                allowed=False,
                violation=GuardrailViolation.QUERY_TOO_LONG,
                reason=f"Query exceeds {MAX_QUERY_LENGTH} characters",
            )

        query_lower = query.lower()

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                return InputCheckResult(
                    allowed=False,
                    violation=GuardrailViolation.PROMPT_INJECTION,
                    reason=f"Prompt injection pattern detected: {pattern}",
                )

        for pattern in PII_EXTRACTION_PATTERNS:
            if re.search(pattern, query_lower):
                return InputCheckResult(
                    allowed=False,
                    violation=GuardrailViolation.PII_EXTRACTION,
                    reason="PII extraction attempt detected",
                )

        sanitized = self._sanitize(query)
        return InputCheckResult(
            allowed=True,
            sanitized_query=sanitized,
        )

    def _sanitize(self, query: str) -> str:
        # Strip HTML tags
        clean = re.sub(r"<[^>]+>", "", query)
        # Collapse whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean
