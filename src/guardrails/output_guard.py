"""
Output guardrails: validate generated answers before returning to the user.

Checks:
1. Hallucination risk: if grounding_score < threshold, add disclaimer
2. PII leakage: scan answer for PII patterns that shouldn't appear in generated text
3. Refusal detection: if the model refused to answer, return a structured refusal
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.generate.llm import GenerationResult

GROUNDING_DISCLAIMER = (
    "[Note: This answer may contain claims not fully supported by the source documents. "
    "Please verify before acting on this information.]"
)

REFUSAL_PATTERNS = [
    r"i('m|\s+am)\s+(unable|not\s+able)\s+to",
    r"i\s+cannot\s+(help|answer|provide)",
    r"that('s|\s+is)\s+(outside|beyond)\s+(my|the)",
    r"i\s+don'?t\s+have\s+(access|information)",
]

PII_OUTPUT_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",                          # SSN
    r"\b4[0-9]{12}(?:[0-9]{3})?\b",                    # Visa card
    r"\b(?:25[0-5]|2[0-4][0-9]|[01]?\d\d?)(?:\.\d{1,3}){3}\b",  # IP address (internal)
]

GROUNDING_SCORE_THRESHOLD = 0.70


@dataclass
class OutputCheckResult:
    safe: bool
    answer: str                         # Possibly modified (disclaimer added)
    grounding_ok: bool
    pii_detected: bool
    is_refusal: bool
    warnings: list[str]


class OutputGuard:
    """
    Post-generation safety checks on the LLM answer.

    Does not block answers outright (except PII leakage) — instead adds
    disclaimers for low-grounding answers and flags refusals for monitoring.
    PII in the output is always blocked: it likely came from PII in source docs,
    which should be redacted at ingestion time.
    """

    def check(
        self, result: GenerationResult, grounding_score: float
    ) -> OutputCheckResult:
        warnings = []
        answer = result.answer

        # Refusal detection
        answer_lower = answer.lower()
        is_refusal = any(re.search(p, answer_lower) for p in REFUSAL_PATTERNS)
        if is_refusal:
            warnings.append("Model returned a refusal")

        # PII detection in output
        pii_detected = any(re.search(p, answer) for p in PII_OUTPUT_PATTERNS)
        if pii_detected:
            warnings.append("PII pattern detected in generated answer — blocked")
            return OutputCheckResult(
                safe=False,
                answer="Unable to display answer: potential PII detected in source documents.",
                grounding_ok=grounding_score >= GROUNDING_SCORE_THRESHOLD,
                pii_detected=True,
                is_refusal=is_refusal,
                warnings=warnings,
            )

        # Grounding check
        grounding_ok = grounding_score >= GROUNDING_SCORE_THRESHOLD
        if not grounding_ok:
            warnings.append(f"Low grounding score: {grounding_score:.2f} < {GROUNDING_SCORE_THRESHOLD}")
            answer = f"{answer}\n\n{GROUNDING_DISCLAIMER}"

        return OutputCheckResult(
            safe=True,
            answer=answer,
            grounding_ok=grounding_ok,
            pii_detected=False,
            is_refusal=is_refusal,
            warnings=warnings,
        )
