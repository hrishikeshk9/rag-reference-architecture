# ADR 0005 — Citation Enforcement Strategy

**Status:** Accepted  
**Date:** 2024-11

---

## Context

Enterprise document Q&A requires that every factual claim in a generated response be traceable to a specific source chunk. Without this, users cannot verify answers, and the system cannot be audited. In regulated industries, "the LLM said so" is not a defensible answer for compliance decisions.

The question is how to enforce citations: through prompt engineering alone, through a post-generation verification step, or through a structured generation approach that makes ungrounded responses structurally impossible.

---

## Options Considered

**Option 1: Prompt instruction only**

Instruct the LLM in the system prompt to cite sources for every claim. Include the source chunk IDs in the context. Ask for responses in a format like "[claim] [Source: chunk_id]".

*Strengths:* Zero implementation overhead. Works with any LLM. Simple.

*Weaknesses:* LLMs do not consistently follow citation instructions, especially for longer responses. The model may cite a chunk that does not actually contain the claim it's referencing (hallucinated citation). The model may omit citations for some claims and include them for others inconsistently. We tested this on GPT-4o and found ~23% of responses had at least one claim with a missing or incorrect citation in early testing.

**Option 2: Post-generation citation verification**

Generate the response normally, then run a second pass that checks each claimed citation against the source chunk. Block or revise responses where a claim cannot be grounded.

*Strengths:* Catches bad citations after the fact. Can use the same LLM or a cheaper model for the verification pass.

*Weaknesses:* Adds a full second LLM call (or at least a verification call) to every response. Verification pass itself can fail — if it uses an LLM, it inherits LLM reliability issues. Increases latency significantly. What do you do on failure? Generating a replacement response doubles the LLM cost again.

**Option 3: Structured generation with grounding constraint**

Use structured output (JSON schema) to force the LLM to produce a list of (claim, source_chunk_id) pairs. After generation, verify each claim against its cited chunk using semantic similarity. Block responses where similarity is below threshold. Return the structured output to the client, which renders citations inline.

*Strengths:* Citation is a first-class structural element, not an afterthought in the prompt. Every claim has exactly one source. Grounding verification is a cheap similarity check (no second LLM call). Client-side rendering is clean because the data structure is predictable.

*Weaknesses:* Requires LLM support for structured output / function calling (available in all major providers). Response format is constrained — long-form prose is harder to produce naturally in JSON. Some information that would normally appear in fluent prose is awkward in a list of (claim, source) pairs.

---

## Decision

Use structured generation with grounding constraint (Option 3), with prompt engineering as a secondary fallback for non-structured providers.

**Implementation:**
- Generate response using `response_format={"type": "json_schema"}` with schema:
  ```json
  {
    "answer": "string",
    "claims": [
      {
        "text": "string",
        "source_chunk_id": "string",
        "confidence": "high | medium | low"
      }
    ]
  }
  ```
- After generation, for each claim, compute cosine similarity between claim embedding and source chunk embedding.
- Threshold: similarity >0.65 = grounded. Below threshold = flag as ungrounded.
- Response with >20% ungrounded claims is blocked; the user receives a "cannot confirm answer" response with the source documents listed for self-service lookup.
- Response with some ungrounded claims but below 20% is returned with ungrounded claims visually marked.

**Why 20% threshold:** Setting it at 0% rejected too many valid responses where the LLM correctly synthesized across multiple chunks but the individual claim-to-chunk similarity was low due to synthesis (the claim is a valid inference, not a direct quote). 20% allows synthesis while blocking predominantly ungrounded responses. Calibrated against 100 human-reviewed responses.

---

## Consequences

**Positive:**
- Every response has a machine-verifiable citation trail
- Grounding check is fast (~10ms, no second LLM call)
- Users can inspect citations in the UI; "show source" is a first-class feature
- Audit log captures grounding scores per claim for compliance review

**Negative:**
- Response format is less fluent than unconstrained prose for some query types
- Structured output requires provider support (GPT-4o, Claude, Gemini all support it; some open-source models do not)
- 20% threshold means some responses with ungrounded claims are shown to users with visual indicators rather than being fully blocked

**Mitigations:**
- For open-source/self-hosted LLMs without structured output support, fall back to Option 1 (prompt instruction) with a post-hoc regex parser to extract citations; grounding verification still applies
- The "ungrounded claim" visual indicator is prominent in the UI; users are not misled about response confidence

---

## Revisit Triggers

- If structured output quality degrades with a new model version (sometimes JSON schema following regresses between model versions), re-run calibration and adjust
- If user feedback shows the structured format feels unnatural, explore a hybrid: generate fluent prose, then run citation extraction as a post-process
- If the 20% threshold generates compliance concerns, lower to 10% or 0% for specific document classification levels via a per-tenant policy
