# ADR 0004 — Guardrails Placement

**Status:** Accepted  
**Date:** 2024-11

---

## Context

Enterprise RAG systems process documents that may contain sensitive information (PII, internal financials, confidential strategy). Users may submit queries that attempt to extract information outside the intended scope, elicit jailbreaks, or probe the system boundaries. The generated output may surface PII from retrieved chunks or produce ungrounded claims.

We need to decide where to place guardrails: before retrieval (input), after generation (output), or both. This is an architectural decision with latency, cost, and coverage trade-offs.

---

## Options Considered

**Option 1: Input guardrails only**

Screen the user's query before the retrieval pipeline runs. Block PII in the query itself, detect jailbreak attempts, enforce topic scope.

*Strengths:* Cheapest. Catches bad inputs before expensive retrieval + generation runs. Simplest to implement.

*Weaknesses:* Does not protect against leakage through generation. A clean query can still produce a response that surfaces PII from retrieved chunks, or generates an ungrounded answer that passes no output filter. Compliance use cases require output protection regardless of input quality.

**Option 2: Output guardrails only**

Screen the generated response. Block PII in the output. Score faithfulness. Detect hallucination signals.

*Strengths:* One check point. Covers all input paths.

*Weaknesses:* Every query incurs the cost of retrieval + LLM generation before the guardrail fires. A high-volume jailbreak attempt or off-topic flood would exhaust LLM budget before being caught. Late failure is more expensive to handle and harder to explain to users ("your query ran but we can't show you the answer").

**Option 3: Input AND output guardrails**

Run input guardrails before retrieval. Run output guardrails after generation. Accept the latency and cost of both.

*Strengths:* Defense in depth. Input guardrails are cheap fast filters that catch the majority of bad inputs at low cost. Output guardrails are the last line of defense against leakage and hallucination. This layered model is what regulated industries (healthcare, finance, legal) expect.

*Weaknesses:* Higher latency (~80ms added for both checks combined). Higher operational complexity — two guardrail systems to maintain and update.

---

## Decision

Use both input and output guardrails (Option 3).

Rationale:
1. **Cost asymmetry.** Input guardrails cost ~5ms and prevent the expensive case (LLM call) from running on bad inputs. The marginal cost of also having output guardrails is ~75ms on top of a 500ms+ generation call — not significant.
2. **Compliance requirement.** The enterprise document corpus contains documents classified at multiple sensitivity levels. The compliance requirement is that PII and confidential financial data must never appear in external-facing responses, regardless of how the query was structured. This is only enforceable at the output layer.
3. **Defense in depth principle.** Input and output guardrails catch different failure classes. Input: off-topic abuse, jailbreaks, injection attacks. Output: retrieval surfacing sensitive data, LLM generating ungrounded claims, PII in cited source text.

**Input guardrails check (in order, fast-fail on first match):**
1. PII detection in query text (regex + NER — catches SSN, email, phone patterns)
2. Jailbreak / prompt injection detection (classifier — fine-tuned on injection attack patterns)
3. Topic scope check (embedding similarity to allowed topic whitelist — catches completely off-domain queries)

**Output guardrails check:**
1. PII scan of full response text (same NER as input)
2. Citation grounding check (every factual claim has a cited source chunk — overlap with citation enforcer)
3. Hallucination signal (LLM-as-judge, sampled 10% of responses in production — not on every call)

---

## Consequences

**Positive:**
- Clean audit trail: every blocked query/response is logged with reason code
- Cheap to protect against abuse at input; comprehensive protection at output
- Input guardrail catches problems before they cost LLM budget

**Negative:**
- Combined latency add is ~80ms (input: ~5ms, output: ~75ms)
- Output PII scan must run against full response text + source chunks, which can be long; async option explored but rejected because it complicates the response path
- Topic scope check has false positive risk — overly tight whitelist rejects legitimate queries

**Mitigations:**
- Topic scope check uses a soft threshold (similarity >0.3 to nearest topic) rather than exact match; flagged cases are reviewed monthly to tune the threshold
- Output guardrail is idempotent — if it fires on a valid response, the user receives a "this content cannot be shown" message with a support path, not a silent failure

---

## Revisit Triggers

- If output PII scan latency exceeds 200ms at scale, move to async post-processing with a response hold pattern
- If hallucination scoring via LLM judge becomes cheap enough to run on every response (rather than 10% sample), upgrade to synchronous full coverage
- If the compliance requirement changes to allow certain document classifications to bypass some guardrails, introduce a policy-based guardrail router rather than one fixed pipeline
