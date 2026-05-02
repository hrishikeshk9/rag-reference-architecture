# ADR 0003 — Evaluation Strategy

**Status:** Accepted  
**Date:** 2024-11

---

## Context

A RAG system without a systematic eval loop is a liability. Changes to chunking parameters, embedding models, reranker versions, or prompt templates all affect quality. Without measurement, teams find out about regressions from users.

We need to decide how to measure quality, at what frequency, and what gates block deployment. This decision shapes how fast the team can ship changes and how much trust we can place in the system in production.

The core tension: rigorous eval is expensive (human annotation, LLM judge calls) and slow. Fast eval is cheap but may miss important failure modes. We need to calibrate precision vs velocity.

---

## Options Considered

**Option 1: Human annotation as primary signal**

Hire or assign annotators to score RAG outputs. Build a golden set of query/answer pairs manually labeled for relevance, faithfulness, and completeness.

*Strengths:* Highest quality signal. Captures nuance that automated metrics miss. Gold standard for research benchmarks.

*Weaknesses:* Expensive and slow. Building a meaningful golden set of 500 examples takes weeks and significant budget. Keeping it current as the document corpus evolves is a recurring cost. Does not scale to CI — you cannot block a PR waiting for human review. Annotator disagreement is a known problem on subjective dimensions like "helpfulness."

We considered this as the primary signal and rejected it for the CI use case. Appropriate as a periodic audit (quarterly), not a gate.

**Option 2: Reference-free automated metrics (RAGAS-style)**

Compute faithfulness, answer relevance, and context precision without ground-truth answers, using an LLM to score each dimension independently.

*Strengths:* No golden set required. Can evaluate on live traffic. Fast and cheap per query.

*Weaknesses:* The LLM judge is doing the heavy lifting, and its quality determines the signal quality. These metrics are noisy on individual examples. Studies show RAGAS-style metrics have moderate correlation with human judgments (~0.6-0.7 Spearman) — usable for trends, not for hard gates. "Answer relevance" and "faithfulness" from the same judge that generates answers can have circular bias.

**Option 3: Golden set regression with LLM-as-judge**

Maintain a versioned golden set of (query, relevant_chunks, reference_answer) triples. Run offline eval against this set. Use an LLM as judge to score faithfulness and relevance against the reference. Gate CI on metric thresholds. Separately sample production traffic for online eval.

*Strengths:* Deterministic on a fixed golden set — same inputs always produce comparable outputs. LLM judge is applied to a clear comparison task (does the generated answer match the reference?) which is a higher quality signal than reference-free scoring. CI integration is straightforward. Golden set can be bootstrapped from real user queries + a one-time human review pass, then maintained incrementally.

*Weaknesses:* Golden set maintenance is an ongoing cost. Queries that are off-distribution from the golden set are not caught. Judge model selection and prompt versioning require care.

---

## Decision

Use Option 3 (golden set regression + LLM judge) as the primary eval mechanism, with Option 2 (reference-free metrics) as a secondary signal on production traffic.

**Offline eval (gates CI):**
- Golden set of 300 query/chunk/answer triples. Bootstrapped from 3 months of real user questions, human-reviewed once to establish quality bar.
- Metrics: Recall@5, MRR (retrieval), Faithfulness, Relevance (generation)
- CI gate thresholds: Recall@5 must stay within 3% of baseline; Faithfulness within 5%
- Golden set is versioned in `tests/golden/` and updated quarterly or on major corpus changes
- Judge model: `claude-3-5-sonnet-20241022`. Prompts in `src/eval/prompts/`. Prompt versions are pinned by hash.

**Online eval (informs weekly reports, no auto-gate):**
- 2% traffic sampling. Reference-free metrics via LLM judge.
- Results surface in a weekly Slack report + Grafana dashboard.
- Does not block deployment but triggers human review if a 7-day rolling metric drops >10%.

**Why this judge model:**
Evaluated GPT-4o, claude-3-5-sonnet, and Gemini 1.5 Pro as judges by comparing their scores to a human-labeled set of 50 examples. Claude 3.5 Sonnet showed highest agreement (0.81 Spearman on faithfulness, 0.74 on relevance). GPT-4o was close (0.78, 0.72) but more expensive per call. Gemini 1.5 Pro scored lower on faithfulness agreement (0.69). See evaluation results in `tests/judge_calibration/`.

---

## Consequences

**Positive:**
- PRs that degrade retrieval or generation quality are caught before merge
- The golden set creates institutional knowledge about what "good" looks like for this corpus
- Separating offline (gated) from online (informational) eval allows fast deployment cycles without losing production quality visibility

**Negative:**
- Golden set must be maintained. If it drifts from the live corpus, CI passes but production quality silently degrades.
- Judge prompt changes can invalidate historical metric comparisons. Prompt versions must be pinned.
- 2% sampling may undersample rare query types where quality is poor.

**Mitigations:**
- Quarterly golden set review is a scheduled engineering task, not ad-hoc
- Judge prompt changes require a calibration run comparing old/new scores on the 50-example human-labeled set before merging
- Online eval logs query type distribution; anomalies in query type proportions trigger alert

---

## Revisit Triggers

- If the team grows a dedicated eval function, move to human annotation as a quarterly audit layer on top of automated CI
- If judge model costs become prohibitive at scale, explore smaller fine-tuned judge models
- If the golden set grows stale (corpus changes significantly), reset with a new bootstrapping pass
