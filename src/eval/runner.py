"""
Offline eval runner. Loads a golden set, runs the retrieval+generation pipeline,
computes metrics, and outputs a report with a CI-ready exit code.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .judge import Judge
from .metrics import compute_mrr, compute_recall_at_k, faithfulness_score


@dataclass
class GoldenExample:
    query: str
    expected_chunk_ids: list[str]
    reference_answer: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    query: str
    retrieved_chunk_ids: list[str]
    generated_answer: str
    recall_at_5: float
    mrr: float
    faithfulness: float
    relevance: float
    ungrounded_claim_ratio: float


@dataclass
class EvalReport:
    results: list[EvalResult]
    mean_recall_at_5: float
    mean_mrr: float
    mean_faithfulness: float
    mean_relevance: float
    baseline: Optional[dict] = None
    regressions: list[str] = field(default_factory=list)

    def to_pr_comment(self) -> str:
        lines = ["## RAG Eval Results", ""]
        lines.append(f"| Metric | Score | Baseline | Delta |")
        lines.append(f"|---|---|---|---|")

        def row(name, val, key):
            baseline_val = self.baseline.get(key) if self.baseline else None
            delta = f"{val - baseline_val:+.3f}" if baseline_val else "—"
            flag = " ⚠️" if self.regressions and name in self.regressions else ""
            return f"| {name}{flag} | {val:.3f} | {baseline_val or '—'} | {delta} |"

        lines.append(row("Recall@5", self.mean_recall_at_5, "recall_at_5"))
        lines.append(row("MRR", self.mean_mrr, "mrr"))
        lines.append(row("Faithfulness", self.mean_faithfulness, "faithfulness"))
        lines.append(row("Relevance", self.mean_relevance, "relevance"))

        if self.regressions:
            lines.append("")
            lines.append(f"**CI gate failed.** Regressions detected: {', '.join(self.regressions)}")

        return "\n".join(lines)


class EvalRunner:
    """
    Loads a golden set JSONL and evaluates the full retrieval + generation pipeline.

    Golden set format (one JSON object per line):
      {"query": "...", "expected_chunk_ids": ["..."], "reference_answer": "..."}
    """

    RECALL_REGRESSION_THRESHOLD = 0.03
    FAITHFULNESS_REGRESSION_THRESHOLD = 0.05

    def __init__(self, pipeline, judge: Judge, baseline_path: Optional[Path] = None):
        self.pipeline = pipeline
        self.judge = judge
        self.baseline = json.loads(baseline_path.read_text()) if baseline_path else None

    def load_golden_set(self, path: Path) -> list[GoldenExample]:
        examples = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(GoldenExample(
                query=obj["query"],
                expected_chunk_ids=obj["expected_chunk_ids"],
                reference_answer=obj["reference_answer"],
                metadata=obj.get("metadata", {}),
            ))
        return examples

    async def run(self, golden_set_path: Path) -> EvalReport:
        examples = self.load_golden_set(golden_set_path)
        results = []

        for ex in examples:
            retrieved, answer, grounding = await self.pipeline.run(ex.query)
            retrieved_ids = [c.chunk_id for c in retrieved]

            faithfulness = await self.judge.score_faithfulness(
                query=ex.query,
                answer=answer,
                reference=ex.reference_answer,
            )
            relevance = await self.judge.score_relevance(
                query=ex.query,
                answer=answer,
            )

            results.append(EvalResult(
                query=ex.query,
                retrieved_chunk_ids=retrieved_ids,
                generated_answer=answer,
                recall_at_5=compute_recall_at_k(retrieved_ids, ex.expected_chunk_ids, k=5),
                mrr=compute_mrr(retrieved_ids, ex.expected_chunk_ids),
                faithfulness=faithfulness,
                relevance=relevance,
                ungrounded_claim_ratio=grounding.ungrounded_ratio,
            ))

        report = self._build_report(results)
        return report

    def _build_report(self, results: list[EvalResult]) -> EvalReport:
        n = len(results)
        mean_recall = sum(r.recall_at_5 for r in results) / n
        mean_mrr = sum(r.mrr for r in results) / n
        mean_faith = sum(r.faithfulness for r in results) / n
        mean_rel = sum(r.relevance for r in results) / n

        regressions = []
        if self.baseline:
            if mean_recall < self.baseline.get("recall_at_5", 0) - self.RECALL_REGRESSION_THRESHOLD:
                regressions.append("Recall@5")
            if mean_faith < self.baseline.get("faithfulness", 0) - self.FAITHFULNESS_REGRESSION_THRESHOLD:
                regressions.append("Faithfulness")

        return EvalReport(
            results=results,
            mean_recall_at_5=mean_recall,
            mean_mrr=mean_mrr,
            mean_faithfulness=mean_faith,
            mean_relevance=mean_rel,
            baseline=self.baseline,
            regressions=regressions,
        )

    def exit_code(self, report: EvalReport) -> int:
        return 1 if report.regressions else 0


async def main(golden_set: str, baseline: Optional[str] = None):
    from src.pipeline import build_pipeline
    pipeline = build_pipeline()
    judge = Judge()
    runner = EvalRunner(pipeline, judge, Path(baseline) if baseline else None)
    report = await runner.run(Path(golden_set))

    print(report.to_pr_comment())
    sys.exit(runner.exit_code(report))
