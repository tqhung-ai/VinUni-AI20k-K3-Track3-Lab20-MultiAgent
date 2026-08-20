from datetime import UTC, datetime

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.report import render_markdown_report


def test_report_renders_markdown() -> None:
    metrics = [
        BenchmarkMetrics(
            run_name="baseline",
            query="Research GraphRAG",
            latency_seconds=1.23,
            quality_score=6,
            citation_coverage=0,
            failure_rate=0,
        ),
        BenchmarkMetrics(
            run_name="multi-agent",
            query="Research GraphRAG",
            latency_seconds=4.56,
            quality_score=9,
            citation_coverage=0.8,
            failure_rate=0,
            trace_url="https://smith.langchain.com/example",
        ),
    ]
    report = render_markdown_report(
        metrics,
        model="test-model",
        generated_at=datetime(2026, 8, 20, tzinfo=UTC),
    )

    assert "Benchmark Report" in report
    assert "baseline" in report
    assert "Aggregate Summary" in report
    assert "Failure Mode Analysis" in report
    assert "Open trace" in report
