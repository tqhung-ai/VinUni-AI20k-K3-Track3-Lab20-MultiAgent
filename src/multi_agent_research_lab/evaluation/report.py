"""Benchmark report rendering and persistence."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    *,
    model: str = "configured OpenAI model",
    generated_at: datetime | None = None,
) -> str:
    """Render aggregate and per-query metrics with failure-mode analysis."""

    generated = generated_at or datetime.now(UTC)
    lines = [
        "# Single-Agent vs Multi-Agent Benchmark Report",
        "",
        f"Generated: {generated.isoformat(timespec='seconds')}  ",
        f"Model: `{model}`  ",
        f"Queries: {len({item.query for item in metrics})}",
        "",
        "Quality is an automated 0–10 proxy based on completeness, length, topical coverage, "
        "citation coverage, source availability, and execution errors. Citation coverage is the "
        "fraction of substantive answer sentences containing a `[S#]` citation.",
        "",
        "## Aggregate Summary",
        "",
        "| System | Runs | Avg latency (s) | Total est. cost (USD) | Avg tokens | "
        "Avg quality | Avg citation coverage | Failure rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, items in _group_by_system(metrics).items():
        lines.append(
            f"| {run_name} | {len(items)} | {_average(items, 'latency_seconds'):.2f} | "
            f"{sum(item.estimated_cost_usd or 0 for item in items):.6f} | "
            f"{_average(items, 'total_tokens'):.0f} | {_average(items, 'quality_score'):.2f} | "
            f"{_average(items, 'citation_coverage'):.0%} | "
            f"{_average(items, 'failure_rate'):.0%} |"
        )

    lines.extend(
        [
            "",
            "## Per-Query Results",
            "",
            "| System | Query | Latency (s) | Tokens | Est. cost (USD) | Quality | "
            "Citation coverage | Failed | Trace |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in metrics:
        trace_link = f"[Open trace]({item.trace_url})" if item.trace_url else "—"
        lines.append(
            f"| {item.run_name} | {_escape(item.query)} | {item.latency_seconds:.2f} | "
            f"{item.total_tokens} | {(item.estimated_cost_usd or 0):.6f} | "
            f"{(item.quality_score or 0):.2f} | {(item.citation_coverage or 0):.0%} | "
            f"{'yes' if item.failure_rate else 'no'} | {trace_link} |"
        )

    lines.extend(
        [
            "",
            "## Failure Mode Analysis",
            "",
            _failure_mode_analysis(metrics),
            "",
            "## Run Notes",
            "",
        ]
    )
    for item in metrics:
        lines.append(f"- **{item.run_name} — {_escape(item.query)}:** {_escape(item.notes)}")
    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    metrics: list[BenchmarkMetrics],
    *,
    model: str,
) -> Path:
    """Write a rendered report and return its resolved path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown_report(metrics, model=model), encoding="utf-8")
    return path.resolve()


def _group_by_system(metrics: list[BenchmarkMetrics]) -> dict[str, list[BenchmarkMetrics]]:
    grouped: dict[str, list[BenchmarkMetrics]] = defaultdict(list)
    for item in metrics:
        grouped[item.run_name].append(item)
    return dict(grouped)


def _average(items: list[BenchmarkMetrics], field: str) -> float:
    values = [getattr(item, field) for item in items]
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _failure_mode_analysis(metrics: list[BenchmarkMetrics]) -> str:
    grouped = _group_by_system(metrics)
    baseline = grouped.get("baseline", [])
    multi = grouped.get("multi-agent", [])
    failures = [item for item in metrics if item.failure_rate]

    observations = []
    if failures:
        failed_runs = ", ".join(f"{item.run_name} ({item.query})" for item in failures)
        observations.append(f"Observed execution failures: {failed_runs}.")
    else:
        observations.append("No execution failed in this benchmark sample.")

    if baseline and multi:
        baseline_latency = _average(baseline, "latency_seconds")
        multi_latency = _average(multi, "latency_seconds")
        baseline_cost = sum(item.estimated_cost_usd or 0 for item in baseline)
        multi_cost = sum(item.estimated_cost_usd or 0 for item in multi)
        latency_ratio = multi_latency / baseline_latency if baseline_latency else 0
        cost_ratio = multi_cost / baseline_cost if baseline_cost else 0
        observations.append(
            f"The main multi-agent failure mode is operational overhead: average latency was "
            f"{latency_ratio:.1f}× baseline and estimated cost was {cost_ratio:.1f}× baseline."
        )
        observations.append(
            f"The baseline's average citation coverage was "
            f"{_average(baseline, 'citation_coverage'):.0%}, versus "
            f"{_average(multi, 'citation_coverage'):.0%} for multi-agent; unsupported or stale "
            "claims therefore remain the baseline's dominant quality risk. Multi-agent can still "
            "fail when search returns weak sources or when a worker propagates a poorly supported "
            "claim, so valid citation IDs do not by themselves guarantee factual correctness."
        )
    return " ".join(observations)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
