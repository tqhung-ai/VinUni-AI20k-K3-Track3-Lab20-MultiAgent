"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    load_benchmark_queries,
    run_benchmark_suite,
)
from multi_agent_research_lab.evaluation.report import write_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import TraceManager
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.utils.timer import elapsed_timer

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _format_metric(value: int | float | None, *, decimals: int | None = None) -> str:
    if value is None:
        return "n/a"
    if decimals is not None:
        return f"{value:.{decimals}f}"
    return str(value)


def _print_baseline_metrics(response: LLMResponse, latency_seconds: float) -> None:
    table = Table(title="Baseline Metrics")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Input tokens", justify="right")
    table.add_column("Output tokens", justify="right")
    table.add_column("Total tokens", justify="right")
    table.add_row(
        _format_metric(latency_seconds, decimals=3),
        _format_metric(response.input_tokens),
        _format_metric(response.output_tokens),
        _format_metric(response.total_tokens),
    )
    console.print(table)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline through the configured LLM."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    system_prompt = (
        "You are a single-agent research assistant. Produce an accurate, self-contained answer "
        "for the requested audience. Clearly distinguish established facts from uncertainty, "
        "and do not claim to have used tools or sources that were not provided."
    )
    user_prompt = f"Research question: {request.query}\nAudience: {request.audience}"

    try:
        client = LLMClient()
        with elapsed_timer() as elapsed:
            response = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        latency_seconds = elapsed()
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=2) from exc

    state.final_answer = response.content
    state.add_trace_event(
        "baseline_completed",
        {
            "model": get_settings().openai_model,
            "latency_seconds": latency_seconds,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
        },
    )
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    _print_baseline_metrics(response, latency_seconds)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Upload nested agent spans to LangSmith"),
    ] = False,
) -> None:
    """Run the multi-agent research workflow."""

    _init()
    settings = get_settings()
    state = ResearchState(request=_parse_query(query))
    tracer = TraceManager(settings, enabled=trace or settings.langsmith_tracing)
    workflow = MultiAgentWorkflow(settings=settings, tracer=tracer)
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Workflow Error", style="red"))
        raise typer.Exit(code=2) from exc
    finally:
        tracer.flush()
    console.print(result.model_dump_json(indent=2))


@app.command("benchmark")
def benchmark_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="YAML file containing benchmark.queries"),
    ] = Path("configs/lab_default.yaml"),
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Markdown report output path"),
    ] = Path("reports/benchmark_report.md"),
    trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace", help="Upload benchmark spans to LangSmith"),
    ] = True,
) -> None:
    """Benchmark baseline and multi-agent systems on the same query suite."""

    _init()
    settings = get_settings()
    tracer = TraceManager(settings, enabled=trace)
    if trace and not tracer.enabled:
        console.print(
            Panel.fit(
                "LangSmith tracing is unavailable; benchmark will continue with local traces.",
                title="Tracing Warning",
                style="yellow",
            )
        )

    try:
        queries = load_benchmark_queries(config)
        console.print(f"Running {len(queries)} queries through baseline and multi-agent...")
        metrics = run_benchmark_suite(queries, settings=settings, tracer=tracer)
        report_path = write_markdown_report(output, metrics, model=settings.openai_model)
    except (OSError, ValueError, AgentExecutionError) as exc:
        console.print(Panel.fit(str(exc), title="Benchmark Error", style="red"))
        raise typer.Exit(code=2) from exc
    finally:
        tracer.flush()

    summary = Table(title="Benchmark Summary")
    summary.add_column("System")
    summary.add_column("Query", max_width=45)
    summary.add_column("Latency", justify="right")
    summary.add_column("Tokens", justify="right")
    summary.add_column("Quality", justify="right")
    summary.add_column("Citations", justify="right")
    summary.add_column("Failed", justify="center")
    for item in metrics:
        summary.add_row(
            item.run_name,
            item.query,
            f"{item.latency_seconds:.2f}s",
            str(item.total_tokens),
            f"{item.quality_score or 0:.2f}",
            f"{item.citation_coverage or 0:.0%}",
            "yes" if item.failure_rate else "no",
        )
    console.print(summary)
    console.print(f"Report written to: {report_path}")


if __name__ == "__main__":
    app()
