"""Reproducible single-agent versus multi-agent benchmark utilities."""

import re
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.tracing import TraceManager
from multi_agent_research_lab.services.llm_client import CompletionClient, LLMClient

Runner = Callable[[str], ResearchState]
_CITATION_PATTERN = re.compile(r"\[S\d+\]")
_WORD_PATTERN = re.compile(r"[\w-]+", flags=re.UNICODE)


def run_single_agent(
    query: str,
    *,
    settings: Settings | None = None,
    llm_client: CompletionClient | None = None,
    tracer: TraceManager | None = None,
) -> ResearchState:
    """Run the same no-search single-agent baseline used for comparison."""

    runtime_settings = settings or get_settings()
    completion_client = llm_client or LLMClient(runtime_settings)
    trace_manager = tracer or TraceManager(runtime_settings)
    state = ResearchState(request=ResearchQuery(query=query))

    with trace_manager.span(
        "single_agent_baseline",
        inputs={"query": query},
        attributes={"model": runtime_settings.openai_model},
        tags=["single-agent", "benchmarkable"],
        link=True,
    ) as span:
        response = completion_client.complete(
            system_prompt=(
                "You are a single-agent research assistant. Produce an accurate, self-contained "
                "answer for technical learners. Clearly distinguish established facts from "
                "uncertainty, and do not claim to use sources or tools that were not provided."
            ),
            user_prompt=f"Research question: {query}",
        )
        state.final_answer = response.content
        usage = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
        state.add_trace_event(
            "baseline_completed", {"model": runtime_settings.openai_model, **usage}
        )
        span["outputs"] = {"final_answer": response.content, **usage}

    if span.get("url"):
        state.add_trace_event(
            "langsmith_trace",
            {"url": span["url"], "run_id": span.get("run_id")},
        )
    return state


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
    settings: Settings | None = None,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure one run and return a state even when execution fails."""

    runtime_settings = settings or get_settings()
    started = perf_counter()
    try:
        state = runner(query)
        failed = bool(state.errors or not state.final_answer)
        notes = "Completed successfully." if not failed else "; ".join(state.errors)
    except Exception as exc:
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(f"{type(exc).__name__}: {exc}")
        failed = True
        notes = state.errors[0]
    latency = perf_counter() - started

    input_tokens, output_tokens = _token_usage(state)
    total_tokens = input_tokens + output_tokens
    citation_coverage = calculate_citation_coverage(state.final_answer or "")
    quality = calculate_quality_score(state, citation_coverage)
    estimated_cost = (
        input_tokens * runtime_settings.openai_input_cost_per_million
        + output_tokens * runtime_settings.openai_output_cost_per_million
    ) / 1_000_000

    metrics = BenchmarkMetrics(
        run_name=run_name,
        query=query,
        latency_seconds=latency,
        estimated_cost_usd=estimated_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        quality_score=quality,
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failed else 0.0,
        trace_url=_trace_url(state),
        notes=notes,
    )
    return state, metrics


def run_benchmark_suite(
    queries: list[str],
    *,
    settings: Settings | None = None,
    tracer: TraceManager | None = None,
) -> list[BenchmarkMetrics]:
    """Run every query through both systems using shared provider clients."""

    if not queries:
        raise ValueError("At least one benchmark query is required.")

    runtime_settings = settings or get_settings()
    trace_manager = tracer or TraceManager(runtime_settings)
    baseline_client = LLMClient(runtime_settings)
    workflow = MultiAgentWorkflow(settings=runtime_settings, tracer=trace_manager)

    def baseline_runner(query: str) -> ResearchState:
        return run_single_agent(
            query,
            settings=runtime_settings,
            llm_client=baseline_client,
            tracer=trace_manager,
        )

    def multi_runner(query: str) -> ResearchState:
        return workflow.run(ResearchState(request=ResearchQuery(query=query)))

    metrics: list[BenchmarkMetrics] = []
    for query in queries:
        metrics.append(run_benchmark("baseline", query, baseline_runner, runtime_settings)[1])
        metrics.append(run_benchmark("multi-agent", query, multi_runner, runtime_settings)[1])

    trace_manager.flush()
    return metrics


def load_benchmark_queries(path: Path) -> list[str]:
    """Load and validate the benchmark query list from the lab YAML config."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark config must contain a mapping.")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("Benchmark config is missing the benchmark section.")
    raw_queries = benchmark.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("Benchmark config is missing benchmark.queries.")
    queries = [item.strip() for item in raw_queries if isinstance(item, str) and item.strip()]
    if not queries:
        raise ValueError("Benchmark query list cannot be empty.")
    return queries


def calculate_citation_coverage(answer: str) -> float:
    """Estimate the share of substantive answer sentences containing [S#]."""

    body = answer.split("## Sources", maxsplit=1)[0]
    segments = re.split(r"(?<=[.!?])\s+|\n+", body)
    claims = [
        segment.strip()
        for segment in segments
        if len(_WORD_PATTERN.findall(segment)) >= 8 and not segment.lstrip().startswith("#")
    ]
    if not claims:
        return 0.0
    cited_claims = sum(bool(_CITATION_PATTERN.search(claim)) for claim in claims)
    return cited_claims / len(claims)


def calculate_quality_score(state: ResearchState, citation_coverage: float) -> float:
    """Return a transparent 0-10 automated quality proxy."""

    answer = state.final_answer or ""
    if not answer.strip():
        return 0.0

    words = _WORD_PATTERN.findall(answer)
    score = 2.0
    score += min(2.0, len(words) / 200)

    query_terms = {
        token.lower() for token in _WORD_PATTERN.findall(state.request.query) if len(token) >= 5
    }
    answer_terms = {token.lower() for token in words}
    topical_coverage = len(query_terms & answer_terms) / len(query_terms) if query_terms else 1.0
    score += 2.0 * topical_coverage
    score += 2.0 * citation_coverage
    score += 1.0 if state.sources else 0.0
    score += 1.0 if not state.errors else 0.0
    return round(min(10.0, score), 2)


def _token_usage(state: ResearchState) -> tuple[int, int]:
    if state.agent_results:
        input_tokens = sum(
            _metadata_int(item.metadata, "input_tokens") for item in state.agent_results
        )
        output_tokens = sum(
            _metadata_int(item.metadata, "output_tokens") for item in state.agent_results
        )
        return input_tokens, output_tokens

    for event in reversed(state.trace):
        if event.get("name") != "baseline_completed":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            return _metadata_int(payload, "input_tokens"), _metadata_int(payload, "output_tokens")
    return 0, 0


def _metadata_int(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _trace_url(state: ResearchState) -> str | None:
    for event in reversed(state.trace):
        if event.get("name") != "langsmith_trace":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("url"), str):
            return cast(str, payload["url"])
    return None
