from pathlib import Path

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    calculate_citation_coverage,
    load_benchmark_queries,
    run_benchmark,
)


def test_citation_coverage_counts_substantive_sentences() -> None:
    answer = (
        "Graph retrieval connects related evidence across documents for complex questions [S1]. "
        "This second substantive sentence deliberately has no citation attached to its claim."
    )

    assert calculate_citation_coverage(answer) == pytest.approx(0.5)


def test_run_benchmark_measures_tokens_cost_and_quality() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.final_answer = (
            "GraphRAG combines graph retrieval with generated answers for complex research "
            "questions and preserves evidence links [S1]."
        )
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=state.final_answer,
                metadata={"input_tokens": 100, "output_tokens": 50},
            )
        )
        return state

    _, metrics = run_benchmark(
        "multi-agent",
        "Research GraphRAG systems",
        runner,
        Settings(
            OPENAI_INPUT_COST_PER_MILLION=1,
            OPENAI_OUTPUT_COST_PER_MILLION=2,
        ),
    )

    assert metrics.total_tokens == 150
    assert metrics.estimated_cost_usd == pytest.approx(0.0002)
    assert metrics.citation_coverage == 1.0
    assert metrics.quality_score and metrics.quality_score > 0
    assert metrics.failure_rate == 0.0


def test_load_benchmark_queries(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark:\n  queries:\n    - First valid query\n", encoding="utf-8")

    assert load_benchmark_queries(config) == ["First valid query"]
