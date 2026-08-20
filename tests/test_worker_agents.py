from collections.abc import Sequence

import pytest

from multi_agent_research_lab.agents import AnalystAgent, ResearcherAgent, WriterAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeSearchClient:
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        assert query == "Research GraphRAG systems"
        assert max_results == 5
        return [
            SourceDocument(
                title="GraphRAG paper",
                url="https://example.com/paper",
                snippet="Graph communities support global summarization.",
                metadata={"provider": "test", "score": 0.95},
            ),
            SourceDocument(
                title="Independent evaluation",
                url="https://example.com/evaluation",
                snippet="Evaluation reports quality and cost trade-offs.",
                metadata={"provider": "test", "score": 0.85},
            ),
        ]


class QueuedLLMClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.prompts.append((system_prompt, user_prompt))
        return LLMResponse(
            content=self.responses.pop(0),
            input_tokens=20,
            output_tokens=10,
        )


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Research GraphRAG systems"))


def test_real_worker_pipeline_populates_debuggable_state() -> None:
    llm = QueuedLLMClient(
        [
            "Graph communities enable global summaries [S1]. Evaluations identify trade-offs [S2].",
            "The primary source supports the mechanism [S1], while evaluation evidence qualifies "
            "the quality and cost claims [S2].",
            "GraphRAG improves global sensemaking through community summaries [S1], but its value "
            "must be weighed against measured quality and cost trade-offs [S2].",
        ]
    )
    workflow = MultiAgentWorkflow(
        settings=Settings(MAX_ITERATIONS=6),
        researcher=ResearcherAgent(FakeSearchClient(), llm),
        analyst=AnalystAgent(llm),
        writer=WriterAgent(llm),
    )

    result = workflow.run(_state())

    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert len(result.sources) == 2
    assert [source.metadata["source_id"] for source in result.sources] == ["S1", "S2"]
    assert result.research_notes and "[S1]" in result.research_notes
    assert result.analysis_notes and "quality and cost" in result.analysis_notes
    assert result.final_answer and "[S1]" in result.final_answer
    assert "[GraphRAG paper](https://example.com/paper)" in result.final_answer
    assert [item.agent for item in result.agent_results] == [
        AgentName.RESEARCHER,
        AgentName.ANALYST,
        AgentName.WRITER,
    ]
    assert all(item.metadata["total_tokens"] == 30 for item in result.agent_results)
    assert len([event for event in result.trace if event["name"] == "agent_completed"]) == 3


def test_writer_rejects_unknown_citation() -> None:
    state = _state()
    state.sources = [
        SourceDocument(
            title="Known source",
            url="https://example.com",
            snippet="Known evidence",
            metadata={"source_id": "S1"},
        )
    ]
    state.research_notes = "Known evidence [S1]."
    state.analysis_notes = "The evidence is relevant [S1]."
    writer = WriterAgent(QueuedLLMClient(["Unsupported citation [S9]."]))

    with pytest.raises(AgentExecutionError, match="unknown source IDs: S9"):
        writer.run(state)
