from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow


class FakeResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        state.sources = [SourceDocument(title="GraphRAG paper", snippet="Evidence")]
        state.research_notes = "Verified research notes"
        return state


class FakeAnalyst(BaseAgent):
    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        state.analysis_notes = "Structured analysis"
        return state


class FakeWriter(BaseAgent):
    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        state.final_answer = "Final answer with citations"
        return state


class StalledResearcher(BaseAgent):
    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        return state


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Research GraphRAG systems"))


def test_workflow_runs_all_routes_in_order() -> None:
    workflow = MultiAgentWorkflow(
        settings=Settings(MAX_ITERATIONS=6),
        researcher=FakeResearcher(),
        analyst=FakeAnalyst(),
        writer=FakeWriter(),
    )

    result = workflow.run(_state())

    assert result.final_answer == "Final answer with citations"
    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert [event["payload"]["next"] for event in result.trace] == [
        "researcher",
        "analyst",
        "writer",
        "done",
    ]


def test_workflow_stops_when_worker_makes_no_progress() -> None:
    workflow = MultiAgentWorkflow(
        settings=Settings(MAX_ITERATIONS=2),
        researcher=StalledResearcher(),
        analyst=FakeAnalyst(),
        writer=FakeWriter(),
    )

    result = workflow.run(_state())

    assert result.final_answer is None
    assert result.route_history == ["researcher", "researcher", "done"]
    assert result.errors == ["Stopped after reaching max_iterations=2."]
