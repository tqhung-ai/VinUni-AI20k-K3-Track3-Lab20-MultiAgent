from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_through_required_stages() -> None:
    supervisor = SupervisorAgent(Settings(MAX_ITERATIONS=6))
    state = _state()

    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [SourceDocument(title="Source", snippet="Evidence")]
    state.research_notes = "Research notes"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "Analysis notes"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "Final answer"
    supervisor.run(state)
    assert state.route_history == ["researcher", "analyst", "writer", "done"]
    assert state.trace[-1]["payload"]["reason"] == "final_answer_available"


def test_supervisor_stops_at_max_iterations() -> None:
    supervisor = SupervisorAgent(Settings(MAX_ITERATIONS=1))
    state = _state()

    supervisor.run(state)
    supervisor.run(state)

    assert state.route_history == ["researcher", "done"]
    assert state.errors == ["Stopped after reaching max_iterations=1."]
