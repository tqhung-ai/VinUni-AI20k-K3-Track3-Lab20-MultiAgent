"""Deterministic supervisor for the multi-agent workflow."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Choose and record the next worker, or the terminal ``done`` route."""

        route, reason = self._select_route(state)
        state.record_route(route)
        state.add_trace_event(
            "route",
            {
                "next": route,
                "reason": reason,
                "iteration": state.iteration,
            },
        )
        return state

    def _select_route(self, state: ResearchState) -> tuple[str, str]:
        if state.final_answer and state.final_answer.strip():
            return "done", "final_answer_available"

        if state.iteration >= self.settings.max_iterations:
            message = f"Stopped after reaching max_iterations={self.settings.max_iterations}."
            if message not in state.errors:
                state.errors.append(message)
            return "done", "max_iterations_reached"

        has_research = bool(state.sources) and bool(
            state.research_notes and state.research_notes.strip()
        )
        if not has_research:
            return "researcher", "research_missing"

        if not state.analysis_notes or not state.analysis_notes.strip():
            return "analyst", "analysis_missing"

        return "writer", "ready_to_write"
