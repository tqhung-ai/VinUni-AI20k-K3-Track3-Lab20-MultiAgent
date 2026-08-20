"""LangGraph orchestration for the research workflow."""

from typing import Any, Protocol, cast

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import TraceManager


class _RunnableGraph(Protocol):
    def invoke(
        self,
        input: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        supervisor: BaseAgent | None = None,
        researcher: BaseAgent | None = None,
        analyst: BaseAgent | None = None,
        writer: BaseAgent | None = None,
        tracer: TraceManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(self.settings)
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.tracer = tracer or TraceManager(self.settings)
        self._compiled: _RunnableGraph | None = None

    @staticmethod
    def _state_update(state: ResearchState) -> dict[str, Any]:
        return state.model_dump(mode="python")

    def _run_supervisor(self, state: ResearchState) -> dict[str, Any]:
        with self.tracer.span(
            "agent.supervisor",
            inputs={
                "iteration": state.iteration,
                "has_sources": bool(state.sources),
                "has_analysis": bool(state.analysis_notes),
                "has_final_answer": bool(state.final_answer),
            },
            attributes={"agent": "supervisor"},
            tags=["multi-agent", "router"],
        ) as span:
            result = self.supervisor.run(state)
            span["outputs"] = {
                "next_route": result.route_history[-1],
                "iteration": result.iteration,
            }
            return self._state_update(result)

    def _run_researcher(self, state: ResearchState) -> dict[str, Any]:
        with self.tracer.span(
            "agent.researcher",
            inputs={"query": state.request.query, "max_sources": state.request.max_sources},
            attributes={"agent": "researcher"},
            tags=["multi-agent", "worker"],
        ) as span:
            result = self.researcher.run(state)
            span["outputs"] = {
                "sources": [source.model_dump() for source in result.sources],
                "research_notes": result.research_notes,
            }
            return self._state_update(result)

    def _run_analyst(self, state: ResearchState) -> dict[str, Any]:
        with self.tracer.span(
            "agent.analyst",
            inputs={
                "research_notes": state.research_notes,
                "source_count": len(state.sources),
            },
            attributes={"agent": "analyst"},
            tags=["multi-agent", "worker"],
        ) as span:
            result = self.analyst.run(state)
            span["outputs"] = {"analysis_notes": result.analysis_notes}
            return self._state_update(result)

    def _run_writer(self, state: ResearchState) -> dict[str, Any]:
        with self.tracer.span(
            "agent.writer",
            inputs={
                "research_notes": state.research_notes,
                "analysis_notes": state.analysis_notes,
                "source_count": len(state.sources),
            },
            attributes={"agent": "writer"},
            tags=["multi-agent", "worker"],
        ) as span:
            result = self.writer.run(state)
            span["outputs"] = {"final_answer": result.final_answer}
            return self._state_update(result)

    @staticmethod
    def _route_after_supervisor(state: ResearchState) -> str:
        if not state.route_history:
            raise AgentExecutionError("Supervisor did not record a route.")

        route = state.route_history[-1]
        allowed_routes = {"researcher", "analyst", "writer", "done"}
        if route not in allowed_routes:
            raise AgentExecutionError(f"Supervisor returned an unsupported route: {route}")
        return route

    def build(self) -> _RunnableGraph:
        """Build and compile the worker loop with conditional supervisor routing."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise AgentExecutionError(
                'LangGraph is not installed. Run: pip install -e ".[llm]"'
            ) from exc

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self._run_supervisor)
        graph.add_node("researcher", self._run_researcher)
        graph.add_node("analyst", self._run_analyst)
        graph.add_node("writer", self._run_writer)

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            self._route_after_supervisor,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "done": END,
            },
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        self._compiled = cast(_RunnableGraph, graph.compile())
        return self._compiled

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the compiled graph and validate its result as shared state."""

        with self.tracer.span(
            "multi_agent_workflow",
            inputs={"query": state.request.query},
            attributes={"max_iterations": self.settings.max_iterations},
            tags=["multi-agent", "benchmarkable"],
            link=True,
        ) as span:
            graph = self._compiled or self.build()
            recursion_limit = (self.settings.max_iterations * 2) + 3
            result = graph.invoke(state, config={"recursion_limit": recursion_limit})
            final_state = ResearchState.model_validate(result)
            span["outputs"] = {
                "final_answer": final_state.final_answer,
                "route_history": final_state.route_history,
                "errors": final_state.errors,
            }

        if span.get("url"):
            final_state.add_trace_event(
                "langsmith_trace",
                {"url": span["url"], "run_id": span.get("run_id")},
            )
        return final_state
