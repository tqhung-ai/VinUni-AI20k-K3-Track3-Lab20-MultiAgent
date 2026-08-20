"""Analyst agent that compares claims and evaluates source reliability."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import CompletionClient, LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: CompletionClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Evaluate research notes and produce a source-aware analysis."""

        if not state.sources or not state.research_notes or not state.research_notes.strip():
            raise AgentExecutionError("Analyst requires sources and non-empty research_notes.")

        source_catalog = "\n".join(
            f"[{source.metadata.get('source_id', f'S{index}')}] {source.title} "
            f"(score={source.metadata.get('score', 'n/a')}, url={source.url or 'N/A'})"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are the Analyst in a multi-agent research system. Compare the supplied "
                "claims, assess source reliability and corroboration, separate strong evidence "
                "from weak or conflicting evidence, and propose a defensible answer outline. "
                "Retain [S#] citations and do not introduce facts absent from the notes."
            ),
            user_prompt=(
                f"Question: {state.request.query}\n\n"
                f"Source catalog:\n{source_catalog}\n\n"
                f"Research notes:\n{state.research_notes}"
            ),
        )
        if not response.content.strip():
            raise AgentExecutionError("Analyst produced empty analysis notes.")

        state.analysis_notes = response.content
        metadata = {
            "source_count": len(state.sources),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
        state.agent_results.append(
            AgentResult(agent=AgentName.ANALYST, content=response.content, metadata=metadata)
        )
        state.add_trace_event("agent_completed", {"agent": "analyst", **metadata})
        return state
