"""Researcher agent that gathers sources and creates grounded notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import CompletionClient, LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient, SearchProvider


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchProvider | None = None,
        llm_client: CompletionClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate sources and notes grounded exclusively in search excerpts."""

        sources = self.search_client.search(
            state.request.query,
            max_results=state.request.max_sources,
        )
        self._assign_source_ids(sources)
        source_context = self._format_source_context(sources)

        response = self.llm_client.complete(
            system_prompt=(
                "You are the Researcher in a multi-agent system. Use only the supplied source "
                "excerpts. Produce concise research notes, attach [S#] to factual claims, identify "
                "agreements or conflicts, and explicitly flag evidence gaps. Never invent a source."
            ),
            user_prompt=(
                f"Research question: {state.request.query}\n\nSource excerpts:\n{source_context}"
            ),
        )
        if not response.content.strip():
            raise AgentExecutionError("Researcher produced empty research notes.")

        state.sources = sources
        state.research_notes = response.content
        self._record_result(state, response)
        return state

    @staticmethod
    def _assign_source_ids(sources: list[SourceDocument]) -> None:
        for index, source in enumerate(sources, start=1):
            source.metadata["source_id"] = f"S{index}"

    @staticmethod
    def _format_source_context(sources: list[SourceDocument]) -> str:
        blocks = []
        for source in sources:
            source_id = source.metadata["source_id"]
            blocks.append(
                f"[{source_id}] {source.title}\n"
                f"URL: {source.url or 'N/A'}\n"
                f"Excerpt: {source.snippet}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _record_result(state: ResearchState, response: LLMResponse) -> None:
        metadata = {
            "source_count": len(state.sources),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
        state.agent_results.append(
            AgentResult(agent=AgentName.RESEARCHER, content=response.content, metadata=metadata)
        )
        state.add_trace_event("agent_completed", {"agent": "researcher", **metadata})
