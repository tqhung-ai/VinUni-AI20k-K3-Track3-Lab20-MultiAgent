"""Writer agent that produces a cited final answer."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import CompletionClient, LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"
    citation_pattern = re.compile(r"\[S(\d+)\]")

    def __init__(self, llm_client: CompletionClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Synthesize the final answer and validate inline source citations."""

        if not state.sources:
            raise AgentExecutionError("Writer requires at least one source.")
        if not state.research_notes or not state.research_notes.strip():
            raise AgentExecutionError("Writer requires non-empty research_notes.")
        if not state.analysis_notes or not state.analysis_notes.strip():
            raise AgentExecutionError("Writer requires non-empty analysis_notes.")

        source_context = self._format_source_context(state.sources)
        response = self.llm_client.complete(
            system_prompt=(
                "You are the Writer in a multi-agent research system. Write a clear, accurate "
                "answer for the requested audience using only the supplied research and analysis. "
                "Every externally verifiable factual claim must include one or more inline "
                "citations such as [S1]. Use only source IDs present in the source catalog. "
                "Do not add a separate sources section; the application will append it."
            ),
            user_prompt=(
                f"Question: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Source catalog:\n{source_context}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis notes:\n{state.analysis_notes}"
            ),
        )

        cited_ids = self._validate_citations(response.content, state.sources)
        sources_section = self._render_sources(state.sources, cited_ids)
        state.final_answer = f"{response.content.rstrip()}\n\n## Sources\n{sources_section}"
        metadata = {
            "citation_count": len(self.citation_pattern.findall(response.content)),
            "cited_source_count": len(cited_ids),
            "cited_source_ids": sorted(cited_ids),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
        state.agent_results.append(
            AgentResult(agent=AgentName.WRITER, content=state.final_answer, metadata=metadata)
        )
        state.add_trace_event("agent_completed", {"agent": "writer", **metadata})
        return state

    @classmethod
    def _validate_citations(
        cls,
        content: str,
        sources: list[SourceDocument],
    ) -> set[str]:
        cited_ids = {f"S{number}" for number in cls.citation_pattern.findall(content)}
        if not cited_ids:
            raise AgentExecutionError("Writer produced an answer without inline [S#] citations.")

        valid_ids = {
            str(source.metadata.get("source_id", f"S{index}"))
            for index, source in enumerate(sources, start=1)
        }
        invalid_ids = cited_ids - valid_ids
        if invalid_ids:
            invalid = ", ".join(sorted(invalid_ids))
            raise AgentExecutionError(f"Writer cited unknown source IDs: {invalid}.")
        return cited_ids

    @staticmethod
    def _format_source_context(sources: list[SourceDocument]) -> str:
        return "\n\n".join(
            f"[{source.metadata.get('source_id', f'S{index}')}] {source.title}\n"
            f"URL: {source.url or 'N/A'}\nExcerpt: {source.snippet}"
            for index, source in enumerate(sources, start=1)
        )

    @staticmethod
    def _render_sources(sources: list[SourceDocument], cited_ids: set[str]) -> str:
        entries = []
        for index, source in enumerate(sources, start=1):
            source_id = str(source.metadata.get("source_id", f"S{index}"))
            if source_id not in cited_ids:
                continue
            if source.url:
                entries.append(f"- [{source_id}] [{source.title}]({source.url})")
            else:
                entries.append(f"- [{source_id}] {source.title}")
        return "\n".join(entries)
