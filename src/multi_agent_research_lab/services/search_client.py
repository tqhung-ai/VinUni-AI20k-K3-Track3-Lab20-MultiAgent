"""Tavily search client used by ResearcherAgent."""

import json
from typing import Any, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchProvider(Protocol):
    """Interface consumed by agents that need external sources."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]: ...


class SearchTransport(Protocol):
    def __call__(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]: ...


class SearchClient:
    """Small Tavily HTTP client with an injectable transport for tests."""

    endpoint = "https://api.tavily.com/search"
    max_snippet_chars = 2_000

    def __init__(
        self,
        settings: Settings | None = None,
        transport: SearchTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._transport = transport or self._post_json

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search Tavily and normalize its results into public source schemas."""

        query = query.strip()
        if not query:
            raise AgentExecutionError("Search query cannot be empty.")
        if not 1 <= max_results <= 20:
            raise AgentExecutionError("max_results must be between 1 and 20.")
        if not self.settings.tavily_api_key:
            raise AgentExecutionError(
                "TAVILY_API_KEY is not configured. Add it to .env or inject a mock SearchProvider."
            )

        payload: dict[str, Any] = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "topic": "general",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "include_usage": True,
        }
        response = self._transport(
            endpoint=self.endpoint,
            api_key=self.settings.tavily_api_key,
            payload=payload,
            timeout=float(self.settings.timeout_seconds),
        )

        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise AgentExecutionError("Tavily returned a response without a results list.")

        sources: list[SourceDocument] = []
        for rank, item in enumerate(raw_results[:max_results], start=1):
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            url = item.get("url")
            score = item.get("score")
            sources.append(
                SourceDocument(
                    title=title.strip() if isinstance(title, str) and title.strip() else "Untitled",
                    url=url if isinstance(url, str) and url else None,
                    snippet=content.strip()[: self.max_snippet_chars],
                    metadata={
                        "provider": "tavily",
                        "rank": rank,
                        "score": score if isinstance(score, int | float) else None,
                    },
                )
            )

        if not sources:
            raise AgentExecutionError("Tavily returned no usable search results.")
        return sources

    @staticmethod
    def _post_json(
        *,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise AgentExecutionError(f"Tavily search failed (HTTP {exc.code}).") from exc
        except (TimeoutError, URLError) as exc:
            raise AgentExecutionError(
                f"Tavily search failed ({type(exc).__name__}). Check the network connection."
            ) from exc
        except json.JSONDecodeError as exc:
            raise AgentExecutionError("Tavily returned malformed JSON data.") from exc

        if not isinstance(decoded, dict):
            raise AgentExecutionError("Tavily returned invalid JSON data.")
        return cast(dict[str, Any], decoded)
