from typing import Any

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.search_client import SearchClient


class FakeTransport:
    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}

    def __call__(
        self,
        *,
        endpoint: str,
        api_key: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        assert endpoint == "https://api.tavily.com/search"
        assert api_key == "test-tavily-key"
        assert timeout == 12.0
        self.payload = payload
        return {
            "results": [
                {
                    "title": "GraphRAG paper",
                    "url": "https://example.com/paper",
                    "content": "Evidence from the paper.",
                    "score": 0.97,
                },
                {"title": "Empty result", "content": ""},
            ]
        }


def test_search_client_normalizes_tavily_results() -> None:
    transport = FakeTransport()
    client = SearchClient(
        settings=Settings(TAVILY_API_KEY="test-tavily-key", TIMEOUT_SECONDS=12),
        transport=transport,
    )

    sources = client.search("GraphRAG", max_results=3)

    assert len(sources) == 1
    assert sources[0].title == "GraphRAG paper"
    assert sources[0].snippet == "Evidence from the paper."
    assert sources[0].metadata == {"provider": "tavily", "rank": 1, "score": 0.97}
    assert transport.payload["query"] == "GraphRAG"
    assert transport.payload["max_results"] == 3


def test_search_client_requires_tavily_key() -> None:
    client = SearchClient(settings=Settings(TAVILY_API_KEY=""), transport=FakeTransport())

    with pytest.raises(AgentExecutionError, match="TAVILY_API_KEY"):
        client.search("GraphRAG")
