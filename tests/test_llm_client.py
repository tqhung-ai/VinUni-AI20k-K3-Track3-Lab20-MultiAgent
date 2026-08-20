from dataclasses import dataclass

import pytest

from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.services.llm_client import LLMClient


@dataclass
class FakeUsage:
    input_tokens: int = 12
    output_tokens: int = 8


@dataclass
class FakeResponse:
    output_text: str = "  A structured answer.  "
    usage: FakeUsage | None = None


class FakeResponsesResource:
    def __init__(self, response: FakeResponse | None = None) -> None:
        self.response = response or FakeResponse(usage=FakeUsage())
        self.request: dict[str, str] = {}

    def create(self, *, model: str, instructions: str, input: str) -> FakeResponse:
        self.request = {"model": model, "instructions": instructions, "input": input}
        return self.response


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponsesResource | None = None) -> None:
        self.responses = responses or FakeResponsesResource()


class FailingResponsesResource(FakeResponsesResource):
    def create(self, *, model: str, instructions: str, input: str) -> FakeResponse:
        raise RuntimeError("sensitive-provider-detail")


def test_complete_returns_normalized_structured_response() -> None:
    responses = FakeResponsesResource()
    client = LLMClient(
        settings=Settings(OPENAI_API_KEY="test-key", OPENAI_MODEL="test-model"),
        client=FakeOpenAIClient(responses),
    )

    result = client.complete("system instructions", "user question")

    assert result.content == "A structured answer."
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.total_tokens == 20
    assert responses.request == {
        "model": "test-model",
        "instructions": "system instructions",
        "input": "user question",
    }


def test_complete_rejects_empty_output() -> None:
    responses = FakeResponsesResource(FakeResponse(output_text="   ", usage=None))
    client = LLMClient(
        settings=Settings(OPENAI_API_KEY="test-key"),
        client=FakeOpenAIClient(responses),
    )

    with pytest.raises(AgentExecutionError, match="empty text response"):
        client.complete("system instructions", "user question")


def test_complete_does_not_expose_provider_error_details() -> None:
    client = LLMClient(
        settings=Settings(OPENAI_API_KEY="test-key"),
        client=FakeOpenAIClient(FailingResponsesResource()),
    )

    with pytest.raises(AgentExecutionError) as raised:
        client.complete("system instructions", "user question")

    assert "RuntimeError" in str(raised.value)
    assert "sensitive-provider-detail" not in str(raised.value)
