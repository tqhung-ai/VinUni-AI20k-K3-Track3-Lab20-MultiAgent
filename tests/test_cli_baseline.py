import pytest
from typer.testing import CliRunner

from multi_agent_research_lab import cli
from multi_agent_research_lab.services.llm_client import LLMResponse


class FakeLLMClient:
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        assert system_prompt
        assert "Explain multi-agent systems" in user_prompt
        return LLMResponse(content="A real baseline answer.", input_tokens=10, output_tokens=5)


def test_baseline_calls_llm_and_prints_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)

    result = CliRunner().invoke(
        cli.app,
        ["baseline", "--query", "Explain multi-agent systems"],
    )

    assert result.exit_code == 0
    assert "A real baseline answer." in result.stdout
    assert "Baseline Metrics" in result.stdout
    assert "15" in result.stdout
