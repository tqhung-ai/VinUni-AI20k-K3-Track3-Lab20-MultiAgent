"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from typing import Protocol, cast

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by every LLM provider implementation."""

    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def total_tokens(self) -> int | None:
        """Return total token usage when both counters are available."""

        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens


class _ResponseUsage(Protocol):
    input_tokens: int
    output_tokens: int


class _ModelResponse(Protocol):
    output_text: str | None
    usage: _ResponseUsage | None


class _ResponsesResource(Protocol):
    def create(self, *, model: str, instructions: str, input: str) -> _ModelResponse: ...


class _OpenAIClient(Protocol):
    @property
    def responses(self) -> _ResponsesResource: ...


class CompletionClient(Protocol):
    """Interface consumed by agents that need a text completion."""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


class LLMClient:
    """OpenAI-backed implementation behind a provider-agnostic interface."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: _OpenAIClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or self._build_client()

    def _build_client(self) -> _OpenAIClient:
        if not self.settings.openai_api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not configured. Add it to .env before running the baseline."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise AgentExecutionError(
                'The OpenAI SDK is not installed. Run: pip install -e ".[llm]"'
            ) from exc

        return cast(
            _OpenAIClient,
            OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=float(self.settings.timeout_seconds),
                max_retries=2,
            ),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Call the Responses API and normalize text and usage metadata."""

        try:
            response = self._client.responses.create(
                model=self.settings.openai_model,
                instructions=system_prompt,
                input=user_prompt,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            status = f"HTTP {status_code}, " if isinstance(status_code, int) else ""
            raise AgentExecutionError(
                f"OpenAI request failed ({status}{type(exc).__name__}). "
                "Check the API key, model access, and network connection."
            ) from exc

        content = (response.output_text or "").strip()
        if not content:
            raise AgentExecutionError("OpenAI returned an empty text response.")

        usage = response.usage
        return LLMResponse(
            content=content,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
        )
