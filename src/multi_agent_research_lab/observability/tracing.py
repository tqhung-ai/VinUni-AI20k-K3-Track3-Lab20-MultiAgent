"""Optional LangSmith tracing with an always-available local span fallback."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Literal

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


class TraceManager:
    """Create nested spans locally and mirror them to LangSmith when enabled."""

    def __init__(self, settings: Settings | None = None, *, enabled: bool | None = None) -> None:
        self.settings = settings or get_settings()
        requested = self.settings.langsmith_tracing if enabled is None else enabled
        self._client: Any | None = None
        self.enabled = bool(requested and self.settings.langsmith_api_key)
        if not self.enabled:
            return

        try:
            from langsmith import Client

            self._client = Client(
                api_key=self.settings.langsmith_api_key,
                timeout_ms=self.settings.timeout_seconds * 1_000,
            )
        except Exception as exc:
            self.enabled = False
            logger.warning("LangSmith tracing disabled: %s", type(exc).__name__)

    @contextmanager
    def span(
        self,
        name: str,
        *,
        run_type: RunType = "chain",
        inputs: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        link: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield a mutable span record and finalize timing/provider metadata."""

        started = perf_counter()
        record: dict[str, Any] = {
            "name": name,
            "attributes": attributes or {},
            "duration_seconds": None,
            "provider": "langsmith" if self.enabled else "local",
        }

        if not self.enabled or self._client is None:
            try:
                yield record
            finally:
                record["duration_seconds"] = perf_counter() - started
            return

        from langsmith.run_helpers import trace, tracing_context

        run = None
        try:
            with (
                tracing_context(
                    enabled=True,
                    client=self._client,
                    project_name=self.settings.langsmith_project,
                ),
                trace(
                    name,
                    run_type=run_type,
                    inputs=inputs or {},
                    project_name=self.settings.langsmith_project,
                    client=self._client,
                    metadata=attributes or {},
                    tags=tags or [],
                ) as run,
            ):
                record["run_id"] = str(run.id)
                try:
                    yield record
                except BaseException:
                    raise
                else:
                    outputs = record.get("outputs")
                    run.end(outputs=outputs if isinstance(outputs, dict) else {"result": outputs})
        finally:
            record["duration_seconds"] = perf_counter() - started

        if run is not None and link:
            try:
                self._client.flush(timeout=10)
                record["url"] = self._client.get_run_url(
                    run=run,
                    project_name=self.settings.langsmith_project,
                )
            except Exception as exc:
                logger.warning("Could not resolve LangSmith run URL: %s", type(exc).__name__)

    def flush(self, timeout: float = 10.0) -> None:
        """Flush queued LangSmith writes before a short-lived CLI process exits."""

        if self._client is None:
            return
        try:
            self._client.flush(timeout=timeout)
        except Exception as exc:
            logger.warning("Could not flush LangSmith traces: %s", type(exc).__name__)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Backward-compatible convenience wrapper around :class:`TraceManager`."""

    with TraceManager().span(name, attributes=attributes) as span:
        yield span
