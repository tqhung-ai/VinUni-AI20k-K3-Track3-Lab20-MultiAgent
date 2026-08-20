from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.observability.tracing import TraceManager


def test_local_trace_span_records_timing_and_outputs() -> None:
    tracer = TraceManager(Settings(LANGSMITH_TRACING=False))

    with tracer.span("worker", inputs={"query": "test"}) as span:
        span["outputs"] = {"answer": "done"}

    assert span["provider"] == "local"
    assert isinstance(span["duration_seconds"], float)
    assert span["duration_seconds"] >= 0
    assert span["outputs"] == {"answer": "done"}
