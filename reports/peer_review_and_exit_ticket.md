# Peer Review and Exit Ticket

- Review date: 2026-08-20
- Rubric: `docs/peer_review_rubric.md`
- Review type: self-review using the peer-review rubric; the score is provisional until confirmed by another group.

## Rubric Review

### 1. Role clarity

Strength: The Supervisor only routes; the Researcher searches and records sources/research notes; the Analyst compares evidence and assesses source reliability; the Writer produces the cited final answer. The worker responsibilities are distinct and the graph makes their handoffs explicit.

Risk / failure mode: Researcher and Analyst can still overlap when the Researcher prompt interprets evidence while summarizing it.

One concrete improvement: Tighten the Researcher output schema to factual excerpts and source metadata, leaving comparison and judgment exclusively to the Analyst.

Score: 2/2

### 2. State design

Strength: `ResearchState` preserves the request, sources, research notes, analysis notes, final answer, route history, iteration count, per-agent results, errors and local trace events. This is sufficient to inspect each handoff without reconstructing context from the final answer.

Risk / failure mode: Notes are free-form strings, so downstream agents depend on prompt conventions and malformed sections cannot be validated structurally.

One concrete improvement: Replace free-form research and analysis notes with typed records for claims, evidence IDs, confidence and contradictions.

Score: 2/2

### 3. Failure guard

Strength: The workflow enforces `max_iterations` and a LangGraph recursion limit; provider calls use configured timeouts; OpenAI requests retry twice; agents validate prerequisites, non-empty LLM output and writer citation IDs. Errors are retained in state.

Risk / failure mode: A worker exception currently stops the run; there is no per-node retry policy or graceful partial-answer fallback after search/model failure.

One concrete improvement: Add bounded retries per worker and route repeated failures to a fallback writer that clearly labels missing or unverified evidence.

Score: 2/2

### 4. Benchmark

Strength: Three identical queries were run through baseline and multi-agent paths. The report compares wall-clock latency, tokens, estimated cost, automated quality proxy, citation coverage and failure rate at aggregate and per-query levels. The observed averages were 8,87 seconds and 6,39 quality for baseline versus 23,90 seconds and 8,06 quality for multi-agent; failure rate was 0% for both.

Risk / failure mode: Three queries are a small sample, and the quality score is an automated proxy rather than independent human assessment.

One concrete improvement: Expand the dataset by task type and difficulty, run repeated trials, and add blinded human scoring with inter-rater agreement.

Score: 2/2

### 5. Trace explanation

Strength: LangSmith exposes the root workflow plus Supervisor, Researcher, Analyst and Writer spans, including inputs, outputs, duration and routing. The traces explain the normal sequence `researcher -> analyst -> writer -> done`, while the report identifies latency/cost overhead, unsupported baseline claims and weak-source propagation as key failure modes. Example verified trace: [Open LangSmith trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e24-b4c4-73f2-881e-486db56e8cb4?poll=true).

Risk / failure mode: Trace visibility depends on LangSmith credentials and retention; local fallback spans are not yet exported into a standalone visualization.

One concrete improvement: Export local spans as JSON/OTLP artifacts in CI so failed runs remain inspectable without provider access.

Score: 2/2

## Provisional Result

| Criterion | Score |
|---|---:|
| Role clarity | 2/2 |
| State design | 2/2 |
| Failure guard | 2/2 |
| Benchmark | 2/2 |
| Trace explanation | 2/2 |
| **Total** | **10/10** |

The implementation satisfies all five rubric categories. The most important follow-up is independent peer validation: a reviewer should challenge the automated quality score, inspect whether cited claims are actually supported, and confirm the provisional 10/10 score.

## Exit Ticket

### When should multi-agent be used?

Use it when a task benefits from genuinely different roles or tools—for example, web research, source assessment and cited synthesis—and when intermediate state, traceability and auditability are valuable. It is especially justified when the expected quality improvement outweighs extra orchestration. In this lab, multi-agent increased the quality proxy from 6,39 to 8,06 and citation coverage from 0% to 39%.

### When should multi-agent not be used?

Avoid it for simple, narrow, deterministic or single-step tasks; when one model already has enough context; or when low latency and cost dominate. Here, multi-agent was about 2,7 times slower and its estimated total cost was about 5 times higher. Coordination also creates more failure surfaces, and valid citation formatting still cannot guarantee source quality or factual correctness.
