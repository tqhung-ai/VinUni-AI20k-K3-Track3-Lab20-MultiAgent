# Single-Agent vs Multi-Agent Benchmark Report

- Generated: 2026-08-20T07:54:59+00:00
- Model: `gpt-4o-mini`
- Queries: 3

Quality is an automated 0–10 proxy based on completeness, length, topical coverage, citation coverage, source availability, and execution errors. Citation coverage is the fraction of substantive answer sentences containing a `[S#]` citation.

## Aggregate Summary

| System | Runs | Avg latency (s) | Total est. cost (USD) | Avg tokens | Avg quality | Avg citation coverage | Failure rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 3 | 8.87 | 0.001064 | 640 | 6.39 | 0% | 0% |
| multi-agent | 3 | 23.90 | 0.005341 | 6602 | 8.06 | 39% | 0% |

## Per-Query Results

| System | Query | Latency (s) | Tokens | Est. cost (USD) | Quality | Citation coverage | Failed | Trace |
|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline | Research GraphRAG state-of-the-art and write a 500-word summary | 12.33 | 767 | 0.000429 | 6.33 | 0% | no | [Open trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e29-0bf3-7a60-b477-804be77c858a?poll=true) |
| multi-agent | Research GraphRAG state-of-the-art and write a 500-word summary | 24.10 | 6968 | 0.001920 | 7.28 | 31% | no | [Open trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e29-3b6c-7630-8ee3-5c58cbfe5b19?poll=true) |
| baseline | Compare single-agent and multi-agent workflows for customer support | 5.02 | 531 | 0.000290 | 6.37 | 0% | no | [Open trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e29-9993-7522-a0e3-44b1917768d3?poll=true) |
| multi-agent | Compare single-agent and multi-agent workflows for customer support | 24.26 | 6311 | 0.001712 | 8.39 | 36% | no | [Open trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e29-ad35-70b0-a6df-9e8e88b41f9e?poll=true) |
| baseline | Summarize production guardrails for LLM agents | 9.25 | 623 | 0.000345 | 6.46 | 0% | no | [Open trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e2a-0bfd-7522-a10f-e286d5063bd2?poll=true) |
| multi-agent | Summarize production guardrails for LLM agents | 23.32 | 6528 | 0.001708 | 8.50 | 50% | no | [Open trace](https://smith.langchain.com/o/ef2b1012-3989-4294-ba29-6f969d31c8aa/projects/p/bf13ceab-6673-4845-8d67-8bc348fa0c11/r/01a01e2a-3023-7c50-a680-0b6210d35619?poll=true) |

## Trace Evidence

[Open a public LangSmith multi-agent trace](https://smith.langchain.com/public/014f09d1-aa75-4166-8081-faa2bf55316c/r). The trace contains the end-to-end workflow and its Supervisor, Researcher, Analyst, and Writer steps.

## Failure Mode Analysis

No execution failed in this benchmark sample. The main multi-agent failure mode is operational overhead: average latency was 2.7× baseline and estimated cost was 5.0× baseline. The baseline's average citation coverage was 0%, versus 39% for multi-agent; unsupported or stale claims therefore remain the baseline's dominant quality risk. Multi-agent can still fail when search returns weak sources or when a worker propagates a poorly supported claim, so valid citation IDs do not by themselves guarantee factual correctness. The mitigation is to prioritize authoritative domains, make the Analyst reject unsupported claims, and add bounded retries for transient worker failures. If evidence remains weak, the Writer should explicitly label uncertainty instead of presenting the claim as verified.

## Run Notes

- **baseline — Research GraphRAG state-of-the-art and write a 500-word summary:** Completed successfully.
- **multi-agent — Research GraphRAG state-of-the-art and write a 500-word summary:** Completed successfully.
- **baseline — Compare single-agent and multi-agent workflows for customer support:** Completed successfully.
- **multi-agent — Compare single-agent and multi-agent workflows for customer support:** Completed successfully.
- **baseline — Summarize production guardrails for LLM agents:** Completed successfully.
- **multi-agent — Summarize production guardrails for LLM agents:** Completed successfully.
