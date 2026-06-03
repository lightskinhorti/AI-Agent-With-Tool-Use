# Architecture Guide — RAG-Agent

## Overview

RAG-Agent is an autonomous research agent that receives a complex task and decides — on its own — which tools to use, in what order, and when the task is sufficiently complete.

Unlike a standard RAG pipeline (retrieve → generate), this agent **reasons in a loop**: it plans, executes tools, reviews results, and can replan if needed. It also pauses for human approval before executing sensitive operations like code execution.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG-Agent System                         │
│                                                                 │
│  ┌───────────┐         ┌──────────────────────────────────────┐│
│  │  FastAPI   │────────▶│        LangGraph StateGraph          ││
│  │  REST API  │         │                                      ││
│  │            │         │  ┌─────────┐     ┌──────────────┐   ││
│  │ /run       │         │  │ Planner │────▶│   Executor   │   ││
│  │ /status    │         │  └─────────┘     └──────┬───────┘   ││
│  │ /approve   │         │       ▲                  │           ││
│  │ /reject    │         │       │           ┌──────▼───────┐   ││
│  │ /trace     │         │       └───────────│   Reviewer   │   ││
│  │ /result    │         │                   └──────┬───────┘   ││
│  └───────────┘         │                          │           ││
│                         │  ┌───────────┐    ┌──────▼───────┐   ││
│  ┌───────────┐         │  │ HITL Gate │    │  Finalizer   │   ││
│  │ Streamlit │         │  │ (pause)   │    │  (synthesize)│   ││
│  │    UI     │         │  └───────────┘    └──────────────┘   ││
│  │           │         │                                      ││
│  │ Chat      │         │  ┌──────────────────────────────┐    ││
│  │ Traces    │         │  │   SqliteSaver Checkpointer   │    ││
│  │ HITL      │         │  │   (persistent state to disk) │    ││
│  │ Metrics   │         │  └──────────────────────────────┘    ││
│  └───────────┘         └──────────────────────────────────────┘│
│                                                                 │
│  TOOLS:                                                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────────┐              │
│  │ rag_search │ │ web_search │ │ document_fetch │              │
│  └────────────┘ └────────────┘ └────────────────┘              │
│  ┌──────────────┐ ┌───────────────┐ ┌──────────────┐          │
│  │code_executor │ │ report_writer │ │ memory_store │          │
│  │  (HITL req.) │ └───────────────┘ └──────────────┘          │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Flow — Step by Step

### 1. Planning Phase
The **Planner** node receives the user's task and the descriptions of all 6 available tools. It calls Claude to generate a JSON execution plan:

```json
[
  {"action": "rag_search", "args": {"query": "...", "top_k": 5}, "reasoning": "Search knowledge base first"},
  {"action": "web_search", "args": {"query": "..."}, "reasoning": "Supplement with web data"},
  {"action": "report_writer", "args": {"title": "...", ...}, "reasoning": "Compile findings"}
]
```

### 2. Execution Phase
The **Executor** iterates through the plan steps. For each step:
- Looks up the tool in `TOOL_REGISTRY`
- Checks if the tool requires human approval (`requires_human_approval`)
- If HITL required → sets `requires_human=True`, graph pauses
- If not → executes the tool, records the result

### 3. Review Phase
After execution, the **Reviewer** evaluates all accumulated results:
- **"complete"** → results are sufficient, proceed to finalize
- **"replan"** → results are insufficient, go back to Planner with feedback
- **"continue"** → more plan steps remain, continue executing

The Reviewer also generates an explicit **reflection** — a 2-3 sentence analysis of what was learned. This creates an auditable reasoning trail.

### 4. HITL Gate
When the agent needs to execute code (`code_executor`), the graph **actually suspends**:
1. LangGraph's `interrupt_before=["hitl_gate"]` pauses execution
2. Full state is checkpointed to SQLite via `SqliteSaver`
3. The API returns `status: "waiting_human"`
4. A human reviews the pending code and approves/rejects via API
5. `graph.aupdate_state()` injects the decision into the checkpoint
6. `graph.ainvoke(None)` resumes from the exact saved state

This is not a `input("Are you sure?")` — the process can die between steps 2 and 5, and the agent will still resume correctly.

### 5. Finalization
The **Finalizer** takes all tool results and reflections, calls Claude to synthesize a comprehensive answer, and computes final observability metrics.

---

## Tool Architecture

Every tool inherits from `BaseTool` and follows the same pattern:

```
BaseTool (abstract)
├── Pydantic input schema (validation)
├── _execute() — core logic (async)
├── run() — validates, measures latency, logs with structlog
├── _run_with_retry() — tenacity retry (3 attempts, exponential backoff)
└── to_langchain_tool() — converts to LangChain StructuredTool
```

### Tool Details

| Tool | Input | What it does | Key detail |
|------|-------|-------------|------------|
| `rag_search` | query, top_k | Hybrid search: ChromaDB dense + BM25 sparse + RRF fusion | Reuses ChromaDB from RAG project |
| `web_search` | query, max_results | DuckDuckGo search, no API key needed | Wrapped in `asyncio.to_thread` |
| `document_fetch` | url | Downloads + parses HTML (BeautifulSoup) and PDF (pypdf) | Truncates to 4000 chars |
| `code_executor` | code, description | RestrictedPython sandbox with safe builtins | **HITL required** |
| `report_writer` | title, sections, content | Generates Markdown with TOC | Can save to file |
| `memory_store` | action, key, value | SQLite key-value store | Persists across sessions |

---

## State Management

The `AgentState` is a TypedDict with annotated fields that control how LangGraph merges state updates:

| Field | Type | Reducer | Purpose |
|-------|------|---------|---------|
| `messages` | list[BaseMessage] | `add_messages` | Conversation history (deduplicates by ID) |
| `tool_calls` | list[ToolCallRecord] | `operator.add` | Accumulates across steps |
| `tool_results` | list[ToolResultRecord] | `operator.add` | Accumulates across steps |
| `reflections` | list[str] | `operator.add` | Accumulates across steps |
| `plan` | list[dict] | replace | Current execution plan |
| `current_step` | int | replace | Progress through plan |
| `requires_human` | bool | replace | HITL flag |
| `error_count` | int | replace | Circuit breaker (max 3) |
| `metadata` | dict | replace | Token usage, latencies |

---

## Observability

Every agent run tracks:
- **Token usage**: input + output tokens per LLM call, accumulated in metadata
- **Tool latency**: milliseconds per tool execution
- **Tool success rate**: ratio of successful tool calls
- **Execution trace**: full history of graph state transitions (retrievable via `/agent/trace`)
- **Reflections**: explicit reasoning from the Reviewer node

---

## Error Handling Strategy

1. **Tool-level**: tenacity retry with exponential backoff (3 attempts) for transient errors
2. **Step-level**: failed tools record `ToolResult(success=False)`, execution continues
3. **Graph-level**: circuit breaker at `error_count >= 3` routes to Finalizer
4. **Review-level**: Reviewer can replan with different approach (max 3 iterations)
5. **API-level**: async tasks fail gracefully, errors logged with structlog

---

## Key Design Decisions

### Why SqliteSaver instead of MemorySaver?
MemorySaver stores state in RAM — it's lost when the process restarts. SqliteSaver persists every graph transition to disk. The agent can crash mid-execution, restart, and resume from the exact step where it stopped. This is non-negotiable for production.

### Why interrupt_before instead of a confirmation dialog?
`interrupt_before` is LangGraph's native mechanism for HITL. It suspends the graph execution, persists state, and allows external code to inject decisions. The graph can be resumed hours or days later. A confirmation dialog is synchronous and blocking.

### Why hybrid search (ChromaDB + BM25)?
Dense embeddings (ChromaDB) capture semantic similarity but miss exact keyword matches. BM25 captures exact terms but misses semantic meaning. Reciprocal Rank Fusion (RRF) combines both rankings for better retrieval quality.

### Why RestrictedPython instead of subprocess?
RestrictedPython runs code in the same process with restricted builtins — no filesystem access, no network, no imports beyond the allowlist. It's simpler and faster than spinning up a subprocess or container for each execution. For production, Docker-based sandboxing would be the next step.
