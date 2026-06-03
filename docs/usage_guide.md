# Usage Guide — RAG-Agent

## Prerequisites

- Python 3.11+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Docker + Docker Compose (optional, for containerized deployment)

---

## Installation

### Option A: Docker (recommended)

```bash
git clone https://github.com/lightskinhorti/AI-Agent-With-Tool-Use.git
cd AI-Agent-With-Tool-Use
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
docker compose up --build
```

Services:
- **API**: http://localhost:8000 (Swagger docs at /docs)
- **UI**: http://localhost:8501

### Option B: Local

```bash
git clone https://github.com/lightskinhorti/AI-Agent-With-Tool-Use.git
cd AI-Agent-With-Tool-Use
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

Start the API:
```bash
uvicorn api.main:app --reload --port 8000
```

Start the UI (separate terminal):
```bash
streamlit run ui/app.py
```

---

## Configuration

All settings are managed via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `AGENT_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `AGENT_MAX_STEPS` | `15` | Maximum execution steps per run |
| `HITL_TIMEOUT_SECONDS` | `300` | Timeout for human approval |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `SQLITE_DB_PATH` | `./data/agent.db` | Agent checkpoint database |
| `MEMORY_DB_PATH` | `./data/memory.db` | Memory tool database |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Using the CLI

The simplest way to test the agent:

```bash
python -m agent.run "What are the key principles of the EU AI Act?"
```

Output includes the final answer, token usage, tool calls, and success rate.

---

## Using the API

### 1. Start a task

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Research the latest trends in AI agent frameworks and create a summary"}'
```

Response:
```json
{
  "run_id": "a1b2c3d4-...",
  "status": "started",
  "message": "Agent run initiated"
}
```

### 2. Check status

```bash
curl http://localhost:8000/agent/status/a1b2c3d4-...
```

Response:
```json
{
  "run_id": "a1b2c3d4-...",
  "status": "executing",
  "current_step": 2,
  "total_steps": 4,
  "requires_human": false,
  "tool_calls_made": 2
}
```

### 3. Handle HITL (when status = "waiting_human")

If the agent wants to execute code, it pauses and waits for approval:

```json
{
  "status": "waiting_human",
  "requires_human": true,
  "pending_tool_call": {
    "tool_name": "code_executor",
    "args": {"code": "result = sum(range(100))", "description": "Sum numbers"},
    "reasoning": "Need to calculate the total"
  }
}
```

Approve:
```bash
curl -X POST http://localhost:8000/agent/approve/a1b2c3d4-...
```

Reject:
```bash
curl -X POST http://localhost:8000/agent/reject/a1b2c3d4-...
```

### 4. Get results

```bash
curl http://localhost:8000/agent/result/a1b2c3d4-...
```

Response includes: final answer, tool results, reflections, and metrics (tokens, latency, success rate).

### 5. View execution trace

```bash
curl http://localhost:8000/agent/trace/a1b2c3d4-...
```

Returns the full history of graph state transitions — useful for debugging and understanding the agent's reasoning process.

---

## Using the Streamlit UI

1. Open http://localhost:8501
2. Type a task in the chat input
3. Watch the agent work in real-time (status polling)
4. If HITL is triggered, review the pending action and approve/reject
5. View execution trace and metrics in the sidebar

---

## Adding Documents to the Knowledge Base

The `rag_search` tool searches a ChromaDB collection. To add documents:

```python
import chromadb

client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_or_create_collection("knowledge_base")

collection.add(
    documents=["Your document text here...", "Another document..."],
    ids=["doc_001", "doc_002"],
    metadatas=[{"source": "manual"}, {"source": "manual"}]
)
```

The agent will automatically search these documents when the Planner decides `rag_search` is relevant.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Only tool tests
pytest tests/test_tools.py -v

# Only graph tests
pytest tests/test_graph.py -v

# Only API tests
pytest tests/test_api.py -v
```

---

## Running Evaluation

The evaluation framework tests the agent against 20 predefined cases:

```bash
# All categories
python -m evaluation.evaluate

# Specific categories
python -m evaluation.evaluate rag_retrieval web_search
```

Categories:
- `rag_retrieval` — Knowledge base search
- `web_search` — Web information retrieval
- `multi_tool` — Tasks requiring multiple tools
- `code_execution` — Computational tasks
- `memory_and_report` — Memory and report generation

---

## Example Tasks

Here are tasks that showcase different agent capabilities:

**Simple retrieval:**
> "What information is in the knowledge base about data protection?"

**Web research:**
> "What are the latest developments in the EU AI Act?"

**Multi-tool research:**
> "Research AI ethics guidelines from both our documents and the web, then compile a structured report"

**Computation (triggers HITL):**
> "Calculate compound interest for 10000 euros at 5% over 10 years"

**Memory:**
> "Remember that my preferred report format is markdown tables"

**Complex multi-step:**
> "Find the latest AI framework comparisons online, analyze them, and create a report with recommendations"
