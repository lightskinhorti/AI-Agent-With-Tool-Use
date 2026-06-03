"""Generate PDF documentation for RAG-Agent.

Usage:
    python docs/generate_pdf.py

Generates docs/RAG_Agent_Guide.pdf with architecture overview,
usage guide, and API reference.
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF


class AgentPDF(FPDF):
    def __init__(self) -> None:
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, "RAG-Agent Documentation", align="R")
        self.ln(5)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_cover(self) -> None:
        self.add_page()
        self.ln(60)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(30, 30, 30)
        self.cell(0, 15, "RAG-Agent", align="C")
        self.ln(15)
        self.set_font("Helvetica", "", 16)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, "AI Agent with Tool Use", align="C")
        self.ln(8)
        self.cell(0, 10, "Architecture & Usage Guide", align="C")
        self.ln(30)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "LangGraph  |  Anthropic Claude  |  FastAPI  |  ChromaDB", align="C")
        self.ln(8)
        self.cell(0, 8, "Persistent Checkpointing  |  Human-in-the-Loop  |  6 Real Tools", align="C")
        self.ln(30)
        self.set_draw_color(200, 200, 200)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, "github.com/lightskinhorti/AI-Agent-With-Tool-Use", align="C")

    def add_toc(self) -> None:
        self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, "Table of Contents")
        self.ln(15)

        toc_items = [
            ("1", "System Overview", 3),
            ("2", "Architecture", 4),
            ("3", "Agent Flow", 5),
            ("4", "Tool System", 7),
            ("5", "State Management", 9),
            ("6", "HITL (Human-in-the-Loop)", 10),
            ("7", "API Reference", 12),
            ("8", "Streamlit UI", 14),
            ("9", "Observability & Metrics", 15),
            ("10", "Evaluation Framework", 16),
            ("11", "Deployment (Docker)", 17),
            ("12", "Configuration Reference", 18),
            ("13", "Design Decisions", 19),
            ("14", "Example Flows", 20),
        ]

        for num, title, page in toc_items:
            self.set_font("Helvetica", "", 11)
            self.set_text_color(50, 50, 50)
            self.cell(10, 8, num + ".")
            self.cell(130, 8, title)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, str(page), align="R")
            self.ln(8)

    def section_title(self, title: str) -> None:
        self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, title)
        self.ln(12)

    def subsection(self, title: str) -> None:
        self.ln(5)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 9, title)
        self.ln(9)

    def body_text(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def code_block(self, code: str) -> None:
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        for line in code.strip().split("\n"):
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            self.cell(0, 4.5, "  " + safe_line, fill=True)
            self.ln(4.5)
        self.ln(3)

    def table_row(self, cells: list[str], bold: bool = False, widths: list[int] | None = None) -> None:
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        if bold:
            self.set_fill_color(235, 235, 235)
            self.set_text_color(30, 30, 30)
        else:
            self.set_fill_color(255, 255, 255)
            self.set_text_color(60, 60, 60)

        if widths is None:
            w = 190 // len(cells)
            widths = [w] * len(cells)

        for cell_text, width in zip(cells, widths):
            safe = cell_text.encode("latin-1", errors="replace").decode("latin-1")
            self.cell(width, 6.5, safe, border=1, fill=True)
        self.ln(6.5)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.set_text_color(60, 60, 60)
        safe = text.encode("latin-1", errors="replace").decode("latin-1")
        self.set_x(10)
        self.multi_cell(0, 5.5, "  - " + safe)


def generate_pdf() -> None:
    pdf = AgentPDF()
    pdf.alias_nb_pages()

    # --- Cover ---
    pdf.add_cover()

    # --- TOC ---
    pdf.add_toc()

    # --- 1. System Overview ---
    pdf.section_title("1. System Overview")
    pdf.body_text(
        "RAG-Agent is a production-ready autonomous research agent that receives a complex question or task "
        "and autonomously decides which tools to use, in what order, and when the task is sufficiently complete."
    )
    pdf.body_text(
        "Unlike a standard RAG pipeline (retrieve -> generate), this agent reasons in a multi-step loop: "
        "it plans, executes tools, reviews results, and can replan with a different strategy if needed. "
        "It also pauses for human approval before executing sensitive operations."
    )
    pdf.subsection("What makes this different from a chatbot?")
    pdf.bullet("Multi-step reasoning with explicit reflection and replanning")
    pdf.bullet("6 real tools with Pydantic validation, retry logic, and structured logging")
    pdf.bullet("Persistent state (SqliteSaver) - survives process restarts")
    pdf.bullet("Human-in-the-Loop via LangGraph interrupt_before mechanism")
    pdf.bullet("Full observability: token usage, tool latencies, success rates")
    pdf.bullet("Automated evaluation framework with 20 test cases")

    # --- 2. Architecture ---
    pdf.section_title("2. Architecture")
    pdf.body_text("The system consists of four layers:")
    pdf.subsection("Layer 1: LangGraph StateGraph (Core)")
    pdf.body_text(
        "A directed graph with 5 nodes: Planner, Executor, Reviewer, HITL Gate, and Finalizer. "
        "Each node is an async function that receives the full agent state and returns a partial update. "
        "Conditional edges route execution based on the agent's status."
    )
    pdf.subsection("Layer 2: Tool System")
    pdf.body_text(
        "6 tools, each inheriting from BaseTool. Every tool has: a Pydantic input schema for validation, "
        "retry logic with exponential backoff (tenacity, 3 attempts), structured logging with structlog, "
        "and latency measurement. Tools register in a central TOOL_REGISTRY."
    )
    pdf.subsection("Layer 3: FastAPI REST API")
    pdf.body_text(
        "Async API with endpoints for running agents, polling status, approving/rejecting HITL pauses, "
        "and retrieving execution traces. Agent tasks run as asyncio.create_task in the background."
    )
    pdf.subsection("Layer 4: Streamlit UI")
    pdf.body_text(
        "Chat interface with real-time status polling, HITL approval panel, execution trace viewer, "
        "and observability metrics. Communicates only with the API (never with the database directly)."
    )

    # --- 3. Agent Flow ---
    pdf.section_title("3. Agent Flow")
    pdf.body_text("The agent follows this flow for every task:")
    pdf.ln(3)

    pdf.subsection("Step 1: Planning")
    pdf.body_text(
        "The Planner node calls Claude with the task and descriptions of all available tools. "
        "It generates a JSON execution plan: a list of steps with tool name, arguments, and reasoning."
    )
    pdf.code_block(
        '[\n'
        '  {"action": "rag_search", "args": {"query": "AI regulation"}, "reasoning": "Search knowledge base"},\n'
        '  {"action": "web_search", "args": {"query": "EU AI Act 2025"}, "reasoning": "Get latest info"},\n'
        '  {"action": "report_writer", "args": {"title": "AI Regulation"}, "reasoning": "Compile report"}\n'
        ']'
    )

    pdf.subsection("Step 2: Execution")
    pdf.body_text(
        "The Executor iterates through plan steps. For each step, it looks up the tool in the registry, "
        "validates arguments, and executes. If the tool requires HITL (like code_executor), execution pauses."
    )

    pdf.subsection("Step 3: Review")
    pdf.body_text(
        "The Reviewer evaluates accumulated results. It decides: 'complete' (sufficient results), "
        "'replan' (try a different approach), or 'continue' (more steps remain). It also generates "
        "an explicit reflection for auditability."
    )

    pdf.subsection("Step 4: Finalization")
    pdf.body_text(
        "The Finalizer synthesizes all tool results and reflections into a comprehensive final answer "
        "using Claude. It also computes observability metrics: total tokens, tool latency, success rate."
    )

    # --- 4. Tool System ---
    pdf.section_title("4. Tool System")
    pdf.body_text("Every tool follows the BaseTool pattern:")
    pdf.code_block(
        "class BaseTool(ABC):\n"
        "    name: str\n"
        "    description: str\n"
        "    requires_human_approval: bool = False\n"
        "\n"
        "    def get_schema() -> type[BaseModel]    # Pydantic input validation\n"
        "    async def _execute(**kwargs) -> str     # Core logic (override this)\n"
        "    async def run(**kwargs) -> ToolResult   # Validate + execute + log\n"
        "    def to_langchain_tool()                 # Convert for LLM binding"
    )

    pdf.subsection("Tool Details")
    widths = [30, 55, 65, 40]
    pdf.table_row(["Tool", "Input", "Description", "HITL"], bold=True, widths=widths)
    pdf.table_row(["rag_search", "query, top_k", "Hybrid ChromaDB + BM25 search", "No"], widths=widths)
    pdf.table_row(["web_search", "query, max_results", "DuckDuckGo web search", "No"], widths=widths)
    pdf.table_row(["document_fetch", "url", "HTML/PDF download + parsing", "No"], widths=widths)
    pdf.table_row(["code_executor", "code, description", "RestrictedPython sandbox", "YES"], widths=widths)
    pdf.table_row(["report_writer", "title, sections, content", "Markdown report with TOC", "No"], widths=widths)
    pdf.table_row(["memory_store", "action, key, value", "SQLite key-value store", "No"], widths=widths)

    pdf.subsection("Hybrid Search (rag_search)")
    pdf.body_text(
        "The rag_search tool combines two retrieval strategies using Reciprocal Rank Fusion (RRF):\n\n"
        "1. Dense retrieval: ChromaDB with cosine similarity on embeddings\n"
        "2. Sparse retrieval: BM25 keyword matching on tokenized documents\n\n"
        "RRF formula: score(d) = sum(1 / (k + rank)) across both rankings, where k=60.\n"
        "This produces better results than either method alone."
    )

    # --- 5. State Management ---
    pdf.section_title("5. State Management")
    pdf.body_text(
        "The AgentState is a TypedDict with Annotated fields. LangGraph uses the annotations "
        "to determine how to merge state updates from nodes."
    )

    widths_state = [35, 50, 35, 70]
    pdf.table_row(["Field", "Type", "Reducer", "Purpose"], bold=True, widths=widths_state)
    pdf.table_row(["messages", "list[BaseMessage]", "add_messages", "Conversation history"], widths=widths_state)
    pdf.table_row(["tool_calls", "list[ToolCallRecord]", "operator.add", "Accumulates tool calls"], widths=widths_state)
    pdf.table_row(["tool_results", "list[ToolResultRecord]", "operator.add", "Accumulates results"], widths=widths_state)
    pdf.table_row(["reflections", "list[str]", "operator.add", "Reviewer insights"], widths=widths_state)
    pdf.table_row(["plan", "list[dict]", "replace", "Current execution plan"], widths=widths_state)
    pdf.table_row(["requires_human", "bool", "replace", "HITL trigger flag"], widths=widths_state)
    pdf.table_row(["error_count", "int", "replace", "Circuit breaker (max 3)"], widths=widths_state)
    pdf.table_row(["metadata", "dict", "replace", "Token usage, latencies"], widths=widths_state)

    # --- 6. HITL ---
    pdf.section_title("6. HITL (Human-in-the-Loop)")
    pdf.body_text(
        "Human-in-the-Loop is implemented using LangGraph's interrupt_before mechanism. "
        "This is a production-grade pattern used in enterprise systems for compliance and safety."
    )
    pdf.subsection("How it works")
    pdf.bullet("1. Executor detects tool requires human approval (code_executor)")
    pdf.bullet("2. Sets requires_human=True, stores pending_tool_call in state")
    pdf.bullet("3. Graph transitions toward hitl_gate node")
    pdf.bullet("4. interrupt_before=['hitl_gate'] suspends execution BEFORE entering the node")
    pdf.bullet("5. Full state is checkpointed to SQLite (process can die here)")
    pdf.bullet("6. API returns status='waiting_human' with the pending tool call details")
    pdf.bullet("7. Human reviews and calls POST /agent/approve or /agent/reject")
    pdf.bullet("8. graph.aupdate_state() injects human_feedback into checkpoint")
    pdf.bullet("9. graph.ainvoke(None) resumes from the exact saved state")
    pdf.bullet("10. hitl_gate node reads feedback: approved -> execute, rejected -> finalize")
    pdf.ln(3)
    pdf.body_text(
        "This is NOT a synchronous confirmation dialog. The process can crash between steps 5 and 7, "
        "and the agent will still resume correctly from the SQLite checkpoint."
    )

    # --- 7. API Reference ---
    pdf.section_title("7. API Reference")

    endpoints = [
        ("POST", "/agent/run", '{"task": "..."}', "Start agent task (async)"),
        ("GET", "/agent/status/{run_id}", "-", "Poll execution status"),
        ("GET", "/agent/result/{run_id}", "-", "Get final answer + metrics"),
        ("GET", "/agent/trace/{run_id}", "-", "Full execution trace"),
        ("POST", "/agent/approve/{run_id}", '{"feedback": ""}', "Approve HITL pause"),
        ("POST", "/agent/reject/{run_id}", '{"feedback": ""}', "Reject HITL pause"),
        ("GET", "/health", "-", "Health check"),
    ]

    widths_api = [20, 55, 50, 65]
    pdf.table_row(["Method", "Path", "Body", "Description"], bold=True, widths=widths_api)
    for method, path, body, desc in endpoints:
        pdf.table_row([method, path, body, desc], widths=widths_api)

    pdf.subsection("Status Response")
    pdf.code_block(
        '{\n'
        '  "run_id": "uuid",\n'
        '  "status": "executing",    // planning|executing|reviewing|waiting_human|complete|error\n'
        '  "current_step": 2,\n'
        '  "total_steps": 4,\n'
        '  "requires_human": false,\n'
        '  "pending_tool_call": null,\n'
        '  "tool_calls_made": 2\n'
        '}'
    )

    # --- 8. Streamlit UI ---
    pdf.section_title("8. Streamlit UI")
    pdf.body_text("The Streamlit frontend provides four components:")
    pdf.bullet("Chat Interface: Send tasks and view agent responses")
    pdf.bullet("Status Polling: Real-time progress updates (2-second interval)")
    pdf.bullet("HITL Panel: Review pending code, approve or reject execution")
    pdf.bullet("Sidebar: Execution trace timeline, token/latency metrics, reflections")
    pdf.ln(3)
    pdf.body_text(
        "The UI communicates exclusively with the FastAPI backend via HTTP requests. "
        "It never accesses SQLite or ChromaDB directly. This single-writer pattern "
        "prevents database locking issues."
    )

    # --- 9. Observability ---
    pdf.section_title("9. Observability & Metrics")
    pdf.body_text("Every agent run captures these metrics:")

    widths_obs = [55, 50, 85]
    pdf.table_row(["Metric", "Source", "Description"], bold=True, widths=widths_obs)
    pdf.table_row(["total_input_tokens", "LLM response", "Accumulated input tokens"], widths=widths_obs)
    pdf.table_row(["total_output_tokens", "LLM response", "Accumulated output tokens"], widths=widths_obs)
    pdf.table_row(["total_tokens", "Computed", "Sum of input + output"], widths=widths_obs)
    pdf.table_row(["total_tool_latency_ms", "Tool execution", "Sum of all tool durations"], widths=widths_obs)
    pdf.table_row(["tool_success_rate", "Computed", "Successful / total tool calls"], widths=widths_obs)
    pdf.table_row(["total_tool_calls", "Counted", "Number of tools executed"], widths=widths_obs)
    pdf.table_row(["tools_used", "Collected", "Unique tool names used"], widths=widths_obs)

    pdf.ln(3)
    pdf.body_text(
        "All tool executions are logged with structlog including: tool name, execution duration, "
        "success/failure, and error details. Logs use structured JSON format in production."
    )

    # --- 10. Evaluation ---
    pdf.section_title("10. Evaluation Framework")
    pdf.body_text(
        "The evaluation framework runs 20 predefined test cases across 5 categories "
        "and measures tool selection accuracy, keyword recall, latency, and token usage."
    )

    widths_eval = [50, 25, 45, 70]
    pdf.table_row(["Category", "Cases", "Expected Tools", "What it tests"], bold=True, widths=widths_eval)
    pdf.table_row(["rag_retrieval", "4", "rag_search", "Knowledge base search quality"], widths=widths_eval)
    pdf.table_row(["web_search", "4", "web_search", "Web information retrieval"], widths=widths_eval)
    pdf.table_row(["multi_tool", "4", "Multiple", "Multi-step reasoning"], widths=widths_eval)
    pdf.table_row(["code_execution", "4", "code_executor", "Computational tasks"], widths=widths_eval)
    pdf.table_row(["memory_and_report", "4", "memory/report", "Persistence + reporting"], widths=widths_eval)

    pdf.subsection("Metrics Computed")
    pdf.bullet("Pass rate: Overall and per category")
    pdf.bullet("Tool selection accuracy: Did the planner choose the right tools?")
    pdf.bullet("Keyword recall: Does the answer contain expected content?")
    pdf.bullet("Average latency: Per category")
    pdf.bullet("Token cost: Average tokens per task")

    # --- 11. Deployment ---
    pdf.section_title("11. Deployment (Docker)")
    pdf.body_text("docker-compose.yml defines two services:")
    pdf.code_block(
        "services:\n"
        "  api:      # FastAPI on port 8000 with health check\n"
        "  ui:       # Streamlit on port 8501 (depends on api)\n"
        "\n"
        "volumes:\n"
        "  agent-data:  # Shared volume for SQLite + ChromaDB persistence"
    )
    pdf.body_text(
        "The Dockerfile uses a multi-stage build (python:3.11-slim) to minimize image size. "
        "ChromaDB runs in embedded mode (not a separate container). "
        "Health checks ensure the UI starts only after the API is ready."
    )

    # --- 12. Configuration ---
    pdf.section_title("12. Configuration Reference")

    widths_conf = [50, 50, 90]
    pdf.table_row(["Variable", "Default", "Description"], bold=True, widths=widths_conf)
    pdf.table_row(["ANTHROPIC_API_KEY", "(required)", "Anthropic API key"], widths=widths_conf)
    pdf.table_row(["AGENT_MODEL", "claude-sonnet-4-20250514", "Claude model identifier"], widths=widths_conf)
    pdf.table_row(["AGENT_MAX_STEPS", "15", "Max execution steps per run"], widths=widths_conf)
    pdf.table_row(["HITL_TIMEOUT_SECONDS", "300", "Human approval timeout"], widths=widths_conf)
    pdf.table_row(["CHROMA_PERSIST_DIR", "./data/chroma", "ChromaDB storage path"], widths=widths_conf)
    pdf.table_row(["SQLITE_DB_PATH", "./data/agent.db", "Checkpoint database path"], widths=widths_conf)
    pdf.table_row(["MEMORY_DB_PATH", "./data/memory.db", "Memory tool database"], widths=widths_conf)
    pdf.table_row(["LOG_LEVEL", "INFO", "Logging level"], widths=widths_conf)

    # --- 13. Design Decisions ---
    pdf.section_title("13. Design Decisions")

    pdf.subsection("SqliteSaver vs MemorySaver")
    pdf.body_text(
        "MemorySaver stores state in RAM. It's lost when the process restarts. "
        "SqliteSaver persists every graph transition to disk. The agent can crash mid-execution, "
        "restart, and resume from the exact step where it stopped. "
        "95% of tutorials use MemorySaver. We use SqliteSaver because this is production, not a demo."
    )

    pdf.subsection("interrupt_before vs confirmation dialog")
    pdf.body_text(
        "LangGraph's interrupt_before suspends graph execution and persists state. "
        "The graph can be resumed hours later via API. A confirmation dialog is synchronous "
        "and blocking. Enterprise systems need async, auditable approval workflows."
    )

    pdf.subsection("Hybrid search (ChromaDB + BM25)")
    pdf.body_text(
        "Dense embeddings capture semantic similarity but miss exact keywords. "
        "BM25 captures exact terms but misses meaning. Reciprocal Rank Fusion combines "
        "both rankings for consistently better retrieval quality."
    )

    pdf.subsection("RestrictedPython vs subprocess")
    pdf.body_text(
        "RestrictedPython runs code in-process with restricted builtins. "
        "Faster and simpler than subprocess/container sandboxing. "
        "The allowlist is explicit: math, json, datetime, statistics, collections. "
        "For production, Docker-based sandboxing would be the natural next step."
    )

    pdf.subsection("asyncio.create_task vs Celery")
    pdf.body_text(
        "Agent tasks run as asyncio tasks in the same event loop. "
        "Celery would add Redis/RabbitMQ as dependencies, which is overkill for this scope. "
        "Since state is checkpointed to SQLite, crashed tasks can be resumed from the last checkpoint."
    )

    # --- 14. Example Flows ---
    pdf.section_title("14. Example Flows")

    pdf.subsection("Example 1: Multi-tool research")
    pdf.body_text('Task: "Research the latest AI agent frameworks and create a comparison report"')
    pdf.ln(2)
    pdf.bullet("Step 1 - Planner generates: [rag_search, web_search, web_search, report_writer]")
    pdf.bullet("Step 2 - rag_search('AI agent frameworks') -> returns knowledge base results")
    pdf.bullet("Step 3 - web_search('AI agent frameworks 2025 comparison') -> web results")
    pdf.bullet("Step 4 - web_search('LangGraph vs CrewAI vs AutoGen') -> more web results")
    pdf.bullet("Step 5 - Reviewer: 'Sufficient information. Proceed to report.'")
    pdf.bullet("Step 6 - report_writer generates structured Markdown report")
    pdf.bullet("Step 7 - Finalizer synthesizes comprehensive answer with references")

    pdf.subsection("Example 2: HITL flow (code execution)")
    pdf.body_text('Task: "Calculate compound interest for 10000 euros at 5% for 10 years"')
    pdf.ln(2)
    pdf.bullet("Step 1 - Planner: [code_executor]")
    pdf.bullet("Step 2 - Executor detects code_executor requires HITL")
    pdf.bullet("Step 3 - Graph PAUSES at hitl_gate. State saved to SQLite")
    pdf.bullet("Step 4 - API returns status='waiting_human' with the code to review")
    pdf.bullet("Step 5 - Human reviews code, calls POST /agent/approve")
    pdf.bullet("Step 6 - Graph RESUMES. code_executor runs: result = 16288.95")
    pdf.bullet("Step 7 - Finalizer: 'The compound interest result is 16,288.95 euros'")

    # Save
    output_path = Path(__file__).parent / "RAG_Agent_Guide.pdf"
    pdf.output(str(output_path))
    print(f"PDF generated: {output_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    generate_pdf()
