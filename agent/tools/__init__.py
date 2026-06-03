from __future__ import annotations

from agent.tools.base import BaseTool

TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tools() -> dict[str, BaseTool]:
    from agent.tools.code_executor import CodeExecutorTool
    from agent.tools.document_fetch import DocumentFetchTool
    from agent.tools.memory_store import MemoryStoreTool
    from agent.tools.rag_search import RAGSearchTool
    from agent.tools.report_writer import ReportWriterTool
    from agent.tools.web_search import WebSearchTool

    for tool_cls in [
        RAGSearchTool,
        WebSearchTool,
        DocumentFetchTool,
        CodeExecutorTool,
        ReportWriterTool,
        MemoryStoreTool,
    ]:
        t = tool_cls()
        TOOL_REGISTRY[t.name] = t
    return TOOL_REGISTRY


def get_langchain_tools() -> list:
    if not TOOL_REGISTRY:
        register_tools()
    return [t.to_langchain_tool() for t in TOOL_REGISTRY.values()]


def get_tool_descriptions() -> str:
    if not TOOL_REGISTRY:
        register_tools()
    return "\n".join(
        f"- {name}: {t.description}" for name, t in TOOL_REGISTRY.items()
    )


def get_hitl_tool_names() -> set[str]:
    if not TOOL_REGISTRY:
        register_tools()
    return {name for name, t in TOOL_REGISTRY.items() if t.requires_human_approval}
