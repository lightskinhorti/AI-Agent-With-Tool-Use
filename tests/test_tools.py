from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp()
os.environ["MEMORY_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test_memory.db")


@pytest.mark.asyncio
async def test_memory_store_set_get():
    from agent.tools.memory_store import MemoryStoreTool

    tool = MemoryStoreTool()

    result = await tool.run(action="set", key="test_key", value="test_value_123")
    assert result.success
    assert "test_key" in result.data

    result = await tool.run(action="get", key="test_key")
    assert result.success
    assert "test_value_123" in result.data


@pytest.mark.asyncio
async def test_memory_store_list():
    from agent.tools.memory_store import MemoryStoreTool

    tool = MemoryStoreTool()
    await tool.run(action="set", key="key_a", value="value_a")
    await tool.run(action="set", key="key_b", value="value_b")

    result = await tool.run(action="list")
    assert result.success
    assert "key_a" in result.data
    assert "key_b" in result.data


@pytest.mark.asyncio
async def test_memory_store_unknown_action():
    from agent.tools.memory_store import MemoryStoreTool

    tool = MemoryStoreTool()
    result = await tool.run(action="delete", key="x")
    assert result.success
    assert "Unknown action" in result.data


@pytest.mark.asyncio
async def test_report_writer():
    from agent.tools.report_writer import ReportWriterTool

    tool = ReportWriterTool()
    result = await tool.run(
        title="Test Report",
        sections=["Introduction", "Findings"],
        content={
            "Introduction": "This is the intro.",
            "Findings": "Key findings here.",
        },
    )
    assert result.success
    assert "# Test Report" in result.data
    assert "Introduction" in result.data
    assert "Key findings here." in result.data
    assert "Table of Contents" in result.data


@pytest.mark.asyncio
async def test_code_executor_safe_code():
    from agent.tools.code_executor import CodeExecutorTool

    tool = CodeExecutorTool()
    result = await tool.run(
        code="result = sum(range(10))",
        description="Sum numbers",
    )
    assert result.success
    assert "45" in result.data


@pytest.mark.asyncio
async def test_code_executor_stdout():
    from agent.tools.code_executor import CodeExecutorTool

    tool = CodeExecutorTool()
    result = await tool.run(
        code="print('hello world')",
        description="Print test",
    )
    assert result.success
    assert "hello" in result.data


@pytest.mark.asyncio
async def test_code_executor_requires_approval():
    from agent.tools.code_executor import CodeExecutorTool

    tool = CodeExecutorTool()
    assert tool.requires_human_approval is True


@pytest.mark.asyncio
async def test_web_search_with_mock():
    from agent.tools.web_search import WebSearchTool

    mock_results = [
        {"title": "Result 1", "href": "https://example.com", "body": "Description 1"},
        {"title": "Result 2", "href": "https://example.org", "body": "Description 2"},
    ]

    tool = WebSearchTool()
    with patch("agent.tools.web_search.DDGS") as MockDDGS:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = mock_results
        MockDDGS.return_value = mock_instance

        result = await tool.run(query="test query", max_results=2)
        assert result.success
        assert "Result 1" in result.data
        assert "Result 2" in result.data


@pytest.mark.asyncio
async def test_rag_search_empty_collection():
    from agent.tools.rag_search import RAGSearchTool

    tool = RAGSearchTool()
    result = await tool.run(query="test query", top_k=3)
    assert result.success
    assert "empty" in result.data.lower() or "no documents" in result.data.lower()


@pytest.mark.asyncio
async def test_rag_search_with_documents():
    from agent.tools.rag_search import RAGSearchTool

    tool = RAGSearchTool()
    tool._collection.add(
        documents=[
            "Python is a programming language used for AI and ML.",
            "Machine learning involves training models on data.",
            "Natural language processing deals with text understanding.",
        ],
        ids=["doc1", "doc2", "doc3"],
        metadatas=[
            {"source": "test1"},
            {"source": "test2"},
            {"source": "test3"},
        ],
    )

    result = await tool.run(query="machine learning", top_k=2)
    assert result.success
    assert "machine learning" in result.data.lower() or "score=" in result.data


@pytest.mark.asyncio
async def test_document_fetch_html():
    from agent.tools.document_fetch import DocumentFetchTool

    tool = DocumentFetchTool()

    html = "<html><body><h1>Title</h1><p>Content here</p></body></html>"

    mock_response = MagicMock()
    mock_response.text = html
    mock_response.content = html.encode()
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("agent.tools.document_fetch.httpx.AsyncClient", return_value=mock_client):
        result = await tool.run(url="https://example.com/page")
        assert result.success
        assert "Title" in result.data
        assert "Content here" in result.data


@pytest.mark.asyncio
async def test_tool_result_model():
    from agent.tools.base import ToolResult

    r = ToolResult(success=True, data="test data", latency_ms=42.5)
    assert r.success
    assert r.data == "test data"
    assert r.error is None
    assert r.latency_ms == 42.5


@pytest.mark.asyncio
async def test_tool_base_to_langchain():
    from agent.tools.memory_store import MemoryStoreTool

    tool = MemoryStoreTool()
    lc_tool = tool.to_langchain_tool()
    assert lc_tool.name == "memory_store"
    assert lc_tool.description
