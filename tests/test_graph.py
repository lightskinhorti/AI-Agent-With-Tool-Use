from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp()
os.environ["SQLITE_DB_PATH"] = str(os.path.join(tempfile.mkdtemp(), "test.db"))
os.environ["MEMORY_DB_PATH"] = str(os.path.join(tempfile.mkdtemp(), "mem.db"))


def test_graph_compiles():
    from agent.graph import compile_graph

    app = compile_graph()
    assert app is not None


def test_graph_has_expected_nodes():
    from agent.graph import build_graph

    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {"planner", "executor", "reviewer", "hitl_gate", "finalizer"}
    assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


def test_state_schema():
    from agent.state import AgentState

    state: AgentState = {
        "task": "test task",
        "messages": [],
        "plan": [],
        "current_step": 0,
        "tool_calls": [],
        "tool_results": [],
        "reflections": [],
        "final_answer": "",
        "requires_human": False,
        "human_feedback": "",
        "pending_tool_call": {},
        "error_count": 0,
        "iteration_count": 0,
        "status": "planning",
        "metadata": {},
    }
    assert state["task"] == "test task"
    assert state["status"] == "planning"


@pytest.mark.asyncio
async def test_planner_node():
    from agent.tools import register_tools

    register_tools()

    plan_json = json.dumps([
        {"action": "web_search", "args": {"query": "test"}, "reasoning": "Search for info"},
    ])

    mock_response = MagicMock()
    mock_response.content = plan_json
    mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

    with patch("agent.nodes.ChatAnthropic") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from agent.nodes import planner

        state = {
            "task": "Find information about AI",
            "messages": [],
            "plan": [],
            "current_step": 0,
            "tool_calls": [],
            "tool_results": [],
            "reflections": [],
            "metadata": {},
        }

        result = await planner(state)

        assert result["status"] == "executing"
        assert len(result["plan"]) > 0
        assert result["plan"][0]["action"] == "web_search"
        assert result["metadata"]["total_input_tokens"] == 100


@pytest.mark.asyncio
async def test_executor_node_runs_tool():
    from agent.tools import register_tools
    from agent.tools.base import ToolResult

    register_tools()

    with patch("agent.tools.web_search.DDGS") as MockDDGS:
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.text.return_value = [
            {"title": "Test", "href": "https://test.com", "body": "Test body"},
        ]
        MockDDGS.return_value = mock_instance

        from agent.nodes import executor

        state = {
            "task": "test",
            "plan": [{"action": "web_search", "args": {"query": "AI"}, "reasoning": "test"}],
            "current_step": 0,
            "error_count": 0,
            "human_feedback": "",
            "tool_calls": [],
            "tool_results": [],
        }

        result = await executor(state)

        assert result["current_step"] == 1
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool_name"] == "web_search"
        assert result["tool_results"][0]["success"]


@pytest.mark.asyncio
async def test_executor_triggers_hitl():
    from agent.tools import register_tools

    register_tools()
    from agent.nodes import executor

    state = {
        "task": "test",
        "plan": [
            {"action": "code_executor", "args": {"code": "print(1)", "description": "test"}, "reasoning": "test"},
        ],
        "current_step": 0,
        "error_count": 0,
        "human_feedback": "",
        "tool_calls": [],
        "tool_results": [],
    }

    result = await executor(state)

    assert result["requires_human"] is True
    assert result["status"] == "waiting_human"
    assert result["pending_tool_call"]["tool_name"] == "code_executor"


@pytest.mark.asyncio
async def test_hitl_gate_approved():
    from agent.tools import register_tools

    register_tools()
    from agent.nodes import hitl_gate

    state = {
        "human_feedback": "approved",
        "requires_human": True,
        "pending_tool_call": {
            "tool_name": "code_executor",
            "args": {"code": "result = 42", "description": "test"},
        },
        "current_step": 0,
        "error_count": 0,
        "tool_results": [],
    }

    result = await hitl_gate(state)

    assert result["requires_human"] is False
    assert result["status"] == "reviewing"
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["success"]


@pytest.mark.asyncio
async def test_hitl_gate_rejected():
    from agent.tools import register_tools

    register_tools()
    from agent.nodes import hitl_gate

    state = {
        "human_feedback": "rejected",
        "requires_human": True,
        "pending_tool_call": {"tool_name": "code_executor", "args": {}},
        "current_step": 0,
        "error_count": 0,
        "tool_results": [],
    }

    result = await hitl_gate(state)

    assert result["status"] == "complete"
    assert "rejected" in result["reflections"][0].lower()


@pytest.mark.asyncio
async def test_reviewer_node():
    mock_response = MagicMock()
    mock_response.content = "DECISION: complete\nREFLECTION: Results are sufficient to answer."
    mock_response.usage_metadata = {"input_tokens": 200, "output_tokens": 30}

    with patch("agent.nodes.ChatAnthropic") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from agent.nodes import reviewer

        state = {
            "task": "test task",
            "plan": [{"action": "web_search", "args": {}, "reasoning": "test"}],
            "current_step": 1,
            "tool_results": [
                {"tool_name": "web_search", "result": "found something", "success": True, "latency_ms": 100, "timestamp": ""},
            ],
            "reflections": [],
            "metadata": {},
            "iteration_count": 0,
        }

        result = await reviewer(state)

        assert result["status"] == "complete"
        assert len(result["reflections"]) == 1


@pytest.mark.asyncio
async def test_finalizer_node():
    mock_response = MagicMock()
    mock_response.content = "Here is the final synthesized answer based on research."
    mock_response.usage_metadata = {"input_tokens": 300, "output_tokens": 100}

    with patch("agent.nodes.ChatAnthropic") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        from agent.nodes import finalizer

        state = {
            "task": "test task",
            "tool_results": [
                {"tool_name": "web_search", "result": "data found", "success": True, "latency_ms": 50, "timestamp": ""},
            ],
            "reflections": ["Good results"],
            "metadata": {},
        }

        result = await finalizer(state)

        assert result["status"] == "complete"
        assert "final synthesized answer" in result["final_answer"]
        assert result["metadata"]["total_tool_calls"] == 1
        assert result["metadata"]["tool_success_rate"] == 1.0
