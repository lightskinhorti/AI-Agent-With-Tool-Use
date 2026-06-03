from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ToolCallRecord(TypedDict):
    tool_name: str
    args: dict[str, Any]
    reasoning: str
    timestamp: str


class ToolResultRecord(TypedDict):
    tool_name: str
    result: str
    success: bool
    latency_ms: float
    timestamp: str


class AgentState(TypedDict):
    task: str
    messages: Annotated[list[BaseMessage], add_messages]

    plan: list[dict[str, Any]]
    current_step: int
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    tool_results: Annotated[list[ToolResultRecord], operator.add]

    reflections: Annotated[list[str], operator.add]
    final_answer: str

    requires_human: bool
    human_feedback: str
    pending_tool_call: dict[str, Any]

    error_count: int
    iteration_count: int
    status: Literal[
        "planning",
        "executing",
        "reviewing",
        "waiting_human",
        "complete",
        "error",
    ]

    metadata: dict[str, Any]
