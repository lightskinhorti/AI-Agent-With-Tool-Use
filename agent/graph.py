from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agent.nodes import executor, finalizer, hitl_gate, planner, reviewer
from agent.state import AgentState


def _route_after_executor(state: AgentState) -> str:
    if state.get("error_count", 0) >= 3:
        return "error"
    if state.get("status") == "waiting_human":
        return "waiting_human"
    plan = state.get("plan") or []
    current = state.get("current_step", 0)
    if current >= len(plan):
        return "reviewing"
    return "executing"


def _route_after_reviewer(state: AgentState) -> str:
    status = state.get("status", "complete")
    if status == "planning":
        return "replanning"
    if status == "executing":
        return "continue_executing"
    return "finalizing"


def _route_after_hitl(state: AgentState) -> str:
    status = state.get("status", "")
    if status == "complete":
        return "finalizing"
    if status == "waiting_human":
        return "pause"
    return "reviewing"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("reviewer", reviewer)
    graph.add_node("hitl_gate", hitl_gate)
    graph.add_node("finalizer", finalizer)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "executor")

    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "executing": "executor",
            "reviewing": "reviewer",
            "waiting_human": "hitl_gate",
            "error": "finalizer",
        },
    )

    graph.add_conditional_edges(
        "reviewer",
        _route_after_reviewer,
        {
            "finalizing": "finalizer",
            "replanning": "planner",
            "continue_executing": "executor",
        },
    )

    graph.add_conditional_edges(
        "hitl_gate",
        _route_after_hitl,
        {
            "reviewing": "reviewer",
            "finalizing": "finalizer",
            "pause": END,
        },
    )

    graph.add_edge("finalizer", END)

    return graph


def compile_graph(checkpointer: Any = None):
    graph = build_graph()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_gate"],
    )
