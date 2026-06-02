from __future__ import annotations

import argparse
import asyncio
import sys
from uuid import uuid4

from agent.checkpointer import get_async_checkpointer
from agent.graph import compile_graph
from agent.logging_config import configure_logging
from agent.state import AgentState
from agent.tools import register_tools


def _initial_state(task: str) -> dict:
    return {
        "task": task,
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


async def run_agent(task: str, thread_id: str | None = None) -> dict:
    configure_logging()
    register_tools()

    thread_id = thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async with get_async_checkpointer() as checkpointer:
        app = compile_graph(checkpointer=checkpointer)
        result = await app.ainvoke(_initial_state(task), config=config)
        return result


async def _main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Agent CLI")
    parser.add_argument("task", help="Task for the agent to execute")
    parser.add_argument("--thread-id", help="Thread ID for persistence")
    args = parser.parse_args()

    result = await run_agent(args.task, args.thread_id)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result.get("final_answer", "No answer generated."))
    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    meta = result.get("metadata", {})
    print(f"  Total tokens:      {meta.get('total_tokens', 'N/A')}")
    print(f"  Tool calls:        {meta.get('total_tool_calls', 'N/A')}")
    print(f"  Tool success rate: {meta.get('tool_success_rate', 'N/A')}")
    print(f"  Tools used:        {meta.get('tools_used', [])}")
    print(f"  Tool latency:      {meta.get('total_tool_latency_ms', 'N/A')}ms")


if __name__ == "__main__":
    asyncio.run(_main())
