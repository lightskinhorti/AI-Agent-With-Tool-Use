from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, HTTPException

from agent.checkpointer import get_async_checkpointer
from agent.graph import compile_graph
from agent.tools import TOOL_REGISTRY, register_tools
from api.routes.agent import active_runs
from api.schemas import HITLRequest, HITLResponse

router = APIRouter()
log = structlog.get_logger()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _resume_graph(run_id: str, feedback: str) -> None:
    try:
        if not TOOL_REGISTRY:
            register_tools()

        async with get_async_checkpointer() as checkpointer:
            app = compile_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": run_id}}

            checkpoint_tuple = await checkpointer.aget_tuple(config)
            if not checkpoint_tuple:
                log.error("resume_no_checkpoint", run_id=run_id)
                return

            state = checkpoint_tuple.checkpoint.get("channel_values", {})

            if feedback == "approved":
                pending = state.get("pending_tool_call") or {}
                tool_name = pending.get("tool_name", "")
                tool_args = pending.get("args", {})

                update = {
                    "human_feedback": "approved",
                    "requires_human": False,
                    "pending_tool_call": {},
                    "status": "reviewing",
                }

                if tool_name in TOOL_REGISTRY:
                    tool = TOOL_REGISTRY[tool_name]
                    result = await tool.run(**tool_args)
                    step_idx = state.get("current_step", 0)

                    update["tool_results"] = [{
                        "tool_name": tool_name,
                        "result": result.data if result.success else (result.error or ""),
                        "success": result.success,
                        "latency_ms": result.latency_ms,
                        "timestamp": _now(),
                    }]
                    update["current_step"] = step_idx + 1
                    update["error_count"] = state.get("error_count", 0) + (0 if result.success else 1)
            else:
                update = {
                    "human_feedback": "rejected",
                    "requires_human": False,
                    "pending_tool_call": {},
                    "status": "complete",
                    "reflections": ["Human rejected the pending tool call. Finalizing with partial results."],
                }

            await app.aupdate_state(config, update, as_node="hitl_gate")
            await app.ainvoke(None, config=config)
            log.info("agent_resumed", run_id=run_id, feedback=feedback)
    except Exception as e:
        log.error("agent_resume_failed", run_id=run_id, error=str(e))


@router.post("/approve/{run_id}", response_model=HITLResponse)
async def approve(run_id: str, request: HITLRequest = HITLRequest()):
    async with get_async_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": run_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if not checkpoint_tuple:
            raise HTTPException(404, f"Run {run_id} not found")

        state = checkpoint_tuple.checkpoint.get("channel_values", {})
        if not state.get("requires_human"):
            raise HTTPException(400, f"Run {run_id} is not waiting for human input")

    task = asyncio.create_task(_resume_graph(run_id, "approved"))
    active_runs[run_id] = task

    return HITLResponse(
        status="approved",
        run_id=run_id,
        message="Agent execution resumed with approval",
    )


@router.post("/reject/{run_id}", response_model=HITLResponse)
async def reject(run_id: str, request: HITLRequest = HITLRequest()):
    async with get_async_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": run_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if not checkpoint_tuple:
            raise HTTPException(404, f"Run {run_id} not found")

        state = checkpoint_tuple.checkpoint.get("channel_values", {})
        if not state.get("requires_human"):
            raise HTTPException(400, f"Run {run_id} is not waiting for human input")

    task = asyncio.create_task(_resume_graph(run_id, "rejected"))
    active_runs[run_id] = task

    return HITLResponse(
        status="rejected",
        run_id=run_id,
        message="Agent execution resumed with rejection — will finalize with partial results",
    )
