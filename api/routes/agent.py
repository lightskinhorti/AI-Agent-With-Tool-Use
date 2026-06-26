from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException

from agent.checkpointer import get_async_checkpointer
from agent.graph import compile_graph
from api.schemas import (
    AgentResultResponse,
    AgentRunRequest,
    AgentRunResponse,
    AgentStatusResponse,
    TraceResponse,
    TraceStep,
)

router = APIRouter()
log = structlog.get_logger()

active_runs: dict[str, asyncio.Task] = {}
run_metadata: dict[str, dict[str, Any]] = {}


def _initial_state(task: str) -> dict[str, Any]:
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


async def _run_graph(run_id: str, task: str) -> None:
    run_metadata[run_id] = {
        "task": task,
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    try:
        async with get_async_checkpointer() as checkpointer:
            app = compile_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": run_id}}
            run_metadata[run_id]["status"] = "running"
            await app.ainvoke(_initial_state(task), config=config)
            run_metadata[run_id]["status"] = "complete"
            log.info("agent_run_complete", run_id=run_id)
    except Exception as e:
        tb = traceback.format_exc()
        run_metadata[run_id]["status"] = "error"
        run_metadata[run_id]["error"] = str(e)
        run_metadata[run_id]["traceback"] = tb
        log.error("agent_run_failed", run_id=run_id, error=str(e), traceback=tb)


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(request: AgentRunRequest):
    run_id = request.thread_id or str(uuid4())

    if run_id in active_runs and not active_runs[run_id].done():
        raise HTTPException(409, f"Run {run_id} is already active")

    run_metadata[run_id] = {
        "task": request.task,
        "status": "queued",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    }
    task = asyncio.create_task(_run_graph(run_id, request.task))
    active_runs[run_id] = task

    return AgentRunResponse(
        run_id=run_id, status="started", message="Agent run initiated"
    )


@router.get("/status/{run_id}", response_model=AgentStatusResponse)
async def get_status(run_id: str):
    async with get_async_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": run_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)

    if not checkpoint_tuple:
        meta = run_metadata.get(run_id)
        if meta:
            return AgentStatusResponse(
                run_id=run_id,
                status=meta.get("status", "starting"),
                current_step=0, total_steps=0,
                requires_human=False, pending_tool_call=None,
                plan=[], tool_calls_made=0,
            )
        if run_id in active_runs:
            return AgentStatusResponse(
                run_id=run_id, status="starting", current_step=0,
                total_steps=0, requires_human=False,
                pending_tool_call=None, plan=[], tool_calls_made=0,
            )
        return AgentStatusResponse(
            run_id=run_id, status="unknown", current_step=0,
            total_steps=0, requires_human=False,
            pending_tool_call=None, plan=[], tool_calls_made=0,
        )

    state = checkpoint_tuple.checkpoint.get("channel_values", {})
    is_active = run_id in active_runs and not active_runs[run_id].done()
    meta_status = run_metadata.get(run_id, {}).get("status")
    status = state.get("status", "unknown")

    if meta_status == "error":
        status = "error"
    elif not is_active and status not in ("complete", "error", "waiting_human"):
        status = "complete"

    return AgentStatusResponse(
        run_id=run_id,
        status=status,
        current_step=state.get("current_step", 0),
        total_steps=len(state.get("plan") or []),
        requires_human=state.get("requires_human", False),
        pending_tool_call=state.get("pending_tool_call") or None,
        plan=state.get("plan") or [],
        tool_calls_made=len(state.get("tool_results") or []),
    )


@router.get("/result/{run_id}", response_model=AgentResultResponse)
async def get_result(run_id: str):
    async with get_async_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": run_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)

    meta = run_metadata.get(run_id, {})

    if not checkpoint_tuple:
        if meta.get("error"):
            return AgentResultResponse(
                run_id=run_id, status="error",
                final_answer=f"Error: {meta['error']}",
                metadata={}, tool_results=[], reflections=[],
            )
        if run_id in active_runs:
            return AgentResultResponse(
                run_id=run_id, status="running",
                final_answer="", metadata={}, tool_results=[], reflections=[],
            )
        raise HTTPException(404, f"Run {run_id} not found")

    state = checkpoint_tuple.checkpoint.get("channel_values", {})
    final_answer = state.get("final_answer", "")
    if not final_answer and meta.get("error"):
        final_answer = f"Error: {meta['error']}"

    return AgentResultResponse(
        run_id=run_id,
        status=meta.get("status") if meta.get("status") == "error" else state.get("status", "unknown"),
        final_answer=final_answer,
        metadata=state.get("metadata") or {},
        tool_results=[
            {
                "tool_name": r.get("tool_name", ""),
                "success": r.get("success", False),
                "result": r.get("result", "")[:500],
                "latency_ms": r.get("latency_ms", 0),
            }
            for r in (state.get("tool_results") or [])
        ],
        reflections=state.get("reflections") or [],
    )


@router.get("/trace/{run_id}", response_model=TraceResponse)
async def get_trace(run_id: str):
    async with get_async_checkpointer() as checkpointer:
        config = {"configurable": {"thread_id": run_id}}
        steps = []
        step_count = 0

        async for checkpoint_tuple in checkpointer.alist(config):
            checkpoint = checkpoint_tuple.checkpoint
            state = checkpoint.get("channel_values", {})
            metadata = checkpoint_tuple.metadata or {}

            steps.append(
                TraceStep(
                    step=step_count,
                    node=metadata.get("source", "unknown"),
                    status=state.get("status", "unknown"),
                    data={
                        "current_step": state.get("current_step", 0),
                        "requires_human": state.get("requires_human", False),
                        "tool_results_count": len(state.get("tool_results") or []),
                        "reflections_count": len(state.get("reflections") or []),
                    },
                )
            )
            step_count += 1

        return TraceResponse(run_id=run_id, steps=steps)
