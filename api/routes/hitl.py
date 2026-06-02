from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException

from agent.checkpointer import get_async_checkpointer
from agent.graph import compile_graph
from api.schemas import HITLRequest, HITLResponse

router = APIRouter()
log = structlog.get_logger()

_resume_tasks: dict[str, asyncio.Task] = {}


async def _resume_graph(run_id: str, feedback: str) -> None:
    try:
        async with get_async_checkpointer() as checkpointer:
            app = compile_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": run_id}}

            await app.aupdate_state(
                config,
                {
                    "human_feedback": feedback,
                    "requires_human": False,
                },
            )

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
    _resume_tasks[run_id] = task

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
    _resume_tasks[run_id] = task

    return HITLResponse(
        status="rejected",
        run_id=run_id,
        message="Agent execution resumed with rejection — will finalize with partial results",
    )
