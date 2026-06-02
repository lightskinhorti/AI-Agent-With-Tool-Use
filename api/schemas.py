from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AgentRunRequest(BaseModel):
    task: str
    thread_id: str | None = None


class AgentRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class AgentStatusResponse(BaseModel):
    run_id: str
    status: str
    current_step: int
    total_steps: int
    requires_human: bool
    pending_tool_call: dict[str, Any] | None = None
    plan: list[dict[str, Any]] = []
    tool_calls_made: int = 0


class AgentResultResponse(BaseModel):
    run_id: str
    status: str
    final_answer: str
    metadata: dict[str, Any] = {}
    tool_results: list[dict[str, Any]] = []
    reflections: list[str] = []


class HITLRequest(BaseModel):
    feedback: str = ""


class HITLResponse(BaseModel):
    status: str
    run_id: str
    message: str


class TraceStep(BaseModel):
    step: int
    node: str
    status: str
    data: dict[str, Any] = {}


class TraceResponse(BaseModel):
    run_id: str
    steps: list[TraceStep] = []
