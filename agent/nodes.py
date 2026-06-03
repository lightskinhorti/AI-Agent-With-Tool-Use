from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.config import settings
from agent.state import AgentState, ToolCallRecord, ToolResultRecord
from agent.tools import TOOL_REGISTRY, get_hitl_tool_names, get_tool_descriptions, register_tools


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.agent_model,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_token_usage(metadata: dict[str, Any], response: AIMessage) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = metadata.copy()
    metadata["total_input_tokens"] = metadata.get("total_input_tokens", 0) + (
        usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
    )
    metadata["total_output_tokens"] = metadata.get("total_output_tokens", 0) + (
        usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
    )
    metadata["total_tokens"] = (
        metadata["total_input_tokens"] + metadata["total_output_tokens"]
    )
    return metadata


def _parse_plan(content: str) -> list[dict[str, Any]]:
    json_match = re.search(r"\[.*\]", content, re.DOTALL)
    if json_match:
        try:
            plan = json.loads(json_match.group())
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass

    steps = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        for tool_name in TOOL_REGISTRY:
            if tool_name in line.lower():
                steps.append({
                    "action": tool_name,
                    "args": {},
                    "reasoning": line,
                })
                break
    return steps if steps else [{"action": "answer", "args": {}, "reasoning": content[:200]}]


def _parse_review(content: str) -> tuple[str, str]:
    content_lower = content.lower()
    if "complete" in content_lower or "sufficient" in content_lower or "finalize" in content_lower:
        decision = "complete"
    elif "replan" in content_lower or "different approach" in content_lower:
        decision = "replan"
    else:
        decision = "continue"
    return decision, content[:500]


async def planner(state: AgentState) -> dict[str, Any]:
    if not TOOL_REGISTRY:
        register_tools()

    llm = _get_llm()
    tool_descriptions = get_tool_descriptions()

    system_prompt = f"""You are a planning agent. Given a task, create a step-by-step execution plan.

Available tools:
{tool_descriptions}

IMPORTANT: Output a JSON array of steps. Each step must have:
- "action": tool name (one of: {', '.join(TOOL_REGISTRY.keys())}) or "answer" for direct answers
- "args": dict of arguments for the tool (match the tool's expected parameters)
- "reasoning": brief explanation of why this step is needed

Example output:
[
  {{"action": "rag_search", "args": {{"query": "relevant search query", "top_k": 5}}, "reasoning": "Search knowledge base first"}},
  {{"action": "web_search", "args": {{"query": "supplementary web query"}}, "reasoning": "Complement with web results"}},
  {{"action": "report_writer", "args": {{"title": "Report Title", "sections": ["Section 1"], "content": {{"Section 1": "content"}}}}, "reasoning": "Compile findings"}}
]

Keep plans focused: 2-5 steps max. Choose tools wisely."""

    messages = [SystemMessage(content=system_prompt)]

    if state.get("reflections"):
        reflection_text = "\n".join(state["reflections"])
        messages.append(
            HumanMessage(
                content=f"Previous attempt reflections (use these to improve the plan):\n{reflection_text}"
            )
        )

    if state.get("tool_results"):
        results_text = "\n".join(
            f"- {r['tool_name']}: {'OK' if r['success'] else 'FAIL'} — {r['result'][:200]}"
            for r in state["tool_results"]
        )
        messages.append(
            HumanMessage(content=f"Results from previous steps:\n{results_text}")
        )

    messages.append(HumanMessage(content=f"Task: {state['task']}"))

    response = await llm.ainvoke(messages)
    plan = _parse_plan(response.content)
    metadata = _update_token_usage(state.get("metadata") or {}, response)

    return {
        "plan": plan,
        "current_step": 0,
        "status": "executing",
        "messages": [response],
        "metadata": metadata,
    }


async def executor(state: AgentState) -> dict[str, Any]:
    if not TOOL_REGISTRY:
        register_tools()

    plan = state.get("plan") or []
    step_idx = state.get("current_step", 0)

    if step_idx >= len(plan):
        return {"status": "reviewing"}

    step = plan[step_idx]
    tool_name = step.get("action", "")
    tool_args = step.get("args", {})

    # Check HITL
    hitl_tools = get_hitl_tool_names()
    if tool_name in hitl_tools and state.get("human_feedback") != "approved":
        return {
            "requires_human": True,
            "pending_tool_call": {
                "tool_name": tool_name,
                "args": tool_args,
                "reasoning": step.get("reasoning", ""),
                "step_index": step_idx,
            },
            "status": "waiting_human",
            "tool_calls": [
                ToolCallRecord(
                    tool_name=tool_name,
                    args=tool_args,
                    reasoning=step.get("reasoning", ""),
                    timestamp=_now(),
                )
            ],
        }

    if tool_name == "answer" or tool_name not in TOOL_REGISTRY:
        return {
            "current_step": step_idx + 1,
            "status": "reviewing" if step_idx + 1 >= len(plan) else "executing",
        }

    tool = TOOL_REGISTRY[tool_name]
    result = await tool.run(**tool_args)

    return {
        "tool_calls": [
            ToolCallRecord(
                tool_name=tool_name,
                args=tool_args,
                reasoning=step.get("reasoning", ""),
                timestamp=_now(),
            )
        ],
        "tool_results": [
            ToolResultRecord(
                tool_name=tool_name,
                result=result.data if result.success else (result.error or "Unknown error"),
                success=result.success,
                latency_ms=result.latency_ms,
                timestamp=_now(),
            )
        ],
        "current_step": step_idx + 1,
        "error_count": state.get("error_count", 0) + (0 if result.success else 1),
        "requires_human": False,
        "human_feedback": "",
        "status": "reviewing" if step_idx + 1 >= len(plan) else "executing",
    }


async def reviewer(state: AgentState) -> dict[str, Any]:
    llm = _get_llm()

    results_summary = "\n".join(
        f"Step {i + 1} ({r['tool_name']}): {'OK' if r['success'] else 'FAIL'} — {r['result'][:300]}"
        for i, r in enumerate(state.get("tool_results") or [])
    )

    plan_summary = "\n".join(
        f"  {i + 1}. {s.get('action', '?')}: {s.get('reasoning', '')}"
        for i, s in enumerate(state.get("plan") or [])
    )

    current = state.get("current_step", 0)
    total = len(state.get("plan") or [])
    iteration = state.get("iteration_count", 0)

    system_prompt = """You are a reviewer agent. Analyze the task and tool results, then decide:

1. "complete" — results are sufficient to answer the task comprehensively
2. "replan" — results are insufficient, need a different approach (only if iteration < 3)
3. "continue" — there are remaining plan steps to execute

Also provide a brief reflection (2-3 sentences) on what was learned and what could be improved.

Format your response as:
DECISION: [complete|replan|continue]
REFLECTION: [your reflection]"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Task: {state['task']}\n\n"
                f"Plan ({current}/{total} steps executed):\n{plan_summary}\n\n"
                f"Results:\n{results_summary}\n\n"
                f"Iteration: {iteration + 1}/3"
            )
        ),
    ]

    response = await llm.ainvoke(messages)
    decision, reflection = _parse_review(response.content)

    if iteration >= 2 and decision == "replan":
        decision = "complete"
        reflection += " [Max iterations reached — finalizing with available results.]"

    next_status = {
        "complete": "complete",
        "replan": "planning",
        "continue": "executing",
    }.get(decision, "complete")

    metadata = _update_token_usage(state.get("metadata") or {}, response)

    return {
        "reflections": [reflection],
        "status": next_status,
        "messages": [response],
        "metadata": metadata,
        "iteration_count": iteration + 1,
    }


async def hitl_gate(state: AgentState) -> dict[str, Any]:
    feedback = state.get("human_feedback", "")

    if feedback == "approved":
        pending = state.get("pending_tool_call") or {}
        tool_name = pending.get("tool_name", "")
        tool_args = pending.get("args", {})

        if tool_name in TOOL_REGISTRY:
            tool = TOOL_REGISTRY[tool_name]
            result = await tool.run(**tool_args)
            step_idx = state.get("current_step", 0)

            return {
                "tool_results": [
                    ToolResultRecord(
                        tool_name=tool_name,
                        result=result.data if result.success else (result.error or ""),
                        success=result.success,
                        latency_ms=result.latency_ms,
                        timestamp=_now(),
                    )
                ],
                "current_step": step_idx + 1,
                "requires_human": False,
                "human_feedback": "",
                "pending_tool_call": {},
                "error_count": state.get("error_count", 0) + (0 if result.success else 1),
                "status": "reviewing",
            }

        return {"requires_human": False, "human_feedback": "", "status": "reviewing"}

    elif feedback == "rejected":
        return {
            "requires_human": False,
            "human_feedback": "",
            "pending_tool_call": {},
            "status": "complete",
            "reflections": ["Human rejected the pending tool call. Finalizing with partial results."],
        }

    return {"status": "waiting_human"}


async def finalizer(state: AgentState) -> dict[str, Any]:
    llm = _get_llm()

    results_context = "\n\n".join(
        f"[{r['tool_name']}]:\n{r['result']}"
        for r in (state.get("tool_results") or [])
        if r.get("success", False)
    )

    reflections_text = "\n".join(state.get("reflections") or [])

    system_prompt = """You are a synthesis agent. Create a comprehensive, well-structured answer
based on the research results provided. Include specific details, data, and references
from the tool results. If results are partial or insufficient, clearly state what
information is available and what gaps remain. Write in the same language as the task."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Task: {state['task']}\n\n"
                f"Research Results:\n{results_context}\n\n"
                f"Reflections:\n{reflections_text}"
            )
        ),
    ]

    response = await llm.ainvoke(messages)

    total_latency = sum(
        r.get("latency_ms", 0) for r in (state.get("tool_results") or [])
    )
    tool_results = state.get("tool_results") or []
    success_count = sum(1 for r in tool_results if r.get("success"))
    success_rate = success_count / max(len(tool_results), 1)

    metadata = _update_token_usage(state.get("metadata") or {}, response)
    metadata.update({
        "total_tool_latency_ms": round(total_latency, 1),
        "tool_success_rate": round(success_rate, 3),
        "total_tool_calls": len(tool_results),
        "tools_used": list({r["tool_name"] for r in tool_results}),
    })

    return {
        "final_answer": response.content,
        "status": "complete",
        "messages": [response],
        "metadata": metadata,
    }
