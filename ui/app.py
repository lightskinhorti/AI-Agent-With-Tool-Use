from __future__ import annotations

import os
import time

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="RAG-Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RAG-Agent")
st.caption("AI Agent with Tool Use — LangGraph + Anthropic Claude")

# --- Session State ---
if "run_id" not in st.session_state:
    st.session_state.run_id = None
    st.session_state.messages = []
    st.session_state.agent_status = "idle"


def _api(method: str, path: str, **kwargs) -> dict | None:
    try:
        resp = getattr(requests, method)(f"{API_BASE}{path}", timeout=10, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# --- Sidebar: Trace & Metrics ---
with st.sidebar:
    st.header("Execution Trace")

    if st.session_state.run_id:
        trace = _api("get", f"/agent/trace/{st.session_state.run_id}")
        if trace and trace.get("steps"):
            for step in trace["steps"]:
                icon = {"complete": "✅", "executing": "⚡", "waiting_human": "⏸️"}.get(
                    step.get("status", ""), "🔄"
                )
                with st.expander(
                    f"{icon} Step {step['step']}: {step['node']}",
                    expanded=False,
                ):
                    st.json(step.get("data", {}))
        else:
            st.info("No trace data yet.")

    st.divider()
    st.header("Metrics")

    if st.session_state.run_id and st.session_state.agent_status == "complete":
        result = _api("get", f"/agent/result/{st.session_state.run_id}")
        if result:
            meta = result.get("metadata", {})
            col1, col2 = st.columns(2)
            col1.metric("Total Tokens", meta.get("total_tokens", "—"))
            col2.metric("Tool Calls", meta.get("total_tool_calls", "—"))

            col3, col4 = st.columns(2)
            rate = meta.get("tool_success_rate", 0)
            col3.metric("Success Rate", f"{rate:.0%}" if isinstance(rate, float) else "—")
            latency = meta.get("total_tool_latency_ms", 0)
            col4.metric("Tool Latency", f"{latency:.0f}ms" if latency else "—")

            tools_used = meta.get("tools_used", [])
            if tools_used:
                st.caption(f"Tools used: {', '.join(tools_used)}")

            st.divider()
            st.subheader("Reflections")
            for r in result.get("reflections", []):
                st.markdown(f"> {r[:300]}")
    else:
        st.info("Run an agent task to see metrics.")

    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.run_id = None
        st.session_state.messages = []
        st.session_state.agent_status = "idle"
        st.rerun()

# --- Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- HITL Panel ---
if st.session_state.agent_status == "waiting_human" and st.session_state.run_id:
    status_data = _api("get", f"/agent/status/{st.session_state.run_id}")
    if status_data and status_data.get("requires_human"):
        pending = status_data.get("pending_tool_call") or {}
        st.warning("⚠️ Human Approval Required")

        with st.container(border=True):
            st.subheader(f"Tool: `{pending.get('tool_name', 'unknown')}`")
            st.caption(pending.get("reasoning", ""))

            args = pending.get("args", {})
            if "code" in args:
                st.code(args["code"], language="python")
            else:
                st.json(args)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Approve", type="primary", use_container_width=True):
                    _api("post", f"/agent/approve/{st.session_state.run_id}")
                    st.session_state.agent_status = "running"
                    st.rerun()
            with col2:
                if st.button("❌ Reject", type="secondary", use_container_width=True):
                    _api("post", f"/agent/reject/{st.session_state.run_id}")
                    st.session_state.agent_status = "running"
                    st.rerun()

# --- Input ---
task = st.chat_input("Ask the agent a question or assign a task...")

if task:
    st.session_state.messages.append({"role": "user", "content": task})
    with st.chat_message("user"):
        st.markdown(task)

    response = _api("post", "/agent/run", json={"task": task})
    if response:
        st.session_state.run_id = response["run_id"]
        st.session_state.agent_status = "running"

# --- Polling ---
if st.session_state.agent_status == "running" and st.session_state.run_id:
    with st.status("Agent is working...", expanded=True) as status_widget:
        max_polls = 120
        for _ in range(max_polls):
            status_data = _api("get", f"/agent/status/{st.session_state.run_id}")
            if not status_data:
                break

            agent_status = status_data.get("status", "unknown")
            current = status_data.get("current_step", 0)
            total = status_data.get("total_steps", 0)
            tools_done = status_data.get("tool_calls_made", 0)

            st.write(f"Status: **{agent_status}** | Step {current}/{total} | Tool calls: {tools_done}")

            if agent_status == "waiting_human":
                st.session_state.agent_status = "waiting_human"
                status_widget.update(label="⏸️ Waiting for human approval", state="error")
                st.rerun()
                break
            elif agent_status in ("complete", "error"):
                st.session_state.agent_status = "complete"
                status_widget.update(
                    label="✅ Complete" if agent_status == "complete" else "❌ Error",
                    state="complete" if agent_status == "complete" else "error",
                )

                result = _api("get", f"/agent/result/{st.session_state.run_id}")
                if result and result.get("final_answer"):
                    answer = result["final_answer"]
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                break

            time.sleep(2)

    if st.session_state.agent_status == "complete":
        st.rerun()
