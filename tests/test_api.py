from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["ANTHROPIC_API_KEY"] = "test-key"
os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp()
os.environ["SQLITE_DB_PATH"] = str(os.path.join(tempfile.mkdtemp(), "test_api.db"))
os.environ["MEMORY_DB_PATH"] = str(os.path.join(tempfile.mkdtemp(), "test_api_mem.db"))


@pytest.fixture
def client():
    from api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_run_agent(client):
    with patch("api.routes.agent._run_graph", new_callable=AsyncMock):
        resp = client.post("/agent/run", json={"task": "Test task"})
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert data["status"] == "started"


def test_status_not_found(client):
    resp = client.get("/agent/status/nonexistent-id")
    assert resp.status_code == 404


def test_result_not_found(client):
    resp = client.get("/agent/result/nonexistent-id")
    assert resp.status_code == 404


def test_trace_empty(client):
    resp = client.get("/agent/trace/nonexistent-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["steps"] == []


def test_approve_not_found(client):
    resp = client.post("/agent/approve/nonexistent-id")
    assert resp.status_code == 404


def test_reject_not_found(client):
    resp = client.post("/agent/reject/nonexistent-id")
    assert resp.status_code == 404


def test_schemas():
    from api.schemas import (
        AgentRunRequest,
        AgentRunResponse,
        AgentStatusResponse,
        AgentResultResponse,
        HITLRequest,
        HITLResponse,
    )

    req = AgentRunRequest(task="test")
    assert req.task == "test"
    assert req.thread_id is None

    resp = AgentRunResponse(run_id="123", status="started", message="ok")
    assert resp.run_id == "123"

    status = AgentStatusResponse(
        run_id="123",
        status="executing",
        current_step=1,
        total_steps=3,
        requires_human=False,
    )
    assert status.current_step == 1

    result = AgentResultResponse(
        run_id="123",
        status="complete",
        final_answer="Answer here",
    )
    assert result.final_answer == "Answer here"

    hitl = HITLRequest(feedback="approved")
    assert hitl.feedback == "approved"

    hitl_resp = HITLResponse(status="approved", run_id="123", message="ok")
    assert hitl_resp.status == "approved"
