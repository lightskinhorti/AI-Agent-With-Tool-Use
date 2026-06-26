from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.logging_config import configure_logging
from agent.config import settings
from agent.tools import register_tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    register_tools()
    Path("data").mkdir(exist_ok=True)
    Path("data/reports").mkdir(exist_ok=True)
    yield


app = FastAPI(
    title="RAG-Agent API",
    description="Production-ready AI agent with tool use, HITL, and observability",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes.agent import router as agent_router
from api.routes.hitl import router as hitl_router
from api.routes.reports import router as reports_router

app.include_router(agent_router, prefix="/agent", tags=["agent"])
app.include_router(hitl_router, prefix="/agent", tags=["hitl"])
app.include_router(reports_router, prefix="/reports", tags=["reports"])


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.agent_model}
