from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent.config import settings
from agent.tools.base import BaseTool


class MemoryStoreInput(BaseModel):
    action: str = Field(description="One of: 'get', 'set', 'list'")
    key: str = Field(default="", description="Key to get or set")
    value: str = Field(default="", description="Value to store (for 'set')")


class MemoryStoreTool(BaseTool):
    name = "memory_store"
    description = (
        "Persistent key-value store for cross-session memory. "
        "Use to remember facts, preferences, or intermediate results. "
        "Actions: get (retrieve by key), set (store key-value), list (show all keys)."
    )

    def __init__(self) -> None:
        settings.memory_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(settings.memory_db_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.commit()
        conn.close()

    def get_schema(self) -> type[BaseModel]:
        return MemoryStoreInput

    async def _execute(
        self, action: str, key: str = "", value: str = ""
    ) -> str:
        return await asyncio.to_thread(self._sync_execute, action, key, value)

    def _sync_execute(self, action: str, key: str, value: str) -> str:
        conn = sqlite3.connect(str(settings.memory_db_path))
        try:
            if action == "get":
                row = conn.execute(
                    "SELECT value FROM memory WHERE key = ?", (key,)
                ).fetchone()
                return row[0] if row else f"No value found for key '{key}'"

            elif action == "set":
                if not key:
                    return "Error: key is required for 'set' action"
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "INSERT OR REPLACE INTO memory (key, value, updated_at) "
                    "VALUES (?, ?, ?)",
                    (key, value, now),
                )
                conn.commit()
                return f"Stored '{key}' = '{value[:100]}'"

            elif action == "list":
                rows = conn.execute(
                    "SELECT key, substr(value, 1, 80), updated_at "
                    "FROM memory ORDER BY updated_at DESC LIMIT 20"
                ).fetchall()
                if not rows:
                    return "Memory is empty."
                return "\n".join(
                    f"- {r[0]}: {r[1]}{'...' if len(r[1]) >= 80 else ''} (updated: {r[2]})"
                    for r in rows
                )

            else:
                return f"Unknown action '{action}'. Use 'get', 'set', or 'list'."
        finally:
            conn.close()
