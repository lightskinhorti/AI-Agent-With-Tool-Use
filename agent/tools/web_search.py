from __future__ import annotations

import asyncio

from duckduckgo_search import DDGS
from pydantic import BaseModel, Field

from agent.tools.base import BaseTool


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, ge=1, le=10, description="Max results")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. Use for current events, facts, "
        "or topics not covered in the knowledge base."
    )

    def get_schema(self) -> type[BaseModel]:
        return WebSearchInput

    async def _execute(self, query: str, max_results: int = 5) -> str:
        return await asyncio.to_thread(self._sync_search, query, max_results)

    def _sync_search(self, query: str, max_results: int) -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No web results found."

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")
            formatted.append(f"[{i}] {title}\n    URL: {href}\n    {body}")

        return "\n\n".join(formatted)
