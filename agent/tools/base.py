from __future__ import annotations

import time
from abc import ABC, abstractmethod

import structlog
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class ToolResult(BaseModel):
    success: bool
    data: str
    error: str | None = None
    latency_ms: float = 0.0


class BaseTool(ABC):
    name: str = ""
    description: str = ""
    requires_human_approval: bool = False

    @abstractmethod
    def get_schema(self) -> type[BaseModel]:
        ...

    @abstractmethod
    async def _execute(self, **kwargs: object) -> str:
        ...

    async def run(self, **kwargs: object) -> ToolResult:
        log = structlog.get_logger(tool=self.name)
        start = time.monotonic()
        try:
            schema = self.get_schema()
            validated = schema(**kwargs)
            result = await self._run_with_retry(**validated.model_dump())
            latency = (time.monotonic() - start) * 1000
            log.info("tool_success", latency_ms=round(latency, 1))
            return ToolResult(success=True, data=result, latency_ms=latency)
        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            log.error("tool_failure", error=str(e), latency_ms=round(latency, 1))
            return ToolResult(
                success=False, data="", error=str(e), latency_ms=latency
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    async def _run_with_retry(self, **kwargs: object) -> str:
        return await self._execute(**kwargs)

    def to_langchain_tool(self):
        from langchain_core.tools import StructuredTool

        tool_instance = self

        async def _fn(**kwargs: object) -> str:
            result = await tool_instance.run(**kwargs)
            if result.success:
                return result.data
            return f"Error: {result.error}"

        return StructuredTool.from_function(
            coroutine=_fn,
            name=self.name,
            description=self.description,
            args_schema=self.get_schema(),
        )
