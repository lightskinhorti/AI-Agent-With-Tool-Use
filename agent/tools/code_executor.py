from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field
from RestrictedPython import PrintCollector, compile_restricted, safe_builtins
from RestrictedPython.Eval import default_guarded_getiter
from RestrictedPython.Guards import guarded_unpack_sequence, safer_getattr

from agent.tools.base import BaseTool


class CodeExecutorInput(BaseModel):
    code: str = Field(description="Python code to execute")
    description: str = Field(
        default="", description="What the code does (shown to human for approval)"
    )


class CodeExecutorTool(BaseTool):
    name = "code_executor"
    description = (
        "Execute Python code in a sandboxed environment. "
        "REQUIRES human approval before execution. "
        "Available modules: math, json, datetime, statistics, collections."
    )
    requires_human_approval = True

    def get_schema(self) -> type[BaseModel]:
        return CodeExecutorInput

    async def _execute(self, code: str, description: str = "") -> str:
        return await asyncio.to_thread(self._sync_execute, code)

    def _sync_execute(self, code: str) -> str:
        byte_code = compile_restricted(code, "<agent_code>", "exec")

        import math, json, datetime, statistics, collections

        builtins = safe_builtins.copy()
        builtins.update({
            "sum": sum,
            "list": list,
            "dict": dict,
            "set": set,
            "frozenset": frozenset,
            "map": map,
            "filter": filter,
            "enumerate": enumerate,
            "max": max,
            "min": min,
            "any": any,
            "all": all,
            "print": print,
            "type": type,
            "reversed": reversed,
            "input": None,
            "open": None,
            "exec": None,
            "eval": None,
            "__import__": None,
        })

        restricted_globals = {"__builtins__": builtins}
        restricted_globals["_getiter_"] = default_guarded_getiter
        restricted_globals["_unpack_sequence_"] = guarded_unpack_sequence
        restricted_globals["_getattr_"] = safer_getattr
        restricted_globals["_print_"] = PrintCollector
        restricted_globals["_getitem_"] = lambda obj, key: obj[key]
        restricted_globals["math"] = math
        restricted_globals["json"] = json
        restricted_globals["datetime"] = datetime
        restricted_globals["statistics"] = statistics
        restricted_globals["collections"] = collections

        local_vars: dict = {}
        exec(byte_code, restricted_globals, local_vars)

        if "result" in local_vars:
            return str(local_vars["result"])

        printed = local_vars.get("_print")
        if printed and hasattr(printed, "txt") and printed.txt:
            return "".join(printed.txt).strip()

        user_vars = {
            k: str(v)[:200]
            for k, v in local_vars.items()
            if not k.startswith("_")
        }
        return str(user_vars) if user_vars else "Code executed successfully (no output)."
