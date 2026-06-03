from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalResult:
    test_id: str
    category: str
    difficulty: str
    passed: bool
    tool_match: bool
    keyword_recall: float
    latency_s: float
    token_count: int
    tools_used: list[str] = field(default_factory=list)
    error: str | None = None


async def run_single_eval(case: dict) -> EvalResult:
    from agent.checkpointer import get_async_checkpointer
    from agent.graph import compile_graph
    from agent.run import _initial_state

    thread_id = f"eval_{case['id']}"
    config = {"configurable": {"thread_id": thread_id}}

    start = time.monotonic()
    try:
        async with get_async_checkpointer() as checkpointer:
            app = compile_graph(checkpointer=checkpointer)

            initial = _initial_state(case["task"])
            # Auto-approve HITL in eval mode
            initial["human_feedback"] = "approved"

            result = await app.ainvoke(initial, config=config)

        latency = time.monotonic() - start

        tool_results = result.get("tool_results") or []
        tools_used = list({r["tool_name"] for r in tool_results})
        expected_tools = set(case.get("expected_tools", []))
        tool_match = expected_tools.issubset(set(tools_used))

        answer = (result.get("final_answer") or "").lower()
        expected_kw = case.get("expected_keywords", [])
        kw_found = sum(1 for kw in expected_kw if kw.lower() in answer)
        keyword_recall = kw_found / max(len(expected_kw), 1)

        passed = tool_match and keyword_recall >= 0.4
        metadata = result.get("metadata") or {}

        return EvalResult(
            test_id=case["id"],
            category=case.get("category", ""),
            difficulty=case.get("difficulty", ""),
            passed=passed,
            tool_match=tool_match,
            keyword_recall=round(keyword_recall, 2),
            latency_s=round(latency, 1),
            token_count=metadata.get("total_tokens", 0),
            tools_used=tools_used,
        )

    except Exception as e:
        return EvalResult(
            test_id=case["id"],
            category=case.get("category", ""),
            difficulty=case.get("difficulty", ""),
            passed=False,
            tool_match=False,
            keyword_recall=0.0,
            latency_s=round(time.monotonic() - start, 1),
            token_count=0,
            error=str(e),
        )


def print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    tool_match = sum(1 for r in results if r.tool_match)
    avg_kw = sum(r.keyword_recall for r in results) / max(total, 1)
    avg_latency = sum(r.latency_s for r in results) / max(total, 1)
    total_tokens = sum(r.token_count for r in results)

    print("\n" + "=" * 70)
    print("RAG-AGENT EVALUATION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    print(f"{'Total test cases':<30} {total:>15}")
    print(f"{'Passed':<30} {passed:>15} ({passed/max(total,1):.0%})")
    print(f"{'Tool selection accuracy':<30} {tool_match:>15} ({tool_match/max(total,1):.0%})")
    print(f"{'Avg keyword recall':<30} {avg_kw:>15.2f}")
    print(f"{'Avg latency (s)':<30} {avg_latency:>15.1f}")
    print(f"{'Total tokens used':<30} {total_tokens:>15}")

    # By category
    categories = sorted(set(r.category for r in results))
    print(f"\n{'Category':<25} {'Pass':>6} {'Total':>7} {'Rate':>7}")
    print("-" * 45)
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_total = len(cat_results)
        print(f"{cat:<25} {cat_passed:>6} {cat_total:>7} {cat_passed/max(cat_total,1):>7.0%}")

    # Individual results
    print(f"\n{'ID':<20} {'Pass':>5} {'Tools':>6} {'KW':>6} {'Latency':>8} {'Error'}")
    print("-" * 70)
    for r in results:
        err = (r.error[:20] + "...") if r.error else ""
        print(
            f"{r.test_id:<20} {'✓' if r.passed else '✗':>5} "
            f"{'✓' if r.tool_match else '✗':>6} {r.keyword_recall:>6.2f} "
            f"{r.latency_s:>7.1f}s {err}"
        )

    print("=" * 70)


async def main() -> None:
    from agent.logging_config import configure_logging
    from agent.tools import register_tools

    configure_logging()
    register_tools()

    test_cases_path = Path(__file__).parent / "test_cases.json"
    with open(test_cases_path) as f:
        cases = json.load(f)

    # Option to run subset
    if len(sys.argv) > 1:
        categories = sys.argv[1:]
        cases = [c for c in cases if c.get("category") in categories]
        print(f"Running {len(cases)} test cases for categories: {categories}")
    else:
        print(f"Running all {len(cases)} test cases")

    results = []
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] Running: {case['id']} — {case['task'][:60]}...")
        result = await run_single_eval(case)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  → {status} (tools: {result.tools_used}, kw: {result.keyword_recall}, {result.latency_s}s)")

    print_summary(results)

    # Save results
    output_path = Path("data/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "tool_match": r.tool_match,
                    "keyword_recall": r.keyword_recall,
                    "latency_s": r.latency_s,
                    "token_count": r.token_count,
                    "tools_used": r.tools_used,
                    "error": r.error,
                }
                for r in results
            ],
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
