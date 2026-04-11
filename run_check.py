"""
按需记忆功能验收测试运行器（NM-01 ~ NM-04）
用法: python run_check.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import traceback

sys.path.insert(0, ".")

from app.services.chat_service import create_chat_service
from app.schemas.chat import ChatRequest

TESTS: list[dict] = [
    {
        "id": "NM-01a",
        "name": "跨会话记忆写入（companion session A）",
        "request": {
            "text": "我叫小明，我最喜欢的颜色是蓝色！",
            "mode": "companion",
            "age_hint": "6",
            "session_id": "nm-sess-A",
            "profile_id": "nm-test-child",
        },
        "checks": [
            ("memory.session_updated == True", lambda r: r.memory.session_updated is True),
            ("memory.written_types 非空", lambda r: len(r.memory.written_types) > 0),
            ("message 非空", lambda r: bool(r.message)),
        ],
    },
    {
        "id": "NM-01b",
        "name": "跨会话记忆召回（companion session B，同 profile_id）",
        "request": {
            "text": "你知道我最喜欢什么颜色吗？",
            "mode": "companion",
            "age_hint": "6",
            "session_id": "nm-sess-B",
            "profile_id": "nm-test-child",
        },
        "checks": [
            # selected_act 是最后一轮迭代的决策（respond=direct），
            # 真正的工具调用发生在第一轮，用 workflow_trace 验证
            ("'tools' in workflow_trace（read_memory 被调用）", lambda r: "tools" in r.metadata.workflow_trace),
            # chatbot 出现 ≥2 次说明经历了：call tool → observe → re-reason
            ("chatbot 迭代 ≥ 2 次", lambda r: r.metadata.workflow_trace.count("chatbot") >= 2),
            ("message 包含'蓝'（最终答案含颜色）", lambda r: "蓝" in r.message),
        ],
    },
    {
        "id": "NM-02a",
        "name": "当前会话记忆写入（companion session C）",
        "request": {
            "text": "我的名字叫小花，今天学会了画画！",
            "mode": "companion",
            "age_hint": "6",
            "session_id": "nm-same-01",
            "profile_id": "nm-test-child-2",
        },
        "checks": [
            ("memory.session_updated == True", lambda r: r.memory.session_updated is True),
            ("message 非空", lambda r: bool(r.message)),
        ],
    },
    {
        "id": "NM-02b",
        "name": "当前会话回忆（同 session，应从上下文直接回答）",
        "request": {
            "text": "你还记得我的名字吗？",
            "mode": "companion",
            "age_hint": "6",
            "session_id": "nm-same-01",
            "profile_id": "nm-test-child-2",
        },
        "checks": [
            ("react.selected_act == 'direct'（无需工具）", lambda r: r.react.selected_act == "direct"),
            ("'tools' not in workflow_trace", lambda r: "tools" not in r.metadata.workflow_trace),
            ("message 包含'小花'", lambda r: "小花" in r.message),
        ],
    },
    {
        "id": "NM-03",
        "name": "Education 模式 profile 记忆自动注入（不触发工具）",
        "request": {
            "text": "蜗牛是什么动物？",
            "mode": "education",
            "age_hint": "6",
            "session_id": "nm-edu-new",
            "profile_id": "nm-test-child",
        },
        "checks": [
            ("'plan' in workflow_trace", lambda r: "plan" in r.metadata.workflow_trace),
            ("'chatbot' not in workflow_trace（不走 ReAct）", lambda r: "chatbot" not in r.metadata.workflow_trace),
            ("message 非空", lambda r: bool(r.message)),
        ],
    },
    {
        "id": "NM-04a",
        "name": "Parent 跨会话记忆写入（session P1）",
        "request": {
            "text": "我的孩子叫小宝，今年4岁，最近在学认字。",
            "mode": "parent",
            "age_hint": "4",
            "session_id": "nm-parent-A",
            "profile_id": "nm-test-parent",
        },
        "checks": [
            ("memory.session_updated == True", lambda r: r.memory.session_updated is True),
            ("message 非空", lambda r: bool(r.message)),
        ],
    },
    {
        "id": "NM-04b",
        "name": "Parent 跨会话记忆召回（session P2，同 profile_id）",
        "request": {
            "text": "你还记得我孩子叫什么名字吗？",
            "mode": "parent",
            "age_hint": "4",
            "session_id": "nm-parent-B",
            "profile_id": "nm-test-parent",
        },
        "checks": [
            ("'tools' in workflow_trace（read_memory 被调用）", lambda r: "tools" in r.metadata.workflow_trace),
            ("chatbot 迭代 ≥ 2 次", lambda r: r.metadata.workflow_trace.count("chatbot") >= 2),
            ("message 包含'小宝'（最终答案含孩子名）", lambda r: "小宝" in r.message),
        ],
    },
]


def fmt(b: bool) -> str:
    return "✓" if b else "✗"


async def run_tests() -> list[dict]:
    print("初始化 ChatService...", flush=True)
    service = create_chat_service()
    print("完成，开始测试\n", flush=True)

    results = []
    for t in TESTS:
        tid, name = t["id"], t["name"]
        print(f"[{tid}] {name}", flush=True)
        try:
            req = ChatRequest.model_validate(t["request"])
            t0 = time.monotonic()
            resp = await service.explain_and_ask(req)
            elapsed = round(time.monotonic() - t0, 2)

            check_results = []
            for cname, cfn in t["checks"]:
                try:
                    passed = cfn(resp)
                except Exception as e:
                    passed = False
                    cname = f"{cname} [异常:{e}]"
                check_results.append((cname, passed))

            status = "PASS" if all(p for _, p in check_results) else "FAIL"
            result = {
                "id": tid, "name": name, "status": status, "elapsed_s": elapsed,
                "checks": check_results,
                "selected_act": resp.react.selected_act,
                "used_rag": resp.grounding.used_rag,
                "session_updated": resp.memory.session_updated,
                "written_types": resp.memory.written_types,
                "workflow_trace": resp.metadata.workflow_trace,
                "message_snippet": resp.message[:100].replace("\n", " "),
                "error": None,
            }
        except Exception:
            result = {
                "id": tid, "name": name, "status": "ERROR", "elapsed_s": 0,
                "checks": [(c, False) for c, _ in t["checks"]],
                "selected_act": None, "used_rag": None, "session_updated": None,
                "written_types": [], "workflow_trace": [], "message_snippet": "",
                "error": traceback.format_exc()[-600:],
            }

        results.append(result)
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥"}[result["status"]]
        print(f"  {icon} {result['status']}  {result['elapsed_s']}s", flush=True)
        print(f"     act={result['selected_act']}  rag={result['used_rag']}  session_updated={result['session_updated']}", flush=True)
        print(f"     written={result['written_types']}", flush=True)
        print(f"     trace={result['workflow_trace']}", flush=True)
        print(f"     msg: {result['message_snippet']}", flush=True)
        for cname, cp in result["checks"]:
            print(f"     {fmt(cp)} {cname}", flush=True)
        if result["error"]:
            print(f"     ERROR:\n{result['error']}", flush=True)
        print(flush=True)

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"{'='*50}")
    print(f"总计: {total}  ✅PASS: {passed}  ❌FAIL: {failed}  💥ERROR: {errors}")
    return results


if __name__ == "__main__":
    asyncio.run(run_tests())
