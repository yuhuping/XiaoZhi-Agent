from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.nodes.execute import ExecuteNode
from app.agent.nodes.plan import PlanNode
from app.tools.calculate import safe_calculate


def _make_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "session_id": "test-session",
        "profile_id": "default_child",
        "user_input": "什么是勾股定理",
        "latest_user_text": "什么是勾股定理",
        "text_input": "什么是勾股定理",
        "interaction_mode": "education",
        "current_topic": None,
        "Memory": {},
        "workflow_trace": [],
        "messages": [],
        "plan_steps": [],
        "plan_raw": "",
        "retrieved_chunks": [],
        "child_age_band": "3-8",
        "image_base64": None,
        "image_url": None,
        "image_mime_type": None,
        "selected_act": "direct",
        "selected_tool": None,
        "tool_input": {},
    }
    base.update(overrides)
    return base


class TestPlanNode:
    def test_plan_generates_steps_and_retrieval(self) -> None:
        """Plan with needs_retrieval=True should set tool_calls and selected_act."""
        fake_plan = {
            "steps": ["理解勾股定理的定义", "用简单例子说明"],
            "needs_retrieval": True,
            "retrieval_query": "勾股定理",
        }
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        node = PlanNode(tools)
        state = _make_state()

        result = asyncio.run(node(state))

        assert result["plan_steps"] == ["理解勾股定理的定义", "用简单例子说明"]
        assert result["selected_act"] == "retrieve_knowledge"
        assert result["selected_tool"] == "retrieve_knowledge"
        assert result["tool_input"]["query"] == "勾股定理"
        assert len(result["messages"]) == 1
        assert result["messages"][0].tool_calls  # has tool_calls

    def test_plan_emits_public_workflow_events(self) -> None:
        fake_plan = {
            "steps": ["观察蜗牛壳的样子", "查找壳的作用", "用小实验总结"],
            "needs_retrieval": True,
            "retrieval_query": "蜗牛 壳 作用",
        }
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        event_writer = AsyncMock()

        asyncio.run(PlanNode(tools)(_make_state(stream_event_writer=event_writer)))

        events = [call.args[0] for call in event_writer.await_args_list]
        assert events[0] == {
            "type": "workflow",
            "event": "planning_started",
            "phase": "planning",
            "status": "active",
            "title": "正在规划学习路线",
        }
        assert events[1]["phase"] == "planning"
        assert events[1]["status"] == "completed"
        assert events[1]["steps"] == fake_plan["steps"]
        assert events[1]["needs_retrieval"] is True
        assert events[2]["phase"] == "retrieval"
        assert events[2]["status"] == "active"
        assert "retrieval_query" not in events[2]

    def test_plan_no_retrieval_goes_direct(self) -> None:
        """Plan with needs_retrieval=False should set selected_act=direct, no tool_calls."""
        fake_plan = {
            "steps": ["计算3+5"],
            "needs_retrieval": False,
            "retrieval_query": "",
        }
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        node = PlanNode(tools)
        state = _make_state(user_input="3+5等于几", latest_user_text="3+5等于几", text_input="3+5等于几")

        result = asyncio.run(node(state))

        assert result["plan_steps"] == ["计算3+5"]
        assert result["selected_act"] == "direct"
        assert result["selected_tool"] is None
        assert result["messages"][0].tool_calls == []


class TestExecuteNode:
    def test_execute_streams_and_writes_result(self) -> None:
        """ExecuteNode should call execute_plan with stream_delta_writer and write execution_result."""
        tools = AsyncMock()
        tools.execute_plan = AsyncMock(return_value="第一步：3+5=8。答案是8。")
        node = ExecuteNode(tools)
        delta_writer = AsyncMock()
        state = _make_state(
            plan_steps=["计算3+5"],
            stream_delta_writer=delta_writer,
        )

        result = asyncio.run(node(state))

        tools.execute_plan.assert_called_once()
        assert result["execution_result"] == "第一步：3+5=8。答案是8。"
        assert result["message_draft"] == "第一步：3+5=8。答案是8。"
        assert result["dialogue_stage"] == "responded"

    def test_execute_without_stream_writer(self) -> None:
        """ExecuteNode works even when stream_delta_writer is None."""
        tools = AsyncMock()
        tools.execute_plan = AsyncMock(return_value="答案是42。")
        node = ExecuteNode(tools)
        state = _make_state(
            plan_steps=["回答问题"],
            stream_delta_writer=None,
        )

        result = asyncio.run(node(state))

        assert result["execution_result"] == "答案是42。"
        assert result["message_draft"] == "答案是42。"

    def test_execute_emits_completion_event(self) -> None:
        tools = AsyncMock()
        tools.execute_plan = AsyncMock(return_value="学习完成。")
        event_writer = AsyncMock()

        asyncio.run(
            ExecuteNode(tools)(
                _make_state(
                    plan_steps=["观察", "理解"],
                    stream_event_writer=event_writer,
                )
            )
        )

        event_writer.assert_awaited_once_with(
            {
                "type": "workflow",
                "event": "workflow_completed",
                "phase": "complete",
                "status": "completed",
                "title": "学习路线已完成",
            }
        )


class TestPlanNodeFallback:
    def test_plan_fallback_on_empty_steps(self) -> None:
        """If generate_plan returns empty steps, PlanNode should fallback."""
        fake_plan = {"steps": [], "needs_retrieval": False, "retrieval_query": ""}
        tools = AsyncMock()
        tools.generate_plan = AsyncMock(return_value=fake_plan)
        node = PlanNode(tools)
        state = _make_state()

        result = asyncio.run(node(state))

        assert result["selected_act"] == "direct"
        assert len(result["plan_steps"]) >= 1  # at least fallback step


class TestSafeCalculate:
    def test_basic_arithmetic_with_brackets(self) -> None:
        result = safe_calculate("(123+456)*789/12")
        expected = str(int((123 + 456) * 789 // 12)) if (123 + 456) * 789 % 12 == 0 else str((123 + 456) * 789 / 12)
        assert result == expected

    def test_integer_result_strips_dot_zero(self) -> None:
        assert safe_calculate("6/2") == "3"
        assert safe_calculate("10-3") == "7"

    def test_division_by_zero_returns_error(self) -> None:
        result = safe_calculate("1/0")
        assert "错误" in result

    def test_empty_expression_returns_error(self) -> None:
        result = safe_calculate("")
        assert "错误" in result

    def test_disallowed_power_operator_returns_error(self) -> None:
        result = safe_calculate("2**8")
        assert "错误" in result

    def test_whitespace_expression_returns_error(self) -> None:
        result = safe_calculate("   ")
        assert "错误" in result

    def test_negative_unary(self) -> None:
        assert safe_calculate("-5+10") == "5"


class TestExecutePlanStepByStep:
    """Tests for model_service.execute_plan step-by-step logic."""

    def _make_chat_request(self, text: str = "计算3+5") -> Any:
        req = MagicMock()
        req.text = text
        req.image_base64 = None
        req.image_url = None
        req.age_hint = "3-8"
        req.mode = "education"
        return req

    def test_single_step_calls_execute_once_and_emits_delta(self) -> None:
        """Single-step plan executes once and emits its real result."""
        from app.services.model_service import ModelService

        chat_request = self._make_chat_request()
        state = _make_state(plan_steps=["计算3+5"])
        delta_writer = AsyncMock()
        mock_step = AsyncMock(return_value="3+5=8")

        svc = object.__new__(ModelService)
        svc._openai_call_semaphore = asyncio.Semaphore(1)
        svc._execute_step_with_tools = mock_step

        result = asyncio.run(svc.execute_plan(chat_request, state, on_delta=delta_writer))

        assert result == "3+5=8"
        mock_step.assert_called_once()
        call_kwargs = mock_step.call_args.kwargs
        assert call_kwargs["on_delta"] is None
        delta_writer.assert_awaited_once_with("3+5=8")

    def test_two_steps_history_passed_to_second_step(self) -> None:
        """Two-step plan: second step prompt contains first step's result."""
        chat_request = self._make_chat_request()
        state = _make_state(plan_steps=["计算123+456", "用孩子语言解释结果"])
        captured_prompts: list[str] = []

        async def fake_step(instruction: str, prompt: str, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "步骤结果"

        from app.services.model_service import ModelService
        svc = object.__new__(ModelService)
        svc._openai_call_semaphore = asyncio.Semaphore(1)
        svc._execute_step_with_tools = fake_step  # type: ignore[method-assign]

        asyncio.run(svc.execute_plan(chat_request, state))

        assert len(captured_prompts) == 2
        assert "步骤1:" in captured_prompts[1]  # history from step 1 in step 2's prompt

    def test_all_step_results_are_accumulated_and_streamed(self) -> None:
        """Each real step result is accumulated into the final streamed answer."""
        chat_request = self._make_chat_request()
        state = _make_state(plan_steps=["步骤A", "步骤B", "步骤C"])
        delta_writer = AsyncMock()
        captured_deltas: list[Any] = []

        call_index = 0

        async def fake_step(instruction: str, prompt: str, on_delta: Any = None, **kwargs: Any) -> str:
            nonlocal call_index
            captured_deltas.append(on_delta)
            call_index += 1
            return f"结果{call_index}"

        from app.services.model_service import ModelService
        svc = object.__new__(ModelService)
        svc._openai_call_semaphore = asyncio.Semaphore(1)
        svc._execute_step_with_tools = fake_step  # type: ignore[method-assign]

        result = asyncio.run(svc.execute_plan(chat_request, state, on_delta=delta_writer))

        assert captured_deltas[0] is None
        assert captured_deltas[1] is None
        assert captured_deltas[2] is None
        streamed = "".join(call.args[0] for call in delta_writer.await_args_list)
        assert streamed == result
        assert "第1步 · 步骤A\n结果1" in result
        assert "第3步 · 步骤C\n结果3" in result

    def test_each_step_emits_active_and_completed_events(self) -> None:
        chat_request = self._make_chat_request()
        state = _make_state(plan_steps=["观察现象", "总结原因"])
        event_writer = AsyncMock()
        state["stream_event_writer"] = event_writer

        async def fake_step(**kwargs: Any) -> str:
            return "步骤结果"

        from app.services.model_service import ModelService
        svc = object.__new__(ModelService)
        svc._openai_call_semaphore = asyncio.Semaphore(1)
        svc._execute_step_with_tools = fake_step  # type: ignore[method-assign]

        asyncio.run(svc.execute_plan(chat_request, state))

        events = [call.args[0] for call in event_writer.await_args_list]
        assert [(event["status"], event["step_index"]) for event in events] == [
            ("active", 0),
            ("completed", 0),
            ("active", 1),
            ("completed", 1),
        ]
        assert all(event["phase"] == "execution" for event in events)
        assert [event["event"] for event in events] == [
            "step_started",
            "step_completed",
            "step_started",
            "step_completed",
        ]
        assert events[0]["title"] == "观察现象"
        assert events[0]["step_count"] == 2

    def test_empty_plan_steps_uses_single_fallback(self) -> None:
        """Empty plan_steps falls back to one default step."""
        chat_request = self._make_chat_request()
        state = _make_state(plan_steps=[])
        call_count = 0

        async def fake_step(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            return "fallback"

        from app.services.model_service import ModelService
        svc = object.__new__(ModelService)
        svc._openai_call_semaphore = asyncio.Semaphore(1)
        svc._execute_step_with_tools = fake_step  # type: ignore[method-assign]

        result = asyncio.run(svc.execute_plan(chat_request, state))

        assert call_count == 1
        assert result == "fallback"
