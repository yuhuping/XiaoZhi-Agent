from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState, append_trace, utc_now_iso
from app.tools.basic_tools import BasicTools, ToolCall

logger = logging.getLogger(__name__)


class MemoryUpdateNode:
    """记忆更新：每轮写working，并只写1条长期episodic拼接记忆。"""

    def __init__(self, tools: BasicTools) -> None:
        self.tools = tools

    async def __call__(self, state: AgentState) -> AgentState:
        """执行记忆写入。"""
        logger.info("entering node=memory_update")

        current_topic = self._normalize_topic(
            state.get("current_topic"),
            state.get("perception_signals", []),
            state.get("selected_act"),
        )
        interaction_mode = state.get("interaction_mode", "education")
        user_text = state.get("latest_user_text") or "[image input]"
        assistant_text = state.get("message_draft") or "Let us learn one small thing together."
        follow_up_question = state.get("follow_up_question")
        ts = utc_now_iso()

        # 重要变量：每轮都写长期记忆，按用户决策执行。
        write_success = True
        written_types: list[str] = []
        settings = self.tools.model_service.settings

        turns = [
            {
                "role": "user",
                "text": user_text,
                "topic": current_topic,
                "asked_question": None,
                "mode": interaction_mode,
                "timestamp": ts,
            },
            {
                "role": "assistant",
                "text": assistant_text,
                "topic": current_topic,
                "asked_question": follow_up_question,
                "mode": interaction_mode,
                "timestamp": ts,
            },
        ]

        # 1) 工作记忆：完整保存本轮turn
        for turn in turns:
            result = self._add_memory(state=state, content=turn["text"], memory_type="working", importance=0.7, metadata={"turn": turn, **turn})
            write_success = write_success and result
        written_types.append("working")

        # 2) 长期记忆：每轮只写1条拼接后的episodic，避免重复写入。
        merged_text = f"user: {user_text}\nassistant: {assistant_text}"
        merged_turn = {
            "role": "dialogue_pair",
            "text": merged_text,
            "topic": current_topic,
            "asked_question": follow_up_question,
            "mode": interaction_mode,
            "timestamp": ts,
            "user_text": user_text,
            "assistant_text": assistant_text,
        }
        episodic_ok = self._add_memory(
            state=state,
            content=merged_text,
            memory_type="episodic",
            importance=0.72,
            metadata={"turn": merged_turn, **merged_turn},
        )
        write_success = write_success and episodic_ok
        written_types.append("episodic")

        # 3) 感知长期记忆：默认关闭，仅保留备用路径。
        enable_perceptual = bool(getattr(settings, "memory_write_perceptual_enabled", False))
        if enable_perceptual and (state.get("image_base64") or state.get("image_url")):
            perceptual_ok = self._add_memory(
                state=state,
                content=state.get("latest_user_text") or "[image input]",
                memory_type="perceptual",
                importance=0.65,
                metadata={
                    "topic": current_topic,
                    "mode": interaction_mode,
                    "modality": "image",
                    "timestamp": ts,
                },
            )
            write_success = write_success and perceptual_ok
            written_types.append("perceptual")

        # 4) 生命周期管理：默认关闭自动consolidate，保留备用逻辑。
        consolidated_count = 0
        forgotten_count = 0
        enable_consolidate = bool(getattr(settings, "memory_auto_consolidate_enabled", False))
        if enable_consolidate:
            consolidate1 = self.tools.run_tool(
                ToolCall(
                    name="memory_execute",
                    args={
                        "action": "consolidate",
                        "kwargs": {
                            "user_id": state["profile_id"],
                            "from_type": "working",
                            "to_type": "episodic",
                            "importance_threshold": settings.memory_consolidate_working_threshold,
                        },
                    },
                ),
                state,
            )
            if consolidate1.success:
                consolidated_count += int(consolidate1.data.get("consolidated", 0))

            consolidate2 = self.tools.run_tool(
                ToolCall(
                    name="memory_execute",
                    args={
                        "action": "consolidate",
                        "kwargs": {
                            "user_id": state["profile_id"],
                            "from_type": "episodic",
                            "to_type": "semantic",
                            "importance_threshold": settings.memory_consolidate_episodic_threshold,
                        },
                    },
                ),
                state,
            )
            if consolidate2.success:
                consolidated_count += int(consolidate2.data.get("consolidated", 0))

        forget = self.tools.run_tool(
            ToolCall(
                name="memory_execute",
                args={
                    "action": "forget",
                    "kwargs": {
                        "user_id": state["profile_id"],
                        "strategy": "time_based",
                        "max_age_days": settings.memory_forget_max_age_days,
                        "threshold": 0.2,
                    },
                },
            ),
            state,
        )
        if forget.success:
            forgotten_count += int(forget.data.get("forgotten", 0))

        return {
            "memory_session_updated": write_success,
            "memory_profile_updated": write_success,
            "memory_written_types": written_types,
            "memory_consolidated_count": consolidated_count,
            "memory_forgotten_count": forgotten_count,
            "dialogue_stage": "responded",
            "workflow_trace": append_trace(state, "memory_update"),
        }

    def _add_memory(
        self,
        state: AgentState,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> bool:
        """通过memory_execute写入单条记忆。"""
        result = self.tools.run_tool(
            ToolCall(
                name="memory_execute",
                args={
                    "action": "add",
                    "kwargs": {
                        "user_id": state["profile_id"],
                        "session_id": state["session_id"],
                        "content": content,
                        "memory_type": memory_type,
                        "importance": importance,
                        "metadata": metadata,
                    },
                },
            ),
            state,
        )
        return bool(result.success)

    def _normalize_topic(self, topic: object, signals: object, selected_act: object) -> str | None:
        """规范化topic字段。"""
        signal_values = signals if isinstance(signals, list) else []
        if "topic_candidate" not in signal_values and selected_act != "read_memory":
            return None
        if not isinstance(topic, str):
            return None
        normalized = topic.strip()
        if not normalized:
            return None
        if normalized.lower() in {"unknown", "picture"}:
            return None
        return normalized
