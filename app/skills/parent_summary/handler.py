from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.memory.manager import MemoryManager
from app.skills.base import BaseSkill, SkillMeta
from app.skills.parent_summary.prompts import PARENT_SUMMARY_INSTRUCTION, build_parent_summary_user_prompt

logger = logging.getLogger(__name__)


class ParentSummaryInput(BaseModel):
    child_name: str = Field(
        default="default_child",
        description=(
            "The child's name extracted from the parent's message. "
            "For example, if the parent says '帮我看看小明的情况', set this to '小明'. "
            "If no child name is mentioned, use 'default_child'."
        ),
    )


class ParentSummarySkill(BaseSkill):
    def __init__(self, meta: SkillMeta, memory_manager: MemoryManager) -> None:
        self.meta = meta
        self.memory_manager = memory_manager

    def as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._execute,
            name=self.meta.tool_name,
            description=self.meta.description,
            args_schema=ParentSummaryInput,
        )

    def _execute(self, child_name: str = "default_child") -> str:
        """数据获取层：读取记忆，无 LLM 调用。"""

        profile_id = self._resolve_profile_id(child_name)
        logger.info("[parent_summary] search profile_id=%r (requested=%r)", profile_id, child_name)
        episodic = self.memory_manager.store.list_items(
            user_id=profile_id, memory_type="episodic", limit=10_000
        )
        semantic = self.memory_manager.store.list_items(
            user_id=profile_id, memory_type="semantic", limit=10_000
        )

        if not episodic and not semantic:
            logger.info("[parent_summary] no memory found, returning empty result")
            return json.dumps(
                {
                    "success": True,
                    "has_memory": False,
                    "memory_texts": [],
                    "child_profile_id": profile_id,
                    "requested_name": child_name,
                },
                ensure_ascii=False,
            )
        memory_texts = [item.content[:500] for item in episodic[:30]] + [
            item.content[:500] for item in semantic[:30]
        ]
        logger.info("[parent_summary] returning %d memory_texts for respond prompt", len(memory_texts))
        return json.dumps(
            {
                "success": True,
                "has_memory": True,
                "memory_texts": memory_texts,
                "child_profile_id": profile_id,
                "requested_name": child_name,
            },
            ensure_ascii=False,
        )

    def _resolve_profile_id(self, child_name: str) -> str:
        """模糊匹配孩子名 → profile_id，失败返回 default_child。"""
        name = (child_name or "").strip()
        if not name or name == "default_child":
            return "default_child"
        all_user_ids = self.memory_manager.store.list_distinct_user_ids()
        logger.info("[parent_summary] all user_ids in sqlite: %r", all_user_ids)
        for uid in all_user_ids:
            if name in uid or uid in name:
                logger.info("[parent_summary] matched child_name=%r → profile_id=%r", name, uid)
                return uid
        logger.info("[parent_summary] no match for child_name=%r, fallback to default_child", name)
        return "default_child"

    def observe_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        has_memory = raw.get("has_memory", False)
        count = len(raw.get("memory_texts", []))
        child = raw.get("child_profile_id", "default_child")
        summary = (
            f"fetched {count} memory records for child={child}"
            if has_memory
            else f"no memory records found for child={child}"
        )
        return {
            "observation_summary": summary,
            "tool_result": raw,
            "tool_success": bool(raw.get("success", False)),
        }

    def get_response_instruction(self) -> str:
        return PARENT_SUMMARY_INSTRUCTION

    def get_response_user_prompt(self, state: AgentState) -> str:
        tool_result = state.get("tool_result", {})
        # 优先用家长原始输入的名字展示，fallback 到 profile_id
        display_name = str(
            tool_result.get("requested_name") or tool_result.get("child_profile_id") or "default_child"
        )
        return build_parent_summary_user_prompt(
            has_memory=bool(tool_result.get("has_memory", False)),
            memory_texts=list(tool_result.get("memory_texts", [])),
            child_profile_id=display_name,
        )


def create_skill(meta: SkillMeta, deps: dict[str, Any]) -> BaseSkill:
    return ParentSummarySkill(meta=meta, memory_manager=deps["memory_manager"])
