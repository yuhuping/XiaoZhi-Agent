from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool

from app.agent.state import AgentState


@dataclass(frozen=True)
class SkillMeta:
    name: str
    display_name: str
    description: str
    version: str
    modes: list[str]
    tool_name: str


class BaseSkill(ABC):
    meta: SkillMeta

    @abstractmethod
    def as_tool(self) -> StructuredTool:
        """返回 LangGraph StructuredTool 定义。"""

    @abstractmethod
    def observe_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        """解读 tool 返回值，返回需更新的 state 字段。
        必须包含: observation_summary, tool_result, tool_success
        """

    def get_response_instruction(self) -> str | None:
        """返回 respond 节点使用的专属 instruction。
        None 表示使用默认 build_response_instruction。
        """
        return None

    def get_response_user_prompt(self, state: AgentState) -> str | None:
        """返回 respond 节点使用的专属 user prompt。
        None 表示使用默认 build_response_user_prompt。
        """
        return None
