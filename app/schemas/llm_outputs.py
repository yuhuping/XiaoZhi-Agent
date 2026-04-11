from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReactResponse(BaseModel):
    """respond 节点（ReAct 子图）的 LLM 结构化输出。"""

    topic: str = ""
    message: str = "Let us learn one small thing together."
    follow_up_question: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"
    safety_notes: str = ""


class PlanResult(BaseModel):
    """plan 节点（PlanExecute 子图）的 LLM 结构化输出。"""

    steps: list[str] = Field(default_factory=lambda: ["直接回答用户问题"])
    needs_retrieval: bool = False
    retrieval_query: str = ""


class EpisodicSummary(BaseModel):
    """memory_compact 节点压缩情景记忆批次的 LLM 结构化输出。"""

    summary: str = ""
    topic_hint: str = ""
    key_points: list[str] = Field(default_factory=list)


class TopicSummary(BaseModel):
    """memory_compact 节点总结话题历史的 LLM 结构化输出。"""

    summary: str = ""
