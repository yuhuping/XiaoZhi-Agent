from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import yaml
from langchain_core.tools import StructuredTool

from app.skills.base import BaseSkill, SkillMeta

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Skill 插件注册表：扫描约定目录，按 mode 过滤提供工具列表。"""

    def __init__(self, skills_dir: str, deps: dict[str, Any]) -> None:
        self._skills: dict[str, BaseSkill] = {}  # key: tool_name
        self._scan(skills_dir, deps)

    def _scan(self, skills_dir: str, deps: dict[str, Any]) -> None:
        """扫描 skills_dir/*/skill.yaml，调用各 handler.py 的 create_skill(meta, deps) 工厂。"""
        base = Path(skills_dir)
        if not base.is_dir():
            logger.warning("skills_dir=%s not found, no skills loaded", skills_dir)
            return
        for yaml_path in sorted(base.glob("*/skill.yaml")):
            try:
                self._load_skill(yaml_path, deps)
            except Exception:
                logger.exception("failed to load skill from %s", yaml_path)

    def _load_skill(self, yaml_path: Path, deps: dict[str, Any]) -> None:
        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        meta = SkillMeta(
            name=str(raw["name"]),
            display_name=str(raw.get("display_name", raw["name"])),
            description=str(raw["description"]).strip(),
            version=str(raw.get("version", "1.0")),
            modes=list(raw.get("modes", [])),
            tool_name=str(raw["tool_name"]),
        )
        # 约定：handler 模块路径由 yaml 文件父目录推断
        # e.g. app/skills/parent_summary/skill.yaml → app.skills.parent_summary.handler
        skill_dir = yaml_path.parent
        # 将路径转为 Python 模块名（e.g. app/skills/parent_summary → app.skills.parent_summary）
        parts = skill_dir.parts
        module_name = ".".join(parts) + ".handler"
        module = importlib.import_module(module_name)
        skill: BaseSkill = module.create_skill(meta, deps)
        self._skills[meta.tool_name] = skill
        logger.info("loaded skill name=%s tool=%s modes=%s", meta.name, meta.tool_name, meta.modes)

    def get_tools(self, mode: str | None = None) -> list[StructuredTool]:
        """按 mode 过滤返回 StructuredTool 列表，供 bind_tools 使用。"""
        result = []
        for skill in self._skills.values():
            if mode is None or mode in skill.meta.modes:
                result.append(skill.as_tool())
        return result

    def get_all_tools(self) -> list[StructuredTool]:
        """返回全部 skill tools（不按 mode 过滤），供 ToolNode 注册。"""
        return [skill.as_tool() for skill in self._skills.values()]

    def find_skill_by_tool_name(self, tool_name: str | None) -> BaseSkill | None:
        """根据 tool_name 查找 skill，供 observe 节点和 model_service 使用。"""
        if not tool_name:
            return None
        return self._skills.get(tool_name)
