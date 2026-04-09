# Parent Summary Skill 设计文档

**日期：** 2026-04-08
**Backlog ID：** v1.5-parent-summary-tool
**范围：** 家长侧摘要生成 + Skill 插件架构

---

## 背景

实现 backlog 任务 `v1.5-parent-summary-tool`：当家长请求孩子学习摘要时，agent 调用工具读取记忆、respond 节点流式生成结构化中文报告。

同时，为项目建立可扩展的 **Skill 插件架构**（约定目录 + yaml 自动发现），使未来新能力可以独立添加，无需修改核心框架文件。

---

## 架构决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| Skill 发现机制 | 约定目录 + `skill.yaml` 自动扫描 | 插件感强，展示架构设计能力 |
| 与 DAG 集成方式 | 注册为 LangGraph StructuredTool | DAG 结构固定约束，复用现有路径 |
| Mode 过滤 | bind_tools 阶段按 mode 过滤 | 每次请求动态生效，模式切换立即识别 |
| 多孩子支持 | `child_name: str` 自然语言匹配，失败 fallback `default_child` | 简单自然，无需前端额外传参 |
| LLM 调用位置 | Skill 只做数据获取，respond 节点流式生成 | 避免 skill 内 LLM 阻塞，SSE 体验与现有路径一致 |
| ActType | 新增 `"skill"` | workflow_trace 可见，observe/respond 分支清晰 |

---

## 目录结构

### 新增文件

```
app/skills/
├── __init__.py                    # 导出 SkillRegistry
├── registry.py                    # SkillRegistry：扫描、加载、按 mode 过滤
├── base.py                        # BaseSkill 抽象基类
└── parent_summary/
    ├── skill.yaml                 # 元数据声明
    ├── __init__.py
    ├── handler.py                 # ParentSummarySkill 实现
    └── prompts.py                 # respond 节点使用的 summary prompt 模板
```

### 修改文件

| 文件 | 改动 |
|---|---|
| `app/schemas/chat.py` | `ActType` 新增 `"skill"` |
| `app/tools/basic_tools.py` | 新增 `skill_registry` 参数；`as_langgraph_tools(mode)` 按 mode 追加 skill tools；新增 `as_all_langgraph_tools()` |
| `app/agent/nodes/reason.py` | `reason_next_action` 传 mode 给 `as_langgraph_tools` |
| `app/services/model_service.py` | `_select_act_from_tool` 识别 skill；`_stream_final_response_text` 支持 skill 专属 prompt；注入 `skill_registry` |
| `app/agent/nodes/observe.py` | 新增 `selected_act == "skill"` 分支，委托 `skill.observe_result()` |
| `app/agent/graph.py` | `ToolNode` 用 `as_all_langgraph_tools()`；`ObserveNode` 注入 `skill_registry` |

### 不变

- LangGraph DAG 结构
- `AgentState` TypedDict 字段
- education / companion 模式行为
- 前端

---

## Skill 框架核心接口

### `app/skills/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.tools import StructuredTool
from app.agent.state import AgentState

@dataclass(frozen=True)
class SkillMeta:
    name: str
    display_name: str
    description: str
    version: str
    modes: list[str]      # 允许调用该 skill 的模式列表
    tool_name: str        # 注册到 LangGraph 的工具名

class BaseSkill(ABC):
    meta: SkillMeta

    @abstractmethod
    def as_tool(self) -> StructuredTool:
        """返回 LangGraph StructuredTool 定义。"""

    @abstractmethod
    def observe_result(self, raw: dict) -> dict:
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
```

### `app/skills/registry.py`

```python
class SkillRegistry:
    def __init__(self, skills_dir: str, deps: dict[str, Any]):
        """
        deps: 依赖容器，传给各 skill 的 create_skill(meta, deps) 工厂函数。
        例如: {"memory_manager": memory_manager_instance}
        """
        self._skills: dict[str, BaseSkill] = {}   # key: tool_name
        self._scan(skills_dir, deps)

    def _scan(self, skills_dir: str, deps: dict[str, Any]) -> None:
        """扫描 skills_dir/*/skill.yaml，调用各 handler.py 的 create_skill(meta, deps) 工厂。"""

    def get_tools(self, mode: str | None = None) -> list[StructuredTool]:
        """按 mode 过滤返回 StructuredTool 列表，供 bind_tools 使用。"""

    def get_all_tools(self) -> list[StructuredTool]:
        """返回全部 skill tools，供 ToolNode 注册使用。"""

    def find_skill_by_tool_name(self, tool_name: str | None) -> BaseSkill | None:
        """供 observe 节点和 model_service 查找 skill。"""
```

每个 `handler.py` 导出一个工厂函数（统一接口，无需 registry 了解每个 skill 的具体依赖）：

```python
# app/skills/parent_summary/handler.py
def create_skill(meta: SkillMeta, deps: dict[str, Any]) -> BaseSkill:
    return ParentSummarySkill(meta=meta, memory_manager=deps["memory_manager"])
```

---

## ParentSummarySkill 实现

### `app/skills/parent_summary/skill.yaml`

```yaml
name: parent_summary
display_name: 家长侧摘要生成
description: >
  Generate a summary report of a specific child's recent learning sessions
  for the parent, based on stored memory. Use this when the parent asks
  for a child activity or learning summary.
version: "1.0"
modes: [parent]
tool_name: generate_parent_summary
```

### Tool Schema

```python
class ParentSummaryInput(BaseModel):
    child_name: str = Field(
        default="default_child",
        description="The name or profile ID of the child to summarize. "
                    "Use the child's name as mentioned by the parent, "
                    "or 'default_child' if not specified.",
    )
```

### `app/skills/parent_summary/handler.py` — 核心逻辑

```python
class ParentSummarySkill(BaseSkill):
    def __init__(self, meta: SkillMeta, memory_manager: MemoryManager):
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
        episodic = self.memory_manager.store.list_items(
            user_id=profile_id, memory_type="episodic", limit=10_000
        )
        semantic = self.memory_manager.store.list_items(
            user_id=profile_id, memory_type="semantic", limit=10_000
        )
        if not episodic and not semantic:
            return json.dumps({
                "success": True,
                "has_memory": False,
                "memory_texts": [],
                "child_profile_id": profile_id,
            }, ensure_ascii=False)
        memory_texts = (
            [item.content[:500] for item in episodic[:30]] +
            [item.content[:500] for item in semantic[:30]]
        )
        return json.dumps({
            "success": True,
            "has_memory": True,
            "memory_texts": memory_texts,
            "child_profile_id": profile_id,
        }, ensure_ascii=False)

    def _resolve_profile_id(self, child_name: str) -> str:
        """模糊匹配孩子名 → profile_id，失败返回 default_child。"""
        name = (child_name or "").strip()
        if not name or name == "default_child":
            return "default_child"
        all_user_ids = self.memory_manager.store.list_distinct_user_ids()
        for uid in all_user_ids:
            if name in uid or uid in name:
                return uid
        return "default_child"

    def observe_result(self, raw: dict) -> dict:
        has_memory = raw.get("has_memory", False)
        count = len(raw.get("memory_texts", []))
        child = raw.get("child_profile_id", "default_child")
        return {
            "observation_summary": (
                f"fetched {count} memory records for child={child}"
                if has_memory else f"no memory records found for child={child}"
            ),
            "tool_result": raw,
            "tool_success": raw.get("success", False),
        }

    def get_response_instruction(self) -> str:
        from app.skills.parent_summary.prompts import PARENT_SUMMARY_INSTRUCTION
        return PARENT_SUMMARY_INSTRUCTION

    def get_response_user_prompt(self, state: AgentState) -> str:
        from app.skills.parent_summary.prompts import build_parent_summary_user_prompt
        tool_result = state.get("tool_result", {})
        has_memory = tool_result.get("has_memory", False)
        memory_texts = tool_result.get("memory_texts", [])
        child_id = tool_result.get("child_profile_id", "default_child")
        return build_parent_summary_user_prompt(
            has_memory=has_memory,
            memory_texts=memory_texts,
            child_profile_id=child_id,
        )
```

### `app/skills/parent_summary/prompts.py`

```python
PARENT_SUMMARY_INSTRUCTION = (
    "你是一位专业的儿童发展助手，正在为家长撰写孩子的学习报告。"
    "用中文生成结构化摘要，语气专业、友好。"
    "Output plain text only. Do not output JSON, markdown, or code fences."
)

def build_parent_summary_user_prompt(
    has_memory: bool,
    memory_texts: list[str],
    child_profile_id: str,
) -> str:
    if not has_memory:
        return f"孩子（{child_profile_id}）目前还没有学习记录，请告知家长暂时无法生成摘要。"
    joined = "\n".join(f"- {t}" for t in memory_texts)
    return (
        f"以下是孩子（{child_profile_id}）与小智互动的记忆记录：\n\n"
        f"{joined}\n\n"
        "请生成结构化摘要，包含：\n"
        "1. 近期感兴趣的话题\n"
        "2. 常问的问题类型\n"
        "3. 学习观察与建议"
    )
```

---

## 集成改动详情

### `app/schemas/chat.py`

```python
ActType = Literal["direct", "retrieve_knowledge", "read_memory", "skill"]
```

### `app/tools/basic_tools.py`

```python
def __init__(self, model_service, memory_tool, retriever, skill_registry=None):
    ...
    self.skill_registry = skill_registry
    self._langgraph_tools = self._build_langgraph_tools()

def as_langgraph_tools(self, mode: str | None = None) -> list[StructuredTool]:
    """基础工具 + 当前 mode 匹配的 skill tools，供 bind_tools 使用。"""
    base = list(self._langgraph_tools)
    if self.skill_registry:
        base.extend(self.skill_registry.get_tools(mode=mode))
    return base

def as_all_langgraph_tools(self) -> list[StructuredTool]:
    """全量工具列表（不按 mode 过滤），供 ToolNode 注册。"""
    base = list(self._langgraph_tools)
    if self.skill_registry:
        base.extend(self.skill_registry.get_all_tools())
    return base

async def reason_next_action(self, state: AgentState) -> ReasonDecision:
    request = state_to_request(state)
    mode = state.get("interaction_mode", "education")
    tools = self.as_langgraph_tools(mode=mode)   # ← 按 mode 过滤
    return await self.model_service.reason_next_action(request, state, tools=tools)
```

### `app/services/model_service.py`

```python
# 构造函数新增 skill_registry
def __init__(self, settings: Settings, skill_registry=None):
    ...
    self.skill_registry = skill_registry

def _select_act_from_tool(self, tool_name: str | None) -> ActType:
    if tool_name in {"retrieve_knowledge", "tavily_search"}:
        return "retrieve_knowledge"
    if tool_name == "read_memory_bundle":
        return "read_memory"
    if tool_name is not None:
        return "skill"    # 非内置 tool → skill
    return "direct"

async def _stream_final_response_text(self, chat_request, state, on_delta):
    # 新增：skill 专属 prompt 路径
    if state.get("selected_act") == "skill" and self.skill_registry:
        skill = self.skill_registry.find_skill_by_tool_name(state.get("selected_tool"))
        if skill:
            instruction = skill.get_response_instruction() or build_response_instruction(chat_request.mode)
            prompt = skill.get_response_user_prompt(state) or build_response_user_prompt(chat_request, state, include_json_contract=False)
        else:
            instruction = build_response_instruction(chat_request.mode)
            prompt = build_response_user_prompt(chat_request, state, include_json_contract=False)
    else:
        instruction = build_response_instruction(chat_request.mode)
        prompt = build_response_user_prompt(chat_request, state, include_json_contract=False)
    # 后续 astream 逻辑不变...
```

### `app/agent/nodes/observe.py`

```python
class ObserveNode:
    def __init__(self, skill_registry=None):
        self.skill_registry = skill_registry

    async def __call__(self, state):
        ...
        elif selected_act == "skill":
            skill = self.skill_registry.find_skill_by_tool_name(
                state.get("selected_tool")
            ) if self.skill_registry else None
            if skill:
                skill_updates = skill.observe_result(last_tool_payload)
            else:
                skill_updates = {
                    "observation_summary": "skill executed",
                    "tool_result": last_tool_payload,
                    "tool_success": True,
                }
            updates.update(skill_updates)
        ...
```

### `app/agent/graph.py`

```python
def __init__(self, tools: BasicTools, skill_registry=None):
    ...
    builder.add_node("tools", ToolNode(
        tools=tools.as_all_langgraph_tools(),   # 全量注册，不过滤
        messages_key="messages"
    ))
    builder.add_node("observe", ObserveNode(skill_registry=skill_registry))
```

---

## 完整数据流

### parent 模式（有记忆）

```
家长: "帮我总结一下小明最近的学习情况"

chatbot
  → as_langgraph_tools(mode="parent")
  → [retrieve_knowledge, tavily_search, read_memory_bundle, generate_parent_summary]
  → LLM 选择 generate_parent_summary(child_name="小明")
  → selected_act="skill"

ToolNode
  → ParentSummarySkill._execute(child_name="小明")
  → _resolve_profile_id("小明") → 匹配或 fallback "default_child"
  → 读取 SQLite episodic + semantic（无 LLM，极快）
  → 返回 {success, has_memory=true, memory_texts=[...], child_profile_id}

observe
  → skill.observe_result(raw)
  → tool_result = {memory_texts: [...], ...}
  → observation_summary = "fetched N memory records for child=..."

respond
  → selected_act == "skill" → 查找 skill
  → 使用 ParentSummarySkill 的 instruction + user_prompt
  → _stream_final_response_text → SSE 流式输出结构化摘要 ✅
```

### parent 模式（无记忆）

```
ToolNode 返回 {has_memory: false, memory_texts: []}
→ observe: "no memory records found for child=..."
→ respond: 使用 skill prompt，生成友好提示"暂无学习记录"
```

### education 模式（切换后）

```
chatbot
  → as_langgraph_tools(mode="education")
  → [retrieve_knowledge, tavily_search, read_memory_bundle]  ← 无 generate_parent_summary
  → LLM 看不到摘要工具
```

---

## `SQLiteMemoryStore` 新增方法

`_resolve_profile_id` 需要查询所有 distinct user_id：

```python
def list_distinct_user_ids(self) -> list[str]:
    """返回 memory_items 表中所有不重复的 user_id。"""
```

---

## 验收标准

- [ ] `generate_parent_summary` 出现在 parent 模式的 bind_tools 列表，不出现在 education/companion
- [ ] 前端切换 education → parent → education，每次请求工具列表正确过滤
- [ ] SQLite 有儿童记忆时，respond 流式生成结构化中文摘要（含三个部分）
- [ ] SQLite 无儿童记忆时，respond 流式生成友好提示
- [ ] `child_name` 匹配失败时自动 fallback `default_child`
- [ ] `workflow_trace` 中包含 `"skill"` 标记
- [ ] education / companion 模式行为完全不受影响
- [ ] `app/skills/` 目录下新增第二个 skill 只需添加目录 + `skill.yaml` + `handler.py`，无需修改框架文件

---

## 初始化与依赖注入

`SkillRegistry` 在应用启动时创建（`app/main.py` 或服务工厂），持有所有服务依赖，然后传给 `ModelService` 和 `AgentGraph`：

```python
# app/main.py（伪代码，具体位置视现有启动结构）
memory_manager = MemoryManager(config)
skill_registry = SkillRegistry(
    skills_dir="app/skills",
    deps={"memory_manager": memory_manager},
)
model_service = ModelService(settings=settings, skill_registry=skill_registry)
basic_tools = BasicTools(
    model_service=model_service,
    memory_tool=memory_tool,
    retriever=retriever,
    skill_registry=skill_registry,
)
agent_graph = AgentGraph(tools=basic_tools, skill_registry=skill_registry)
```

`skill_registry` 同一个实例共享给三处：`BasicTools`（as_langgraph_tools 过滤）、`ModelService`（respond prompt 分发）、`AgentGraph`（ObserveNode 注入）。

---

## 不在本次范围内

- 前端 UI 新增摘要按钮（当前靠 LLM 自动识别用户意图触发）
- 多孩子 profile 管理 API
- skill 热加载（重启后生效即可）
