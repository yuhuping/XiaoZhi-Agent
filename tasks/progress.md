# 本文件记录：
- 今天做了什么
- 下一个 session 启动时先看什么

---

# v1.4

## 2026-03-31
- 今天做了什么：完成 P0 `v1.4-sse-chat-streaming`（后端 `/api/v1/chat/explain-and-ask` 改为 `text/event-stream`，SSE 事件为 `delta` + `done`；前端改为流式解析并将多段文本拼接为同一条 assistant 消息；新增 `tests/test_chat_sse_streaming.py` 覆盖流式头、增量顺序、单次结束信号与尾段一致性）。
- 下一个 session 启动时先看什么：确认 `tasks/backlog.json` 的 P1 `v1.4-original-image-pass-through` 是否启动（本次未实现）。

## 2026-03-31（修正）
- 今天做了什么：将"后置切片伪流式"修正为"模型生成阶段真实流式"——API 改为任务并发 + 队列转发；`ModelService.generate_final_response` 新增 `on_delta` 并走 `stream=True` 回调；SSE 事件改为仅 `delta` + `done`（`done` 不再携带 `response`）；前端改为字符队列逐字渲染并在 `done` 后收尾。
- 下一个 session 启动时先看什么：浏览器手工验证首字到达时间（确认首字在模型完成前出现），以及图文请求下流式体验是否稳定。

---

# v1.5

## 2026-04-07
- 今天做了什么：完成 P0 `v1.5-rag-multi-query`（RAG 多查询扩展）。发现 `act_retrieve.py` 未连接到 AgentGraph DAG，实际检索路径为 LangGraph `ToolNode` → `BasicTools._langgraph_retrieve_knowledge`。因此将多查询逻辑实现在正确位置：`ModelService` 新增同步方法 `expand_queries_sync`（使用 sync OpenAI client），`BasicTools._langgraph_retrieve_knowledge` 改写为先调扩展再各自检索、按 `chunk_id` 去重、score 降序合并取 top-k；tool_result 新增 `expanded_queries` 字段。`act_retrieve.py` 保留但加注释说明未连接。所有测试（4 passed）通过。
- 下一个 session 启动时先看什么：确认 `tasks/backlog.json` 的 P0 `v1.5-parent-summary-tool`（家长侧摘要生成工具）是否启动。

## 2026-04-08
- 今天做了什么：完成 P0 `v1.5-parent-summary-tool`，同时建立 **Skill 插件架构**。新增 `app/skills/` 目录体系（`SkillRegistry` 自动扫描 `*/skill.yaml`，按请求 mode 过滤注入 `bind_tools`）。`generate_parent_summary` skill 作为首个 skill 实现：`_execute()` 纯 SQLite 读取（无 LLM 阻塞），respond 节点凭 skill 专属 prompt 流式生成结构化中文摘要；支持 `child_name` 自然语言模糊匹配 + fallback `default_child`。新增 `ActType="skill"`，`observe` / `model_service` 各增一个分支处理 skill 路径。`SQLiteMemoryStore` 新增 `list_distinct_user_ids()`。所有测试 4 passed。
- 下一个 session 启动时先看什么：`tasks/backlog.json` todo 列表已清空，v1.5 全部完成。可开始规划 v1.6 或手工验证 parent 模式摘要流式输出体验。

---

# v1.6

## Graph 节点示意图

```
┌─────────────────────── Top-level Router Graph ───────────────────────┐
│                                                                      │
│  START → understand → state_update ─┬─ route_by_mode ──┐            │
│                                     │                   │            │
│         ┌───────────────────────────┘                   │            │
│         │ education                          parent /   │            │
│         ▼                                   companion   │            │
│  ┌─ PlanExecute Subgraph ──────┐      ┌─ ReAct Subgraph ─────────┐ │
│  │                              │      │                           │ │
│  │  plan ─┬─ plan_route ──┐    │      │  reason ─┬─ tools_cond ─┐│ │
│  │        │               │    │      │    ▲     │              ││ │
│  │        │ retrieve      │    │      │    │     │ has_tool     ││ │
│  │        ▼               │    │      │    │     ▼              ││ │
│  │     tools → observe    │    │      │    │   tools → observe  ││ │
│  │              │         │    │      │    │            │        ││ │
│  │              │ direct  │    │      │    │ should_    │        ││ │
│  │              ▼         ▼    │      │    │ continue   ▼        ││ │
│  │           execute ──► END   │      │    └── react ◄─┘        ││ │
│  │                              │      │                  │ direct││ │
│  └──────────────────────────────┘      │                  ▼      ││ │
│         │                              │   respond ──► END  ◄───┘│ │
│         │                              │                          │ │
│         │                              └──────────────────────────┘ │
│         │                                       │                    │
│         └──────────────┬────────────────────────┘                    │
│                        ▼                                             │
│                  memory_update → response → memory_compact → END     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

PlanExecute 路径 (education):
  understand → state_update → plan → [tools → observe →] execute → memory_update → response → memory_compact → END
  LLM 调用: plan(1次) + execute(1次) = 最多 2 次

ReAct 路径 (parent/companion):
  understand → state_update → reason ⇄ tools → observe (循环 max 3 轮) → respond → memory_update → response → memory_compact → END
  LLM 调用: reason(每轮1次) + respond(1次) = 最多 4 次
```

## 2026-04-10
- 今天做了什么：完成 v1.6 双框架重构。顶层 Router Graph 按 `interaction_mode` 分发：education → PlanExecute 子图（PlanNode 生成步骤 + 可选 RAG 检索 + ExecuteNode 流式输出解题过程），parent/companion → ReAct 子图（多轮 reason→tools→observe 循环，max_iterations=3）。清理 AgentState 无用字段（`normalized_text`、`detected_object`、`source_mode`），新增 `plan_steps`、`execution_result`、`react_iteration`、`react_history` 等 6 个字段。新增 3 个测试文件覆盖子图路由、Plan 生成/降级、ReAct 迭代终止。全部 18 个测试通过。
- 下一个 session 启动时先看什么：手工验证 education 模式数学题的流式解题体验，以及 parent 模式多轮搜索是否正常工作。
