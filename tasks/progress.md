# 本文件记录：
- 今天做了什么
- 下一个 session 启动时先看什么

---

# v1.7

## 2026-05-04
- 今天做了什么：完成 P1 `v1.7-plan-execute-step-by-step`（Execute 节点真正的 Plan-and-Execute 逐步执行）。核心改动：① 新建 `app/tools/calculate.py`，实现安全 AST eval 四则运算工具（无项目内部导入，避免 basic_tools→model_service 循环依赖）；② `plan_prompts.py` 新增 `build_step_execute_instruction` 和 `build_step_execute_user_prompt` 两个 per-step 提示；③ `model_service.execute_plan` 从单次 astream 重写为步骤循环，每步调用 `_execute_step_with_tools`（bind_tools + 最多 5 轮工具 re-invoke），步骤结果累积进 history 传给下一步，on_delta 仅传最后一步；④ 新增 `TestSafeCalculate`（纯函数测试）和 `TestExecutePlanStepByStep`（步骤逻辑测试），16/16 测试通过。面试被质疑"execute 只有一次 LLM 调用"的问题在此版本中完全修复。
- 仍待处理：无
- 下一个 session 启动时先看什么：todo 已清空，可集成验证数学题场景（education 模式发"计算(123+456)×789/12"，观察日志中 [execute_plan] step=N/N 多步输出及 calculate 工具调用）。

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

## 2026-04-11
- 今天做了什么：三项并行改进。① **结构化输出重构**：新增 `app/schemas/llm_outputs.py`（`PlanResult`、`ReactResponse`、`EpisodicSummary`、`TopicSummary`），`ModelService._call_llm_json` 从手写 JSON schema dict + 原始 OpenAI client 改为 Pydantic + `langchain_openai with_structured_output(method="json_mode")`，删除 `_text_client`、`_vision_client`、`_build_payload`、`_build_schema_instruction`、`_parse_json_from_text` 等样板。② **ReAct 链路可观测性**：`chat_service`、`reason`、`observe`、`respond`、`basic_tools` 各节点统一增加结构化 `logger.info`（前缀 `[chat_request]`/`[react_reason_*]`/`[react_observe]`/`[react_respond]`/`[react_tool_start]`），同时在 `model_service` 中对 reason/response 两处 prompt 调用 `_dump_debug_json` 写调试文件。③ **Prompt 优化**：`plan_prompts` 细化 RAG 触发策略（动物/植物/自然科学明确强制 `needs_retrieval=true`）；`tutor_prompts` 在 reason user prompt 注入 `selected_act`/`selected_tool`/`tool_success`/`observation_summary`/`react_history`，新增工具去重指导（同一工具不重复调用）；新增 `check.md` 含 T01–T15 共 15 条手工验收测试用例。
- 下一个 session 启动时先看什么：按 `check.md` T01–T15 手工跑一遍，重点验 T02/T03（RAG 触发）和 T09（Tavily 搜索）。

## 2026-04-11（makeorder）
- 已完成并归档：`fix-reason-read-memory-prompt`、`fix-respond-json-mode-empty-message`（共 2 项）
- 仍待处理：无
- 下一个 session 启动时先看什么：todo 已清空，可按 check.md NM-01b/NM-04b 重新跑验收，确认跨会话召回修复生效。

## 2026-04-11（第二轮修复）
- 今天做了什么：根据第二轮验收报告（check_results.md）补全 respond 节点修复。① `ReactResponse.message` 默认值从兜底字符串改为 `""`（llm_outputs.py），使 plain invoke fallback 能够正确触发；② read_memory 工具触发机制已在第一轮修复中验证生效（NM-01b/NM-04b workflow_trace 含 tools + chatbot 2 次迭代）。
- 仍待处理：无
- 下一个 session 启动时先看什么：todo 已清空，建议跑第三轮验收确认 NM-01b/NM-02b/NM-04b message 内容正确（不再为兜底值）。

## 2026-04-12
- 今天做了什么：四项修复与优化。① **视觉路由修复**：新增 `direct_answer` no-op 工具，LLM 在图片/简单问题时可显式选择直接回答，`ReasonNode._override_act_from_selected_tool` 识别后映射为 `direct` act，`tutor_prompts` 注入最高优先级 Vision policy（has_image/has_image_from_history 时图片类问题 MUST 调 direct_answer），prompt 新增 `Has image from history` 字段（`_has_image_in_history` 回溯历史轮次），`graph.should_continue_react` 增加 skill 成功后直接跳 respond。② **LLM schema 稳定性**：`_call_llm_json` 新增 `_build_schema_hint`，生成精确字段提示避免 qwen 等模型自造字段名或类型。③ **图片历史透传**：`_call_with_bind_tools` 改用 `_build_image_content_from_request_or_history`，reason 阶段也能看到历史轮次图片。④ **前端图片展示**：用户消息气泡直接渲染 `<img>` 标签替代 "Image URL attached." 占位文本，同时支持 image_url 和 image_base64 两种来源。另：parent_summary skill 触发条件收紧（明确排除图片/通识问题）；plan_prompts 步骤标记改为仅多步骤教育场景使用；README 全面重写为 v1.6 双框架架构描述；新增多处 debug dump 日志提升可观测性。
- 仍待处理：无
- 下一个 session 启动时先看什么：手工验证图片问答（发送图片 URL + "这是什么"），确认不再触发 retrieve_knowledge / parent_summary，直接走 direct_answer 路径生成回复。
