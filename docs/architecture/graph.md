# Graph 架构

## 概览

`Graph` 是小智当前对话执行链路的主骨架，负责把一次 `ChatRequest` 从输入解析、状态恢复、规划决策、工具调用、回答生成，到最终响应组装串起来。当前实现基于 LangGraph，真实入口在 `app/agent/graph.py`，并由 `app/services/chat_service.py` 负责初始化与调用。

当前主链路是：

```text
START
  -> understand
  -> state_update
  -> chatbot
  -> (tools?)
  -> observe
  -> respond
  -> memory_update
  -> response
  -> memory_compact
  -> END
```

这里的核心设计目标不是开放式 Agent 自主探索，而是一个受控的、单轮决策的 ReAct 工作流：

- 先做轻量输入理解
- 再恢复记忆上下文
- 由规划模型决定本轮是否需要工具
- 最多执行一次受控工具调用
- 再由回复模型生成最终回答
- 最后统一更新记忆并返回标准响应

## 入口与装配

### 1. 请求入口

对话入口是 `POST /api/v1/chat/explain-and-ask`，路由位于 `app/api/chat.py`。

路由层只做两件事：

- 接收并校验 `ChatRequest`
- 调用 `ChatService.explain_and_ask()`

### 2. ChatService 角色

`app/services/chat_service.py` 是 Graph 的服务装配层，负责：

- 读取 `Settings`
- 初始化 `ModelService`
- 初始化 `MemoryManager` / `MemoryTool`
- 初始化 `LocalKnowledgeRetriever`
- 初始化 `BasicTools`
- 创建 `AgentGraph`

一次请求的运行流程为：

1. 由 `build_initial_state(request)` 构造初始 `AgentState`
2. 注入 `state["rag_enabled"]`
3. 调用 `await self.graph.run(state=state)`
4. 从 `final_state["final_response"]` 反序列化为 `ChatResponse`

### 3. 图编译方式

`AgentGraph` 在初始化时使用 `StateGraph(AgentState)` 编译工作流，并注册以下节点：

| 节点名 | 实现类 | 作用 |
| --- | --- | --- |
| `understand` | `UnderstandNode` | 轻量输入理解，推测 topic 与输入信号 |
| `state_update` | `StateUpdateNode` | 读取 session/profile 记忆并恢复上下文 |
| `chatbot` | `ReasonNode` | 调用规划模型做一次 ReAct 决策 |
| `tools` | `ToolNode` | 执行 LangGraph 工具 |
| `observe` | `ObserveNode` | 把工具输出回填到 state |
| `respond` | `RespondNode` | 调用回复模型生成最终回答草稿 |
| `memory_update` | `MemoryUpdateNode` | 写 working / episodic 等记忆 |
| `response` | `ResponseNode` | 组装标准 `ChatResponse` |
| `memory_compact` | `MemoryCompactNode` | 批量压缩长期记忆 |

`chatbot -> tools / observe` 这一段通过 `tools_condition` 做条件分支：

- 若规划模型产出 tool call，则进入 `tools`
- 若没有 tool call，则直接进入 `observe`

这意味着当前工作流是“单次规划 + 可选一次工具调用”，而不是多轮循环式 Agent。

## AgentState 设计

`app/agent/state.py` 中的 `AgentState` 是整条链路共享的数据载体。它既承担运行时状态，也承担最终响应所需的拼装字段。

### 1. 输入相关字段

这部分主要来自 `ChatRequest`：

- `session_id`
- `profile_id`
- `user_input`
- `latest_user_text`
- `text_input`
- `normalized_text`
- `image_base64`
- `image_url`
- `image_mime_type`
- `input_modality`
- `interaction_mode`
- `child_age_band`

其中 `input_modality` 在初始阶段按以下规则推导：

- 只有文本：`text`
- 只有图片：`image`
- 同时有文本和图片：`multimodal`

### 2. 理解与规划字段

这些字段在 `understand`、`state_update`、`chatbot` 阶段逐步填充：

- `current_topic`
- `detected_object`
- `topic_hint`
- `perception_signals`
- `history`
- `last_agent_question`
- `pending_topic`
- `route_reason`
- `react_decision`
- `selected_act`
- `selected_tool`
- `tool_input`
- `messages`

### 3. 工具与观察字段

这些字段主要由 `tools` / `observe` 更新：

- `tool_result`
- `tool_success`
- `retrieved_chunks`
- `observation_summary`
- `short_Memory`
- `Memory`

### 4. 回答与输出字段

这些字段在 `respond`、`response`、`memory_update` 阶段形成：

- `message_draft`
- `follow_up_question`
- `confidence`
- `safety_notes`
- `source_mode`
- `memory_session_updated`
- `memory_profile_updated`
- `memory_written_types`
- `memory_consolidated_count`
- `memory_forgotten_count`
- `final_response`
- `workflow_trace`

`workflow_trace` 是当前最直接的运行轨迹输出，会被写入最终响应的 `metadata.workflow_trace`，便于调试与回放。

## 节点级职责

### understand

实现位于 `app/agent/nodes/understand.py`。

作用：

- 调用 `BasicTools.detect_object()` 做轻量 topic/object 猜测
- 调用 `BasicTools.perceive_signals()` 抽取输入信号
- 初始化：
  - `current_topic`
  - `detected_object`
  - `topic_hint`
  - `perception_signals`
  - `dialogue_stage="understood"`

当前这一步是启发式规则，不依赖 LLM，也不访问工具。

### state_update

实现位于 `app/agent/nodes/state_update.py`。

作用：

- 通过 `read_memory_bundle` 读取 session 与 profile 记忆
- 从 `session.recent_turns` 中恢复：
  - `history`
  - `turn_index`
  - `last_agent_question`
  - `pending_topic`
  - `current_topic`
- 把 session/profile 快照分别填到：
  - `short_Memory`
  - `Memory`

这一节点是每轮开头的上下文恢复点，决定了本轮不是纯无状态推理。

### chatbot

实现位于 `app/agent/nodes/reason.py`。

作用：

- 调用 `BasicTools.reason_next_action()`
- 由 `ModelService.reason_next_action()` 驱动规划模型
- 把模型结果收敛到受控动作集合

当前受支持的动作类型为：

- `direct`
- `retrieve_knowledge`
- `read_memory`

工具名与动作的映射有一层收敛逻辑：

- `retrieve_knowledge` / `tavily_search` 统一归入 `retrieve_knowledge`
- `read_memory_bundle` 统一归入 `read_memory`

若决定调用工具，会在 state 中放入 `AIMessage(tool_calls=...)`，供 LangGraph 的 `ToolNode` 继续执行。

### tools

这是 LangGraph 预置的 `ToolNode`，执行的工具集合来自 `BasicTools.as_langgraph_tools()`。

当前真正注册到 Graph 中的工具有 3 个：

- `retrieve_knowledge`
- `tavily_search`
- `read_memory_bundle`

其中：

- 本地 RAG 用于稳定知识
- Tavily 用于时间敏感或联网信息
- `read_memory_bundle` 用于显式回读 session/profile 记忆

### observe

实现位于 `app/agent/nodes/observe.py`。

作用：

- 从 `messages` 中找到最近的 `ToolMessage`
- 解析其中的 JSON 内容
- 根据 `selected_act` 决定如何回填 state

分支行为如下：

- `retrieve_knowledge`
  - 回填 `retrieved_chunks`
  - 回填 `tool_result`
  - 更新 `tool_success`
- `read_memory`
  - 回填 `short_Memory`
  - 回填 `Memory`
  - 更新 `tool_result`
  - 更新 `tool_success`
- `direct`
  - 视为未使用外部工具

### respond

实现位于 `app/agent/nodes/respond.py`。

作用：

- 调用 `BasicTools.generate_final_response()`
- 进一步调用 `ModelService.generate_final_response()`
- 产出：
  - `message_draft`
  - `follow_up_question`
  - `confidence`
  - `safety_notes`
  - `source_mode`
  - `current_topic`

这一步是真正面向用户的回答生成阶段。

### memory_update

实现位于 `app/agent/nodes/memory_update.py`。

作用：

- 每轮写入两条 working turn
  - 一条 user
  - 一条 assistant
- 每轮写入一条拼接后的 `episodic`
- 可选写入 `perceptual`
  - 但默认关闭
- 执行遗忘
- 可选执行自动 consolidate
  - 默认关闭

当前默认策略是：

- 总是写 `working`
- 总是写 `episodic`
- 默认不写 `perceptual`
- 默认不自动做 consolidate
- 总是执行 forget

### response

实现位于 `app/agent/nodes/response.py`。

作用：

- 把 state 组装成标准 `ChatResponse`
- 输出字段包括：
  - `message`
  - `follow_up_question`
  - `topic`
  - `react`
  - `grounding`
  - `memory`
  - `metadata`

当前 `grounding.used_rag` 的判断方式很直接：

- 只要 `retrieved_chunks` 非空，就视为用了 grounding

### memory_compact

实现位于 `app/agent/nodes/memory_compact.py`。

作用：

- 针对 `episodic` + `semantic` 的长期记忆执行批量压缩
- 每次取最旧的 10 条未压缩记忆
- 调用 `ModelService.summarize_episodic_batch()`
- 若成功则写回 1 条新的 `episodic` 摘要，并物理删除原始 10 条

这一步的目的是降低长期记忆膨胀，而不是参与当前轮回答生成。

## Graph 与模型/工具的关系

### 1. Graph 不直接调用底层 API

Graph 节点本身不直接访问 OpenAI 或外部服务，而是通过两层抽象完成：

- `BasicTools`
- `ModelService`

这让节点保持相对薄，便于后续替换模型策略或工具实现。

### 2. BasicTools 的职责

`BasicTools` 既承担工具导出，也承担轻量启发式逻辑：

- LangGraph 工具注册
- `retrieve_knowledge`
- `tavily_search`
- `read_memory_bundle`
- topic/object 猜测
- 输入信号抽取

### 3. ModelService 的职责

`ModelService` 负责两类 LLM 调用：

- `reason_next_action()`：规划
- `generate_final_response()`：回答

同时，图片输入会在规划阶段与回答阶段都继续透传给模型，这也是当前多模态链路成立的关键。

## 当前链路的关键特性

### 1. 单轮受控 ReAct

当前不是一个可无限循环的 Agent，而是一次请求里只做一轮受控决策。这种设计的优点是：

- 路径稳定
- 易测试
- 更容易约束工具调用
- 更适合教育陪伴场景的响应延迟控制

### 2. 记忆默认前置恢复，后置写回

每轮对话都会：

- 先读 `read_memory_bundle`
- 再根据本轮结果写 `working + episodic`

这样上下文恢复与长期积累是显式串联的，不依赖模型自行“记住”历史。

### 3. 原图透传已经在链路中生效

Graph 自身不处理图片内容，但 state 会保留：

- `image_base64`
- `image_url`
- `image_mime_type`

这些字段通过 `state_to_request()` 回传给 `ModelService`，因此图片会同时参与规划与回答阶段。

## 当前边界与限制

### 1. 没有多轮工具循环

一旦完成本轮工具调用和观察，链路就会进入回答阶段，不会再次计划第二轮工具动作。

### 2. 没有中间事件流

当前 Graph 输出的是最终 `ChatResponse`，不会暴露中间节点事件、工具状态流或推理流。

### 3. `observe` 依赖工具输出 JSON

工具结果是通过 `ToolMessage` 中的 JSON 字符串回传并解析的，因此工具输出结构需要保持稳定。

### 4. 历史节点文件存在遗留代码

`app/agent/nodes/` 下仍保留了 `act_direct.py`、`act_retrieve.py`、`act_memory.py` 等旧节点文件，但当前主图并未挂载这些节点。阅读时应以 `app/agent/graph.py` 中真实注册的节点为准。

## 调试建议

定位 Graph 问题时，建议优先看以下几类信息：

- `metadata.workflow_trace`
- `react.selected_act`
- `react.tool_name`
- `grounding.sources`
- `memory.written_types`
- LangSmith trace

如果需要进一步排查，可沿着下面顺序进入代码：

1. `app/services/chat_service.py`
2. `app/agent/graph.py`
3. `app/agent/nodes/*.py`
4. `app/tools/basic_tools.py`
5. `app/services/model_service.py`
