# XiaoZhi v1.2

XiaoZhi 是一个面向儿童学习场景的多模态 Agent 原型项目。

当前版本重点不是做一个开放式聊天机器人，而是搭建一个可运行、可观察、可扩展的工作流式 Agent 后端。项目强调显式状态、显式节点边界，以及一条稳定的教学交互链路。

## 项目定位

- 面向 3 到 8 岁儿童的学习辅助 Agent
- 支持文本输入、图片输入和图文混合输入
- 输出短、简单、温和、带一点引导性的教学回复
- 当前优先保证本地可运行、流程清晰、便于后续扩展

## 当前能力

- 接收文本、图片 URL、Base64 图片
- 基于工作流执行一次完整对话处理
- 识别当前主题或对象
- 根据上下文规划下一步动作
- 生成儿童友好的解释和一个追问
- 返回结构化响应而不是纯文本
- 提供前端 Playground 便于本地联调

当前主链路是：

1. `perception`
2. `state_update`
3. `planning`
4. `action_router`
5. `generation` 对应动作节点
6. `response`

## 技术重点

- 后端框架：FastAPI
- 工作流编排：LangGraph
- 状态管理：显式 `AgentState`
- 模型调用：兼容 OpenAI SDK 风格的 LLM API
- 结构化输入输出：Pydantic
- 可观测性：基础日志 + 可选 LangSmith tracing

项目核心不是“路由函数里直接调模型”，而是通过图结构驱动节点执行，让状态和流程在代码中可见。

## 功能重点

- 儿童学习场景优先，不做开放式闲聊
- 输出强调：
  - 简短
  - 易懂
  - 鼓励式表达
  - 一次只推进一个小学习步骤
- 对话结果包含：
  - `action`
  - `message`
  - `follow_up_question`
  - `topic`
  - `metadata`

## 目录结构

```text
app/
  api/          FastAPI 路由
  agent/        工作流图、状态、节点
  core/         配置、日志、LangSmith
  frontend/     本地测试页面
  prompts/      Prompt 模板
  schemas/      请求/响应模型
  services/     模型调用与业务服务
  tools/        轻量工具层

tests/          测试
scripts/        运行和调试脚本
```

几个关键文件：

- `app/main.py`
- `app/api/chat.py`
- `app/api/health.py`
- `app/agent/graph.py`
- `app/agent/state.py`
- `app/services/model_service.py`
- `app/services/chat_service.py`
- `app/frontend/index.html`

## 接口说明

健康检查：

- `GET /health`

主接口：

- `POST /api/v1/chat/explain-and-ask`

示例请求：

```json
{
  "text": "Tell me about an apple.",
  "age_hint": "4-6"
}
```

示例响应：

```json
{
  "session_id": "demo-session",
  "action": "explain_and_ask",
  "message": "An apple is a fruit. It can be red or green.",
  "follow_up_question": "What color apple do you like?",
  "topic": "apple",
  "metadata": {
    "source_mode": "openai",
    "confidence": "high",
    "safety_notes": "",
    "used_image": false,
    "dialogue_stage": "responded",
    "planned_action": "explain_and_ask",
    "workflow_trace": [
      "perception",
      "state_update",
      "planning",
      "action_router",
      "explain_and_ask",
      "response"
    ],
    "input_modality": "text",
    "route_reason": "apple topic requested"
  }
}
```

## 运行方式

安装依赖：

```powershell
pip install -r requirements.txt
```

启动后端：

```powershell
python -m uvicorn app.main:app --reload
```

运行测试：

```powershell
pytest
```

打开页面：

- 首页：`/`
- 接口文档：`/docs`

## 环境变量

当前项目优先使用 `LLM_*` 配置，旧的 `OPENAI_*` 仍兼容。

常用配置：

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_PLANNING_MODEL`
- `LLM_MAX_CONCURRENCY`
- `LLM_IMAGE_REQUEST_TIMEOUT_SECONDS`
- `REQUEST_TIMEOUT_SECONDS`
- `LANGSMITH_TRACING`

## 当前边界

当前版本暂不追求：

- 长期记忆
- RAG / 检索增强
- 复杂工具系统
- 完整安全子系统
- 流式输出
- 生产部署优化

v1.2 的目标很明确：先把 Agent 的工作流骨架、状态结构和本地演示路径立住。
