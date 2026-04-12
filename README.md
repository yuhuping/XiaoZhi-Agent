<div align="center">
  <img src="app/logo.png" alt="XiaoZhi Logo" width="160" />

  <h1>小智 Agent</h1>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/LangGraph-Dual%20Framework-1C3C3C?style=flat-square" alt="LangGraph" />
    <img src="https://img.shields.io/badge/FastAPI-SSE%20Streaming-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/RAG-FAISS%20Local%20Retrieval-0467DF?style=flat-square" alt="RAG" />
    <img src="https://img.shields.io/badge/Memory-Working%20%2B%20Long--Term-E92063?style=flat-square" alt="Memory" />
    <img src="https://img.shields.io/badge/Skills-Parent%20Summary-D96D43?style=flat-square" alt="Skills" />
  </p>

  <p><strong>面向儿童教育、陪伴交流与家长辅助的智慧教育 Agent</strong></p>
  <p>基于 LangGraph 的双框架工作流，融合 ReAct、Plan-and-Execute、RAG、Memory 与 Skill 扩展能力</p>
</div>

---

## 项目简介

小智不是一个只会直接给答案的聊天机器人，而是尝试按不同角色切换不同推理方式：

- `education`：优先走 `Plan-and-Execute`，更强调分步骤讲解、启发式教学和过程引导。
- `companion`：优先走多轮 `ReAct`，偏自然交流、轻陪伴和灵活回应。
- `parent`：同样走多轮 `ReAct`，但会启用家长侧技能，例如根据记忆生成孩子近期学习摘要。

当前仓库已经演进为 `v1.6` 的双框架架构，不再是早期单一路径的 ReAct 原型。

## 当前能力

- `双框架路由`
  - 教育模式自动路由到 `plan -> (tools?) -> execute`
  - 陪伴 / 家长模式自动路由到多轮 `reason -> tools -> observe -> respond`
- `多模态输入`
  - 支持纯文本、图片 URL、Base64 图片
- `本地 RAG`
  - 从 `KG/` 目录加载 `.txt` / `.pdf`
  - 使用 `FAISS` 建立本地向量索引
- `分层记忆`
  - `working / episodic / semantic / perceptual`
  - 工作流内自动恢复上下文、写入长期记忆、执行遗忘/压缩
- `联网搜索`
  - 通过 `Tavily` 处理时效性或联网问题
- `Skill 扩展`
  - 已内置 `generate_parent_summary` 家长侧摘要技能
- `流式前端`
  - 根路径 `/` 内置 playground
  - 聊天接口通过 `SSE` 流式返回 `delta`
- `LangSmith 可观测性`
  - 可选开启 tracing

## 架构概览

顶层图会先做输入理解和记忆恢复，再按模式进入不同子图：

```text
START
  -> understand
  -> state_update
  -> route_by_mode
      -> education  -> plan_execute subgraph
      -> companion  -> react subgraph
      -> parent     -> react subgraph
  -> memory_update
  -> response
  -> memory_compact
  -> END
```

两个核心子图：

```text
education:
plan -> (retrieve_knowledge?) -> observe -> execute

companion / parent:
reason -> (tools -> observe -> reason)* -> respond
```

相关设计文档：

- [Graph 架构](./docs/architecture/graph.md)
- [RAG 架构](./docs/architecture/rag.md)
- [Memory 架构](./docs/architecture/memory.md)
- [v1.6 双框架设计](./docs/superpowers/specs/2026-04-09-v1.6-dual-framework-design.md)

## 目录结构

```text
XiaoZ/
├── app/
│   ├── agent/          # LangGraph 状态、节点、子图路由
│   ├── api/            # FastAPI 路由
│   ├── core/           # 配置、日志、LangSmith
│   ├── frontend/       # 内置 playground
│   ├── memory/         # 分层记忆系统
│   ├── prompts/        # 各节点 prompt
│   ├── rag/            # 本地知识检索与索引
│   ├── schemas/        # 请求 / 响应结构
│   ├── services/       # ChatService / ModelService
│   ├── skills/         # 技能插件系统
│   └── tools/          # RAG / Tavily / Memory 工具封装
├── docs/               # 架构文档与设计稿
├── tests/              # pytest 测试
├── .env.example        # 环境变量示例
└── requirements.txt    # Python 依赖
```

说明：

- `KG/` 目录默认作为知识库目录；若不存在且 `KG_AUTO_BOOTSTRAP=true`，项目会自动创建最小示例语料。
- `data/` 目录会在运行过程中按配置自动生成，用于保存记忆库和索引。
- 项目当前虚拟环境已按仓库约定放在 `./.venv`。

## 技术栈

- `LangGraph`
- `ReAct`
- `Plan-and-Execute`
- `FastAPI`
- `Pydantic v2`
- `FAISS`
- `langchain-openai`
- `LangSmith`

## 快速开始

### 1. 安装依赖

如果你已经在仓库内配置好了 `./.venv`，建议直接使用它：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 `.env`

项目会自动读取根目录下的 `.env`。可以基于 `.env.example` 复制一份：

```bash
cp .env.example .env
```

最小可运行配置如下：

```env
LLM_BASE_URL=https://your-llm-endpoint
LLM_API_KEY=your_llm_api_key
LLM_MODEL=your_text_model
```

如果你需要更多能力，可以继续补充：

```env
# 可选：视觉模型
vllm_base_url=https://your-vision-endpoint
vllm_api_key=your_vision_api_key
vllm_model=your_vision_model

# 可选：联网搜索
TAVILY_API_KEY=your_tavily_api_key
TAVILY_BASE_URL=https://api.tavily.com

# 可选：RAG embedding（当前 retriever 独立读取该 key）
RAG_embedding_model_key=your_rag_embedding_model_key

# 可选：LangSmith 追踪
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=XiaoZhi
```

### 3. 启动服务

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

启动后可访问：

- `http://127.0.0.1:8000/`：内置 playground
- `http://127.0.0.1:8000/health`：健康检查

## 接口说明

### 聊天接口

`POST /api/v1/chat/explain-and-ask`

这是一个 `SSE` 流式接口，响应内容会按事件持续返回：

- `{"delta": "..."}`：增量文本
- `{"done": true}`：流结束

请求体示例：

```json
{
  "text": "为什么月亮有时候是圆的，有时候不是？",
  "mode": "education",
  "age_hint": "7",
  "session_id": "demo-session-001",
  "profile_id": "default_child"
}
```

使用 `curl` 调试：

```bash
curl -N \
  -X POST "http://127.0.0.1:8000/api/v1/chat/explain-and-ask" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "帮我总结一下小明最近学了什么",
    "mode": "parent",
    "session_id": "demo-parent-session",
    "profile_id": "default_parent"
  }'
```

### 支持的输入字段

- `text`：文本输入，可为空，但文本和图片不能同时都为空
- `image_base64`：Base64 图片内容
- `image_url`：公网图片地址
- `image_mime_type`：上传 Base64 图片时必填，支持 `image/png`、`image/jpeg`、`image/webp`
- `mode`：`education | companion | parent`
- `age_hint`：年龄提示
- `session_id`：会话 ID
- `profile_id`：画像 ID

## 模式与工具

### Education 模式

- 使用 `PlanNode` 先生成内部步骤
- 需要时调用 `retrieve_knowledge`
- 由 `ExecuteNode` 按计划生成讲解内容
- 更适合讲题、解释概念、循序渐进教学

### Companion / Parent 模式

- 使用 `ReasonNode` 做多轮受控 ReAct 决策
- 最多迭代 `3` 轮
- 支持工具：
  - `retrieve_knowledge`
  - `tavily_search`
  - `read_memory_bundle`
- `parent` 模式额外启用：
  - `generate_parent_summary`

### 家长摘要技能

当家长提出这类请求时，模型会优先调用内置技能：

- “帮我总结一下孩子最近学了什么”
- “看看小明最近学习情况”
- “给我一份孩子近期进展摘要”

该技能会从记忆库读取 `episodic / semantic` 信息，并生成面向家长的总结文本。

## RAG 与 Memory 说明

### RAG

- 默认知识库目录：`KG/`
- 支持文件类型：`.txt`、`.pdf`
- 索引缓存目录：`app/rag/data/rag_index/`
- 当前 RAG embedding 依赖环境变量 `RAG_embedding_model_key`
- 如果 embedding 不可用，RAG 会降级为空结果，不阻塞主服务启动

### Memory

- 默认记忆数据库路径：`data/memory.sqlite3`
- 默认索引目录：`data/memory_index/`
- 每轮会自动写入：
  - `working`
  - `episodic`
- 可选写入：
  - `perceptual`
- 启动时如果 `MEMORY_RESET_ON_START=true`，会清空现有记忆库

如果你希望保留历史记忆，务必在 `.env` 中显式设置：

```env
MEMORY_RESET_ON_START=false
```

## 测试

当前仓库已包含以下测试：

- `tests/test_plan_execute.py`
- `tests/test_react_loop.py`
- `tests/test_graph_routing.py`
- `tests/test_chat_sse_streaming.py`
- `tests/test_memory_update_node.py`
- `tests/test_reason_node.py`

运行方式：

```bash
source .venv/bin/activate
pytest
```

## 开发提示

- 项目会在启动时自动创建 `ChatService`，并装配 `ModelService`、`MemoryManager`、`LocalKnowledgeRetriever`、`SkillRegistry` 等依赖。
- `README` 中描述的是当前代码已落地的能力，不代表所有设计稿都已完全产品化。
- 仓库里保留了一些用于手动调试的注释 `print`，不要在整理代码时误删。

## Support

如果这个项目对你有帮助，欢迎点一个 `Star` 支持一下。
