<div align="center">
  <img src="app/logo.png" alt="XiaoZ Logo" width="160" style="display:block; margin:0 auto 8px;" />

  <h1 style="margin: 0 0 12px;">小智Agent</h1>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
    <img src="https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C?style=flat-square" alt="LangGraph" />
    <img src="https://img.shields.io/badge/FastAPI-Web%20API-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Pydantic-Data%20Validation-E92063?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic" />
    <img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=flat-square" alt="FAISS" />
    <img src="https://img.shields.io/badge/OpenAI-LLM-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI" />
  </p>

  <p><strong>基于 ReAct + LangGraph + RAG + Memory 的智慧教育 Agent 助手</strong></p>
  <p>面向孩子学习陪伴，也支持家长模式与基础联网搜索能力</p>
</div>

---

## ✦ 项目简介

小智 的目标不是只给出答案，而是尽量按照教学陪伴的方式组织回复：

- `教育模式`：偏引导、启发、循序渐进
- `陪伴模式`：偏自然交流、轻陪伴
- `家长模式`：偏实用建议、问题排查、基础信息支持

当前项目技术核心包括：

- `ReAct` 工作流决策
- `LangGraph` 节点编排
- `RAG` 本地知识检索
- `Memory` 会话/长期记忆
- 文本与图片输入
- 前端流式回答展示

相关细节设计文档：

- [graph.md](./docs/architecture/graph.md)
- [rag.md](./docs/architecture/rag.md)
- [memory.md](./docs/architecture/memory.md)

---

## ✦ 目录结构

```text
XiaoZ/
├── app/                  # 后端主代码
│   ├── agent/            # LangGraph 节点与状态
│   ├── api/              # FastAPI 路由
│   ├── frontend/         # 简单前端页面
│   ├── memory/           # Memory 相关实现
│   ├── prompts/          # Prompt 模板
│   ├── rag/              # 本地检索
│   └── services/         # 模型/会话/业务服务
├── KG/                   # 本地知识库素材
├── docs/                 # 项目文档
├── .env                  # 环境设置存放api_key
└── requirements.txt      # Python 依赖

```

---

## ✦ 环境准备

### 1. 安装依赖
```
pip install -r requirements.txt
```

---

### 2.环境配置必填 Key

要让项目至少能正常启动并调用模型，在.env文件里面配置：

```env
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://your-llm-endpoint
LLM_MODEL=your-text-model
# 如果你要使用图片输入能力，建议同时配置下面这些：
vllm_api_key=your_vision_api_key
vllm_base_url=https://your-vision-endpoint
vllm_model=your-vision-model
# 联网搜索能力
TAVILY_API_KEY=your_tavily_api_key
TAVILY_BASE_URL=https://api.tavily.com
```


---

## ✦ 启动项目


```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## ✦ Support

如果这个项目对你有帮助，欢迎点一个 `Star` 支持一下。

