# ROADMAP.md

## 项目目标
XiaoZhi 是一个面向 3–8 岁儿童的多模态学习陪伴 Agent 原型项目。

这个项目的目标不是做成完整产品，而是做成一个**能写进简历、能演示、能在面试里讲清楚的 Agent 工程项目**。

整个项目希望逐步体现这些能力：

- 多模态输入理解
- 基于工作流的 Agent 编排
- Tools / Skills 分层设计
- 本地 RAG 检索增强
- 短期记忆与轻量长期记忆
- 儿童场景安全控制
- 可观测、可调试、可解释

---

## 这份 roadmap 怎么看
这份 roadmap 不是写给外人看的，而是写给开发者自己看的。

核心思想是：

- 不一次性做完所有 Agent 热点能力
- 每个版本只解决一类核心问题
- 每一版都必须保留一个可运行 demo
- 先做“能跑”，再做“像 Agent”，再做“像工程项目”

---

## 版本总览
项目分为以下 5 个阶段：

- **v1.1：最小可运行原型**
- **v1.2：工作流化 Agent**
- **v1.3：RAG + Memory + Summary**
- **v1.4：Safety + Streaming + Observability**
- **v1.5：Tools / Skills 成型 + 架构整理 + 简历包装**

---

# v1.1：最小可运行原型

## 这一版要解决什么
先把最小主链路跑通，证明这个项目不是空想。

核心目标只有一个：

> 孩子输入图片或文字，系统能识别主题，给出儿童友好的解释，并追问一个简单问题。

这一版不是完整 Agent，只是整个项目的起点。

---

## 用户侧最终效果
最简单的 demo 路径是：

1. 输入一张图片或一段文字
2. 系统识别物体 / 主题
3. 系统用儿童化语言解释
4. 系统追问一个简单问题
5. 返回结构化结果

例如：
- 用户上传一张苹果图片
- 系统回答：“这是苹果，它是一种水果，吃起来甜甜的。你知道苹果是什么颜色吗？”

---

## 主要技术
### 1. FastAPI
用于搭建后端接口。

这一版至少需要：
- 一个主接口：接收图片或文本输入
- 一个健康检查接口

### 2. Gemini
作为当前核心模型：
- 做图像理解 / 文本理解
- 生成儿童友好的解释与追问

### 3. Prompt 模板
这一版的效果很大程度取决于 prompt 设计。

需要至少有：
- 儿童解释 prompt
- 儿童追问 prompt
- 统一的角色与语气约束

### 4. Pydantic Schema
明确请求与返回格式。

至少要定义：
- 输入 schema
- 输出 schema

推荐输出字段：
- `topic`
- `explanation`
- `follow_up_question`

---

## 这一版的 tools
v1.1 不需要正式搞 tool registry，但建议从一开始就按“工具意识”写代码。

可先有这些轻量 tools / helpers：

- `detect_topic`
- `generate_explanation`
- `generate_followup_question`

此时它们可以只是 service 函数，不必强行上复杂抽象。

---

## 这一版的 skills
v1.1 只需要一个核心 skill 雏形：

### ExplainAndAskSkill
职责：
- 理解图片/文本
- 识别对象或主题
- 给出儿童友好解释
- 追问一个简单问题

注意：
这一版不用单独做 skill framework，但在设计思路上，要知道你现在做的是这个 skill 的最小版本。

---

## 这一版建议的代码结构
建议尽量简单：

- `app/api/`
- `app/schemas/`
- `app/services/`
- `app/prompts/`
- `app/core/`

例如：
- `app/main.py`
- `app/api/chat.py`
- `app/schemas/chat.py`
- `app/services/chat_service.py`
- `app/services/model_service.py`
- `app/prompts/tutor_prompts.py`

---

## 这一版完成标准
满足以下条件即可进入下一版：

- 本地后端能启动
- 图片或文本输入能正常返回结果
- 输出中包含“解释 + 一个追问”
- 回复风格基本儿童友好
- 代码结构没有明显乱掉

---

## 这一版不要做什么
- 不要急着上 LangGraph
- 不要急着做 RAG
- 不要急着做 Memory
- 不要急着搞复杂前端
- 不要急着做 MCP server
- 不要急着搞很多工具注册机制

一句话：
**先把最小链路跑通。**

---

# v1.2：工作流化 Agent

## 这一版要解决什么
把 v1.1 的“单接口 + 一坨逻辑”升级成真正有 **工作流** 的 Agent。

核心目标：

> 让项目从“调用大模型的 demo”变成“有节点、有状态流转的 Agent 原型”。

---

## 这一版为什么重要
这是项目里最关键的一个升级点。

因为从这一版开始，你就能比较自然地说：

- 我不是只写了 prompt
- 我做的是显式 workflow Agent
- 我有 perception / planning / generation 的节点划分

这对简历和面试都很重要。

---

## 主要技术
### 1. LangGraph
作为 Agent 工作流编排框架。

建议先做最小图，不要复杂化。

### 2. AgentState
定义一个显式状态对象，至少包含：

- 当前输入
- 识别到的主题
- 当前阶段
- 下一步动作
- 回复草稿
- 最终回复

### 3. Node 拆分
把逻辑拆成节点，例如：

- `perception`
- `planning`
- `generation`
- `response`

后续再逐渐加别的节点。

---

## 这一版的 tools
从这一版开始，建议正式体现 tools 层。

推荐先抽这些基础 tools：

- `detect_object`
- `load_age_policy`
- `generate_child_response`

说明：
这些 tool 还不一定是对外暴露的 MCP tool，但在代码结构上要开始“像工具”。

要求：
- 有清晰职责
- 输入输出尽量明确
- 不要把全部逻辑藏进一个 node 里

---

## 这一版的 skills
这一版建议明确两个任务级 skill：

### 1. ExplainAndAskSkill
继续沿用 v1.1 的核心能力，但现在放在 workflow 中调度。

### 2. PlanNextStepSkill
根据当前输入和状态，判断下一步该做什么：
- 解释
- 提问
- 简单继续

这一版的 skill 不要太多，但要开始建立“skill 是任务级能力，不是单一函数”的意识。

---

## 建议的 workflow
最小可用工作流：

1. `perception`
2. `planning`
3. `generation`
4. `response`

如果你想稍微预埋后续扩展位，可以加空壳节点：
- `retrieval`
- `memory_update`
- `safety_check`

但不要求现在做完。

---

## 这一版建议新增代码结构
可新增：

- `app/agent/graph.py`
- `app/agent/state.py`
- `app/agent/nodes/`
- `app/tools/`

---

## 这一版完成标准
满足以下条件即可进入 v1.3：

- 项目主链路已经跑在 graph 上
- state 是显式定义的
- 主要逻辑已经拆成节点
- 至少有基础 tools 形态
- 代码明显更像 Agent，而不是普通接口服务

---

## 这一版不要做什么
- 不要把 graph 设计得过度复杂
- 不要一上来做多 agent
- 不要过度追求智能规划
- 不要把所有节点都做全

一句话：
**先把“工作流”这件事立起来。**

---

# v1.3：RAG + Memory + Parent Summary

## 这一版要解决什么
让 Agent 不只是“会说”，还要：

- 会查知识
- 会记住上下文
- 会给家长做总结

这是项目从“Agent 骨架”变成“完整 AI 应用”的关键一版。

---

## 主要技术
### 1. Embedding + FAISS
构建本地知识库检索能力。

推荐先做很小的知识库：
- 动物
- 植物
- 食物
- 交通工具

RAG 流程：
- 文本整理
- chunk 切分
- embedding
- FAISS 建库
- 检索结果拼接进 prompt

### 2. Session Memory
短期记忆，用于记住当前对话上下文。

建议记录：
- 最近几轮对话
- 当前主题
- 当前对象
- 最近一次纠错
- 当前学习阶段

### 3. Profile Memory
轻量长期记忆，用于记录少量长期信息。

建议只保留：
- 年龄段
- 喜欢的话题
- 常见错误点

不要做太重。

### 4. Parent Summary Generation
生成家长可读的简短摘要，例如：
- 今天学了什么
- 哪些地方答得好
- 哪些地方还需要练习

---

## 这一版的 tools
v1.3 开始 tools 会明显丰富起来。

建议加入：

- `retrieve_knowledge`
- `read_session_memory`
- `write_session_memory`
- `read_profile_memory`
- `write_profile_memory`
- `generate_parent_summary`

这时候你已经能比较自然地说：
> Agent 通过工具完成检索、记忆和总结。

---

## 这一版的 skills
这是最适合把 skills 明确化的一版。

建议至少形成以下几个技能：

### 1. ObjectTeachingSkill
流程：
- 感知物体
- 检索知识
- 生成儿童解释
- 提一个问题

### 2. CorrectionSkill
流程：
- 读取孩子回答
- 判断是否正确 / 部分正确
- 先鼓励
- 再纠错
- 给一个简单引导

### 3. TopicExtensionSkill
流程：
- 基于当前主题继续延伸一轮
- 控制难度不要过高

### 4. ParentSummarySkill
流程：
- 读取 session 信息
- 总结表现
- 输出家长摘要

注意：
skills 是组合型能力，不是单点函数。

---

## 这一版建议新增代码结构
可新增：

- `app/rag/`
- `app/memory/`
- `app/skills/`

例如：
- `app/skills/object_teaching.py`
- `app/skills/correction.py`
- `app/skills/parent_summary.py`

---

## 这一版完成标准
满足以下条件即可进入 v1.4：

- 回答已经能结合检索知识
- 多轮中能记住简单上下文
- 已有 session / profile memory 雏形
- 能生成家长摘要
- 至少形成 2–4 个明确 skills

---

## 这一版不要做什么
- 不要做超复杂长期记忆
- 不要做大型知识库
- 不要做“什么都记”的 memory
- 不要把 summary 做成太长报告

一句话：
**够用即可，重点是把 RAG 和 Memory 这两个卖点立起来。**

---

# v1.4：Safety + Streaming + Observability

## 这一版要解决什么
让项目从“功能完整”升级成“工程上更靠谱”。

儿童场景下，不能只会回答，还得：

- 回答安全
- 输出稳定
- 可流式展示
- 可定位问题

---

## 主要技术
### 1. Safety Check
增加安全审查层。

可以先做成：
- prompt-based 安全判断
- 规则过滤
- 或二者结合

重点关注：
- 危险建议
- 成人不当内容
- 情感依赖诱导
- 越界角色扮演
- 偏离教育场景的聊天

### 2. Fallback 机制
一旦发现风险，不直接返回原答案，而是进入 fallback：

- 拒绝
- 重定向
- 安全解释
- 必要时提示找家长/老师

### 3. SSE
加入流式输出，让 demo 更像实时 Agent。

这一阶段先用 SSE 就够了，不要急着做 WebSocket。

### 4. Logging / Trace
加入结构化日志或简易 trace。

至少记录：
- session_id
- 当前节点
- 调用了哪些 tools / skills
- 是否触发 safety
- 最终走了哪条路径
- 错误发生在哪一步

---

## 这一版的 tools
新增工程向和安全向 tools：

- `check_child_safety`
- `fallback_safe_response`
- `stream_response`
- `trace_step`

---

## 这一版的 skills
建议补齐安全相关 skill：

### 1. SafeRedirectSkill
遇到不合适内容时，安全转移回学习话题。

### 2. SafeRefusalSkill
当内容明确不能答时，进行儿童友好拒答。

### 3. UncertainRecognitionSkill
当图像识别不确定时，不乱答，先做澄清。

---

## 这一版建议新增代码结构
可新增：

- `app/observability/`
- `app/safety/`
- `app/api/stream.py`

---

## 这一版完成标准
满足以下条件即可进入 v1.5：

- 不安全内容能被拦截
- 存在 fallback 路径
- 已支持 SSE 流式输出
- 日志足够帮助调试
- 调用 tools / skills 的过程基本可追踪

---

## 这一版不要做什么
- 不要追求复杂的安全评测平台
- 不要急着接入很多外部观测平台
- 不要把 streaming 做得太重

一句话：
**先做到“安全可控、可演示、可调试”。**

---

# v1.5：Tools / Skills 成型 + 架构整理 + 简历包装

## 这一版要解决什么
这一版不一定是功能最多的，但很关键：

> 把整个项目整理成一个“能清楚表达的 Agent 工程项目”。

这一版重点是结构、规范、表达，而不是再疯狂加功能。

---

## 主要技术 / 工程整理点
### 1. Tools / Resources / Prompts / Skills 分层
这是这一版最关键的架构表达。

建议正式形成四层：

#### Tools
原子能力：
- 检索
- 记忆读写
- 安全检查
- 总结生成
- 流式输出

#### Resources
静态或半静态资源：
- 年龄规则
- 安全规则
- 知识片段
- 用户画像

#### Prompts
提示模板：
- 解释 prompt
- 追问 prompt
- 鼓励纠错 prompt
- 家长总结 prompt

#### Skills
任务级能力：
- ExplainAndAskSkill
- CorrectionSkill
- TopicExtensionSkill
- ParentSummarySkill
- SafeRedirectSkill

### 2. Tool Contract 标准化
建议给 tool 明确约束：
- 输入 schema
- 输出 schema
- 错误处理
- 调用日志

### 3. Config / Settings 规范化
建议整理：
- `.env`
- 配置中心
- model name
- 路径配置
- 日志级别
- demo 开关

### 4. README / Demo / 架构图
这一版开始准备对外表达材料：
- README
- 架构图
- 运行命令
- demo 脚本
- 项目截图

---

## 这一版的 tools
从“有工具”升级成“工具体系清楚”。

目标不是增加一堆新 tools，而是让 tools 具备：
- 清晰命名
- 明确职责
- 统一接口
- 好记录日志
- 好扩展

---

## 这一版的 skills
从“有几个 skill 文件”升级成“skills 层成型”。

建议最终保留几类核心 skills：

- `ExplainAndAskSkill`
- `CorrectionSkill`
- `TopicExtensionSkill`
- `ParentSummarySkill`
- `SafeRedirectSkill`

不宜过多，重点是代表性强、可讲清楚。

---

## 这一版建议新增代码结构
如果前面还没彻底整理，这一版应整理为接近以下结构：

- `app/api/`
- `app/agent/`
- `app/tools/`
- `app/skills/`
- `app/resources/`
- `app/prompts/`
- `app/rag/`
- `app/memory/`
- `app/safety/`
- `app/observability/`
- `app/core/`

---

## 这一版完成标准
整个 v1.x 结束时，项目应达到：

- 支持多模态输入
- 有显式 workflow
- 有 tools 层
- 有 skills 层
- 有本地 RAG
- 有 session / profile memory
- 有 parent summary
- 有 safety + fallback
- 有 SSE
- 有日志与基本 trace
- 有清晰 README 和可演示结构

---

# 技术演进总表

## v1.1
重点技术：
- FastAPI
- Gemini
- Prompt
- Pydantic Schema

关键词：
- 跑通最小 demo

---

## v1.2
重点技术：
- LangGraph
- AgentState
- workflow nodes
- 基础 tools

关键词：
- 从 demo 变成 Agent

---

## v1.3
重点技术：
- Embedding
- FAISS
- RAG
- Session Memory
- Profile Memory
- Summary
- 明确 skills

关键词：
- 会查、会记、会总结

---

## v1.4
重点技术：
- Safety check
- fallback
- SSE
- logging / trace
- 安全相关 skills

关键词：
- 安全、可演示、可调试

---

## v1.5
重点技术：
- Tools / Resources / Prompts / Skills 分层
- tool contract
- settings/config
- README / 架构图 / demo 包装

关键词：
- 架构成型、表达清晰、适合简历和面试

---

# 总开发原则

## 1. 每一版都必须能跑
不要为了后面版本，把当前版本搞得跑不起来。

## 2. 不要一次性做全
一次只解决一类问题：
- 先跑通
- 再 workflow
- 再 RAG / memory
- 再 safety / observability
- 再整理表达

## 3. 优先做最能写进简历的部分
对这个项目来说，最有价值的不是堆很多功能，而是把下面几点做好：
- workflow
- RAG
- memory
- safety
- observability
- tools / skills 分层

## 4. skills 不要太多
几个代表性 skill 就够，不要造很多名字但没有实际价值的 skill。

## 5. tools 要逐步标准化
前期可以是 helper function，后期再统一成正式 tools 体系。

---

# 当前建议的开发顺序
建议实际开发时按这个顺序推进：

1. v1.1 跑通最小 demo
2. v1.2 接入 LangGraph 和状态流转
3. v1.3 做本地 RAG
4. v1.3 做 session / profile memory
5. v1.3 做 parent summary
6. v1.4 做 safety + fallback
7. v1.4 做 SSE + logging
8. v1.5 做 tools / skills / resources / prompts 分层整理
9. v1.5 补 README、架构图、演示材料

---

# 最终目标
当 v1.x 做完时，这个项目应该能够让你比较有底气地讲：

- 我做了一个面向儿童场景的多模态学习 Agent
- 它不是简单 prompt demo，而是基于 LangGraph 的显式工作流系统
- 我设计了 tools 与 skills 分层
- 我引入了本地 RAG、短期记忆、轻量长期记忆与家长总结
- 我考虑了儿童场景安全、fallback、流式交互与可观测性
- 这个项目既能 demo，也能作为简历中的 Agent 工程项目讲清楚