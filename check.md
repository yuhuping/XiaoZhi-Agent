# 小智功能验收测试 Prompts

> 运行前确保服务已启动：`python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
>
> 每个测试用 curl 或 httpie 发 POST 到 `http://127.0.0.1:8000/api/v1/chat/explain-and-ask`
>
> 观察要点写在每条用例后面。

---

## 一、Education 模式 — Plan-and-Execute（教育模式）

### T01 直接回答（不触发 RAG）

```json
{
  "text": "1加1等于几？",
  "mode": "education",
  "age_hint": "5-6",
  "session_id": "test-edu-01"
}
```

**观察点：**
- `react.selected_act` 应为 `direct`
- `grounding.used_rag` 应为 `false`
- `metadata.workflow_trace` 应包含 `plan` → `execute`（不含 `tools`）
- 回答语气应童趣、有引导性追问

---

### T02 知识检索（触发 RAG）— 动物类问题

```json
{
  "text": "蜗牛为什么背着一个壳？",
  "mode": "education",
  "age_hint": "5-6",
  "session_id": "test-edu-02"
}
```

**观察点：**
- `react.selected_act` 应为 `retrieve_knowledge`
- `grounding.used_rag` 应为 `true`，`grounding.sources` 非空
- `metadata.workflow_trace` 应包含 `plan` → `tools` → `observe` → `execute`
- 回答应基于《十万个为什么》内容，有具体解释

---

### T03 知识检索（触发 RAG）— 植物类问题

```json
{
  "text": "植物为什么要晒太阳？",
  "mode": "education",
  "age_hint": "4-5",
  "session_id": "test-edu-03"
}
```

**观察点：**
- 同 T02，验证植物类也走 RAG 路径（plan 节点有显式规则）
- 回答应提到光合作用，但用儿童友好的语言

---

### T04 复杂推理（数学/逻辑，不触发 RAG）

```json
{
  "text": "小明有5个苹果，送给小红2个，还剩几个？",
  "mode": "education",
  "age_hint": "6-7",
  "session_id": "test-edu-04"
}
```

**观察点：**
- `grounding.used_rag` 应为 `false`
- plan_steps 应有步骤拆解（数苹果的逻辑）
- 有引导孩子自己算的追问

---

## 二、Companion 模式 — ReAct（伴陪聊天）

### T05 日常闲聊（direct，不调工具）

```json
{
  "text": "今天天气真好，我想去玩！",
  "mode": "companion",
  "age_hint": "5-6",
  "session_id": "test-comp-01"
}
```

**观察点：**
- `react.selected_act` 应为 `direct`
- `metadata.workflow_trace` 应只有 `reason` → `respond`（无 tools）
- 语气温暖自然，无说教感
- `follow_up_question` 非空（继续话题）

---

### T06 记忆读取（触发 read_memory）

> 先发 T05，再发这条，使用同一 session_id

```json
{
  "text": "你还记得我刚才说了什么吗？",
  "mode": "companion",
  "age_hint": "5-6",
  "session_id": "test-comp-01"
}
```

**观察点：**
- `react.selected_act` 应为 `read_memory`
- `metadata.workflow_trace` 应包含 `reason` → `tools` → `observe` → `respond`
- 回答应正确引用上一轮说的"天气好想玩"

---

### T07 多轮连续对话（记忆连贯性）

> 发送顺序：先 T07a，再 T07b

**T07a（第一轮）：**
```json
{
  "text": "我最喜欢的动物是小猫咪！",
  "mode": "companion",
  "age_hint": "5",
  "session_id": "test-comp-02"
}
```

**T07b（第二轮）：**
```json
{
  "text": "你知道我最喜欢什么动物吗？",
  "mode": "companion",
  "age_hint": "5",
  "session_id": "test-comp-02"
}
```

**观察点：**
- T07b 的 `memory.session_updated` 应为 `true`
- 回答应提到"小猫咪"（working memory 生效）

---

## 三、Parent 模式 — ReAct + Tavily（家长模式）

### T08 直接回答家长问题（不调工具）

```json
{
  "text": "孩子不肯吃蔬菜怎么办？",
  "mode": "parent",
  "age_hint": "4-5",
  "session_id": "test-parent-01"
}
```

**观察点：**
- 语气专业，针对家长而非孩子
- `react.selected_act` 应为 `direct`
- 建议具体可操作

---

### T09 触发 Tavily 网络搜索（时效性问题）

```json
{
  "text": "最近有没有关于儿童早教的新研究？",
  "mode": "parent",
  "age_hint": "3-5",
  "session_id": "test-parent-02"
}
```

**观察点：**
- `react.selected_act` 应为 `tavily_search`
- `metadata.workflow_trace` 应包含 `reason` → `tools` → `observe` → `respond`
- 回答应引用近期信息（非截断知识）
- `grounding.sources` 非空，有来源 URL

---

### T10 家长询问孩子情况（触发 read_memory）

> 使用已有记录的 session/profile，或先在 education 模式建立几轮对话后再问

```json
{
  "text": "我的孩子最近学了什么内容？",
  "mode": "parent",
  "session_id": "test-parent-03",
  "profile_id": "default_child"
}
```

**观察点：**
- `react.selected_act` 应为 `read_memory`
- 回答应汇总从 episodic/semantic memory 中检索到的孩子学习记录
- 若无记录，应给出礼貌提示

---

## 四、RAG 细节验证

### T11 RAG 未命中（知识库里查不到的内容）

```json
{
  "text": "量子力学是什么？",
  "mode": "education",
  "age_hint": "6",
  "session_id": "test-rag-01"
}
```

**观察点：**
- 即使触发检索，`grounding.sources` 可能为空或低分
- 系统应降级到直接回答，而非崩溃或乱引用
- `metadata.confidence` 可能为 `medium` 或 `low`

---

### T12 RAG 命中验证（《十万个为什么》明确有的内容）

```json
{
  "text": "鸟为什么会飞？",
  "mode": "education",
  "age_hint": "5-6",
  "session_id": "test-rag-02"
}
```

**观察点：**
- `grounding.used_rag` 应为 `true`
- `grounding.sources[0].score` 应相对高（表示命中）
- 回答内容应与《十万个为什么》中的鸟类相关内容一致

---

## 五、边界与异常测试

### T13 极短输入

```json
{
  "text": "为什么？",
  "mode": "education",
  "age_hint": "5",
  "session_id": "test-edge-01"
}
```

**观察点：**
- 系统不应崩溃
- 应给出澄清引导，例如"你是想问什么的为什么？"

---

### T14 混合中英文输入

```json
{
  "text": "为什么天空是blue的？",
  "mode": "education",
  "age_hint": "6",
  "session_id": "test-edge-02"
}
```

**观察点：**
- 系统应正常处理混合语言输入
- 回答应用中文，保持风格一致

---

### T15 ReAct 最大迭代边界（companion 模式中连续追问工具）

```json
{
  "text": "今天是2026年4月11日，上周三是几月几日？",
  "mode": "companion",
  "age_hint": "7",
  "session_id": "test-edge-03"
}
```

**观察点：**
- ReAct 可能调用时间计算逻辑
- 最多 3 次迭代后应给出答案，不应死循环
- `metadata.workflow_trace` 中 `reason` 出现次数 ≤ 3

---

## 执行方式（curl 示例）

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/explain-and-ask \
  -H "Content-Type: application/json" \
  -d '{"text":"蜗牛为什么背着一个壳？","mode":"education","age_hint":"5-6","session_id":"test-edu-02"}' \
  | python3 -m json.tool
```

或使用 SSE 流式接口：

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/explain-and-ask-stream \
  -H "Content-Type: application/json" \
  -d '{"text":"蜗牛为什么背着一个壳？","mode":"education","age_hint":"5-6","session_id":"test-edu-02"}'
```

---

## 测试重点总结

| 编号 | 模式 | 核心特性 | 关键字段 |
|------|------|----------|----------|
| T01 | education | Plan 直接回答 | `selected_act=direct`, `used_rag=false` |
| T02 | education | Plan + RAG（动物） | `selected_act=retrieve_knowledge`, `used_rag=true` |
| T03 | education | Plan + RAG（植物） | 同上 |
| T04 | education | Plan 推理（数学） | `used_rag=false`, plan_steps 有拆解 |
| T05 | companion | ReAct 直接 | `selected_act=direct`, 无 tools |
| T06 | companion | ReAct + read_memory | `selected_act=read_memory` |
| T07 | companion | 多轮记忆连贯 | `memory.session_updated=true` |
| T08 | parent | 直接建议 | 语气专业 |
| T09 | parent | Tavily 搜索 | `selected_act=tavily_search`, sources 非空 |
| T10 | parent | 孩子记录查询 | `selected_act=read_memory` |
| T11 | education | RAG 未命中降级 | 不崩溃，confidence 可能低 |
| T12 | education | RAG 高置信命中 | `sources[0].score` 高 |
| T13 | education | 极短输入 | 不崩溃，引导澄清 |
| T14 | education | 混合语言 | 中文回答，逻辑正常 |
| T15 | companion | ReAct 迭代上限 | `reason` 出现次数 ≤ 3 |

---

## 六、按需记忆读取功能验收（v1.7 新功能）

> **背景：** 原来跨会话历史全部注入上下文，导致 `read_memory_bundle` 工具形同虚设。
> 现已改为：`companion`/`parent` 模式启动时 `Memory={}` 空载，LLM 按需调用工具拉取跨会话记忆；`education` 模式仍自动注入 profile 记忆。
> 以下用例使用独立 profile_id `nm-test-child` / `nm-test-parent` 隔离数据。

---

### NM-01a 跨会话记忆写入（companion，session A）

```json
{
  "text": "我叫小明，我最喜欢的颜色是蓝色！",
  "mode": "companion",
  "age_hint": "6",
  "session_id": "nm-sess-A",
  "profile_id": "nm-test-child"
}
```

**观察点：**
- `memory.session_updated == true`（当前 session 写入成功）
- `memory.written_types` 非空
- 正常有回复

---

### NM-01b 跨会话记忆召回（companion，session B，同 profile_id）

> 必须在 NM-01a 之后执行，使用不同 session_id 但同一 profile_id

```json
{
  "text": "你知道我最喜欢什么颜色吗？",
  "mode": "companion",
  "age_hint": "6",
  "session_id": "nm-sess-B",
  "profile_id": "nm-test-child"
}
```

**观察点：**
- `react.selected_act == 'read_memory'`（主动调工具拉取跨会话记忆）
- `'tools' in workflow_trace`
- 回答包含「蓝」（成功从 profile 记忆中找到颜色偏好）

---

### NM-02a 当前会话记忆写入（companion，session C）

```json
{
  "text": "我的名字叫小花，今天学会了画画！",
  "mode": "companion",
  "age_hint": "6",
  "session_id": "nm-same-01",
  "profile_id": "nm-test-child-2"
}
```

**观察点：**
- `memory.session_updated == true`
- 正常有回复

---

### NM-02b 当前会话回忆（同 session，应从上下文直接回答，不调工具）

> 与 NM-02a 使用完全相同的 session_id

```json
{
  "text": "你还记得我的名字吗？",
  "mode": "companion",
  "age_hint": "6",
  "session_id": "nm-same-01",
  "profile_id": "nm-test-child-2"
}
```

**观察点：**
- `react.selected_act == 'direct'`（无需调工具，short_Memory 已含当前 session 历史）
- `'tools' not in workflow_trace`
- 回答包含「小花」

---

### NM-03 Education 模式 profile 记忆自动注入（不触发工具）

> 沿用 nm-test-child profile（NM-01a 已写入数据），用全新 session_id

```json
{
  "text": "蜗牛是什么动物？",
  "mode": "education",
  "age_hint": "6",
  "session_id": "nm-edu-new",
  "profile_id": "nm-test-child"
}
```

**观察点：**
- `'plan' in workflow_trace`（走 education 路径）
- workflow_trace 中**无** `chatbot`（说明 education 模式不走 ReAct，Memory 已自动注入）
- 正常有回复

---

### NM-04 Parent 模式跨会话必须调工具

> 先建立一些 parent profile 数据，再新 session 查询

**NM-04a（写入，session P1）：**
```json
{
  "text": "我的孩子叫小宝，今年4岁，最近在学认字。",
  "mode": "parent",
  "age_hint": "4",
  "session_id": "nm-parent-A",
  "profile_id": "nm-test-parent"
}
```

**NM-04b（读取，session P2，新 session 同 profile）：**
```json
{
  "text": "你还记得我孩子叫什么名字吗？",
  "mode": "parent",
  "age_hint": "4",
  "session_id": "nm-parent-B",
  "profile_id": "nm-test-parent"
}
```

**观察点（NM-04b）：**
- `react.selected_act == 'read_memory'`（parent 模式跨 session 也必须调工具）
- `'tools' in workflow_trace`
- 回答包含「小宝」

---

## 按需记忆功能总结

| 编号 | 模式 | 场景 | 预期行为 |
|------|------|------|---------|
| NM-01a | companion | 写入跨 session 记忆 | session_updated=true |
| NM-01b | companion | 跨 session 召回 | selected_act=read_memory，回答含「蓝」 |
| NM-02a | companion | 写入当前 session | session_updated=true |
| NM-02b | companion | 同 session 回忆 | 无工具调用，直接从上下文回答「小花」 |
| NM-03 | education | profile 自动注入 | 走 plan 路径，无 chatbot 节点 |
| NM-04a | parent | 写入跨 session 记忆 | session_updated=true |
| NM-04b | parent | 跨 session 召回 | selected_act=read_memory，回答含「小宝」 |
