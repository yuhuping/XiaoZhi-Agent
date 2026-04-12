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
- **LLM 主观评估：** 回答是否用儿童能懂的语言解释了 1+1=2，语气是否轻松有趣，是否自然带出一个引导孩子思考的追问

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
- **LLM 主观评估：** 解释蜗牛背壳的原因是否准确（保护/家/移动）、语言是否符合 5-6 岁理解水平，有无生动比喻

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
- **LLM 主观评估：** 光合作用说明是否用了儿童能理解的比喻（如"植物做饭"），是否避免了照搬专业术语

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
- **LLM 主观评估：** 是否引导孩子自己想（而非直接给出"剩3个"），语气是否鼓励、有趣，追问是否推动孩子自主思考

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
- **LLM 主观评估：** 回复是否表现出真诚的陪伴感（而非机械应答），是否自然延伸话题（如问去哪玩、和谁玩），有无说教或灌输

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
- **LLM 主观评估：** 记忆引用是否自然融入对话（不生硬复述），是否让孩子感受到"被记住"的温暖感

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
- **LLM 主观评估：** 两轮对话衔接是否流畅，T07b 的回答是否感觉像真正记住了而非刻意搜索，有无借机展开猫咪相关话题

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
- **LLM 主观评估：** 建议是否实用（如分阶段引导、混入喜欢的食物等），是否站在家长视角而非教条说教，回答是否简洁不啰嗦

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
- **LLM 主观评估：** 搜索结果是否被有效整合（非逐条粘贴），是否提炼出对家长有实际参考价值的关键信息，可信度标注是否清晰

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
- **LLM 主观评估：** 记忆汇总是否有条理（按学科/时间整理），信息呈现是否让家长一目了然，是否附有建议性的后续行动

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
- **LLM 主观评估：** 降级回答是否诚实承认知识边界（"这个问题超出我现在的知识范围"），是否有引导性提示，而非捏造内容

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
- **LLM 主观评估：** 解释鸟类飞行原理是否准确（翅膀/骨骼/肌肉/空气动力），内容是否源于知识库而非凭空生成，语言是否适合儿童

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
- **LLM 主观评估：** 澄清方式是否友好（不让孩子觉得自己问错了），是否给出 2-3 个具体的方向示例引导孩子继续表达

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
- **LLM 主观评估：** 是否正确理解了"blue"即"蓝色"，解释瑞利散射是否用了儿童能接受的类比（如"太阳光是彩虹色，蓝光最爱跑"）

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
- **LLM 主观评估：** 最终给出的日期是否计算正确（2026-04-04），若无时间工具是否诚实说明，回答风格是否仍保持轻松儿童友好

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
- **LLM 主观评估：** 回复是否热情回应了孩子的自我介绍（称呼小明、提到蓝色），语气是否像朋友而非问答机器人

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
- **LLM 主观评估：** 回答是否自然地说出"你最喜欢蓝色"（而非"根据记忆你喜欢蓝色"的机械措辞），是否顺势延伸（如问为什么喜欢蓝色）

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
- **LLM 主观评估：** 是否为孩子学会画画的成就感到真诚高兴，是否鼓励孩子继续画，有无提问她画了什么

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
- **LLM 主观评估：** 是否自然地叫出"小花"（而非"你的名字是小花"），语气是否亲昵，是否有趣地延续了画画话题

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
- **LLM 主观评估：** 蜗牛解释是否准确（软体动物、壳的功能），语言是否适合 6 岁理解水平，有无激发好奇心的追问

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

**观察点（NM-04a）：**
- `memory.session_updated == true`
- 正常有回复
- **LLM 主观评估（NM-04a）：** 是否礼貌地确认了孩子信息（小宝、4岁、认字），是否给出符合认字阶段的一条简单建议或鼓励

**观察点（NM-04b）：**
- `react.selected_act == 'read_memory'`（parent 模式跨 session 也必须调工具）
- `'tools' in workflow_trace`
- 回答包含「小宝」
- **LLM 主观评估（NM-04b）：** 是否自然说出"小宝"（而非机械复述记忆），是否顺势关切孩子的认字进展

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

---

## 七、图片理解测试（Vision + 多轮追问）

> **背景：** ChatRequest 支持 `image_url` / `image_base64` 字段；视觉推理走 vision LLM（`vllm_*` 配置）。
> 追问轮不再携带图片，视觉上下文由 `state_update` 从 working memory 的 `recent_turns` 中恢复。
> 以下两条用例共用 `session_id="im-sess-01"` 形成一个对话对。

---

### IM-01a 图片识别（"这是什么？" + image_url）

```json
{
  "text": "这是什么？",
  "image_url": "https://ts1.tc.mm.bing.net/th/id/OIP-C.Mykh0w4k5mpqd4xGDKBgPQHaE7?w=193&h=135&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2",
  "mode": "companion",
  "age_hint": "6",
  "session_id": "im-sess-01"
}
```

**观察点：**
- `message` 非空，包含「长颈鹿」（图片内容为长颈鹿）
- `metadata.workflow_trace` 含 `chatbot` → `respond`（companion 模式 ReAct 路径）
- 视觉 LLM 被调用（请求携带 image_url）
- **LLM 主观评估：** 图片描述是否生动准确（不仅说"这是长颈鹿"，还能描述颜色/场景/动作），语言是否让 6 岁孩子感到惊喜和好奇

---

### IM-01b 图片追问（无图，同 session，从历史恢复图片上下文）

> 必须在 IM-01a 之后、同一测试进程中执行，同 session_id，不携带图片字段

```json
{
  "text": "那么图中具体有几只长颈鹿呢？站位情况怎么样？",
  "mode": "companion",
  "age_hint": "6",
  "session_id": "im-sess-01"
}
```

**观察点：**
- `message` 非空，包含数量描述（如「一只」「两只」「三只」或具体数字）
- 模型能根据历史中的 `image_url` 继续分析图片（`_build_image_content_from_request_or_history` 生效）
- 不应答非所问或说"没有图片"
- **LLM 主观评估：** 是否清楚说明了长颈鹿的数量和相对站位（如"一只在前、两只在后"），描述是否符合图片实际内容，语言是否儿童友好

---

## 图片理解功能总结

| 编号 | 场景 | 关键字段 | 预期行为 |
|------|------|----------|---------|
| IM-01a | 首轮带图识别 | image_url，message含「长颈鹿」 | 视觉 LLM 描述图片内容 |
| IM-01b | 追问轮（无图） | 同 session，历史含 image_url | 从 working memory 恢复图片继续分析 |

---

## 八、Skill 功能测试（generate_parent_summary）

> **背景：** `parent_summary` skill 在家长询问孩子学习情况时触发，调用 `generate_parent_summary` 工具，
> 从 episodic/semantic memory 中检索孩子记录并生成结构化摘要。
> 以下先写入一条孩子学习数据（education 模式，`profile_id` 默认为 `default_child`），再由家长查询。

---

### SK-01a Skill 前置：写入孩子学习数据（education 模式）

```json
{
  "text": "今天学了加法，1+1=2，1+2=3，老师还教我们数数到20！",
  "mode": "education",
  "age_hint": "6",
  "session_id": "sk-edu-01"
}
```

**观察点：**
- `memory.session_updated == true`（写入 working + episodic 记忆，profile_id="default_child"）
- `metadata.workflow_trace` 含 `plan` → `execute`（education 路径）
- 正常有回复
- **LLM 主观评估：** 是否为孩子学会加法感到鼓励，是否继续引导做更多加法练习，语气是否有趣而非枯燥

---

### SK-01b Skill 触发：家长查询孩子学习情况

> 必须在 SK-01a 之后执行（或已有 default_child 历史数据）

```json
{
  "text": "帮我看看孩子最近的学习情况，有什么进展吗？",
  "mode": "parent",
  "age_hint": "6",
  "session_id": "sk-parent-01"
}
```

**观察点：**
- `react.selected_act == 'skill'`（触发 skill 路由）
- `'tools' in workflow_trace`（generate_parent_summary 工具被调用）
- `metadata.workflow_trace` 含 `chatbot` → `tools` → `observe` → `respond`
- `message` 非空，内容应汇总孩子的学习记录（如加法、数数等）
- **LLM 主观评估：** 摘要是否结构清晰（按学科分类或时间排列），内容是否准确反映 SK-01a 输入（加法 1+1=2、数数到20），对家长是否有参考价值，有无给出后续学习建议

---

## Skill 功能总结

| 编号 | 模式 | 场景 | 预期行为 |
|------|------|------|---------|
| SK-01a | education | 写入孩子学习数据 | session_updated=true，走 plan 路径 |
| SK-01b | parent | 查询孩子学习进展（skill） | selected_act=skill，调用 generate_parent_summary |
