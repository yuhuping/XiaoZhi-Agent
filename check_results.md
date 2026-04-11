# 按需记忆读取功能验收报告（修复后终测）

**日期：** 2026-04-11（第三轮）
**模型：** qwen-plus-2025-07-28
**测试范围：** NM-01a ~ NM-04b（7条，对应 check.md 第六节）

---

## 汇总

| ID | 名称 | 状态 | 耗时 | tools 调用 | chatbot 迭代 | message 正确 |
|----|------|------|------|-----------|------------|------------|
| NM-01a | 跨会话写入（companion A） | ✅ PASS | 6.7s | 无（写入轮） | 1 | ✅ 含"蓝" |
| NM-01b | 跨会话召回（companion B） | ✅ PASS | 6.6s | ✅ 有 | ✅ 2次 | ✅ 含"蓝" |
| NM-02a | 当前会话写入（companion C） | ✅ PASS | 3.5s | 无（写入轮） | 1 | ✅ 含"小花" |
| NM-02b | 当前会话回忆（同 session） | ✅ PASS | 4.0s | ✅ 无（正确） | 1 | ✅ 含"小花" |
| NM-03 | Education profile 自动注入 | ✅ PASS | 11.5s | plan+rag | — | ✅ 正常 |
| NM-04a | Parent 跨会话写入 | ✅ PASS | 9.7s | 无（写入轮） | 1 | ✅ 含"小宝" |
| NM-04b | Parent 跨会话召回 | ✅ PASS | 5.0s | ✅ 有 | ✅ 2次 | ✅ 含"小宝" |

**PASS: 7 / 部分: 0 / ERROR: 0**

---

## 核心功能验证结论

### ✅ read_memory_bundle 工具触发机制：正常

NM-01b 和 NM-04b 的 workflow_trace：

```
understand → state_update → chatbot → tools → observe → chatbot → respond → memory_update → response
```

- `'tools' in trace` ✓
- `chatbot 出现 2 次` ✓（第一轮调工具，第二轮根据观察结果回答）

### ✅ NM-02b：同 session 不调工具（正确）

```
trace: understand → state_update → chatbot → respond → memory_update → response
```

同一 session 内的回忆走上下文，不触发工具，行为正确。message 含"小花" ✓

### ✅ NM-03：Education 模式 profile 自动注入正常

走 `plan → tools(rag) → observe → execute` 路径，无 chatbot 节点，Memory 自动注入。

### ✅ respond 节点 plain invoke fallback：现已生效

所有测试均出现日志 `[generate_final_response] json_mode returned empty message, falling back to plain invoke`，
说明 `ReactResponse.message` 默认值改为 `""` 后，fallback 正常触发，LLM 真实回复被正确写入 message。

---

## 本次修复内容

| 文件 | 修改 | 效果 |
|------|------|------|
| `app/schemas/llm_outputs.py` | `message: str = ""` （原为 `"Let us learn one small thing together."`） | Pydantic 不再用兜底字符串填充，fallback 检测生效 |
| `app/services/model_service.py` | try/except + `if not raw_message.strip()` → `_invoke_response_plain()` | json_mode 返回空时切换 plain invoke，获取真实 LLM 回复 |
| `app/prompts/tutor_prompts.py` | `read_memory_bundle` 策略加 MUST 条件 | LLM 跨会话问题时主动调用记忆工具 |

---

## 最终结论

| 维度 | 结论 |
|------|------|
| state_update 改动（Memory 空载） | ✅ 正确，已验证 |
| reason prompt 修改（read_memory 触发） | ✅ 生效，工具被调用 |
| Education 模式 profile 自动注入 | ✅ 正常 |
| respond 节点 plain invoke fallback | ✅ 已修复，message 内容正确 |

**按需记忆读取功能全部验证通过（7/7）。**
