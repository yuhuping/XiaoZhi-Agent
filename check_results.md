# 按需记忆读取功能验收报告（修复后复测）

**日期：** 2026-04-11（第二轮）
**模型：** qwen-plus-2025-07-28
**测试范围：** NM-01a ~ NM-04b（7条，对应 check.md 第六节）

---

## 汇总

| ID | 名称 | 状态 | 耗时 | tools 调用 | chatbot 迭代 | message 正确 |
|----|------|------|------|-----------|------------|------------|
| NM-01a | 跨会话写入（companion A） | ✅ PASS | 3.3s | 无（写入轮） | 1 | — |
| NM-01b | 跨会话召回（companion B） | ⚠️ 部分 | 3.5s | ✅ 有 | ✅ 2次 | ❌ 无'蓝' |
| NM-02a | 当前会话写入（companion C） | ✅ PASS | 3.0s | 无（写入轮） | 1 | — |
| NM-02b | 当前会话回忆（同 session） | ⚠️ 部分 | 2.5s | ✅ 无（正确） | 1 | ❌ 无'小花' |
| NM-03 | Education profile 自动注入 | ✅ PASS | 8.4s | plan+rag | — | ✅ 正常 |
| NM-04a | Parent 跨会话写入 | ✅ PASS | 8.6s | 无（写入轮） | 1 | — |
| NM-04b | Parent 跨会话召回 | ⚠️ 部分 | 3.1s | ✅ 有 | ✅ 2次 | ❌ 无'小宝' |

**PASS: 4 / 部分: 3 / ERROR: 0**

---

## 核心功能验证结论

### ✅ read_memory 工具触发机制：已修复

NM-01b 和 NM-04b 的 workflow_trace：

```
understand → state_update → chatbot → tools → observe → chatbot → respond → memory_update → response
```

- `'tools' in trace` ✓
- `chatbot 出现 2 次` ✓（第一轮调工具，第二轮根据观察结果回答）

**read_memory 工具现在确实被调用**，`reason` 节点 prompt 的修改生效。

### ✅ NM-02b：同 session 不调工具（正确）

```
trace: chatbot → respond（无 tools）
```

同一 session 内的回忆走上下文，不触发工具，行为正确。

### ✅ NM-03：Education 模式 profile 自动注入正常

走 `plan → execute` 路径，无 chatbot 节点，Memory 自动注入，不需要工具。

---

## 唯一残留问题：respond 节点兜底消息未被修复

**现象：** NM-01b / NM-02b / NM-04b 的 message 均为 `"Let us learn one small thing together."`

**根因：** 新增的 plain invoke fallback 从未被触发。

修复逻辑是：
```python
raw_message = result.message or ""
if not raw_message.strip():         # ← 永远不触发
    raw_message = await _invoke_response_plain(...)
```

问题在于 `ReactResponse.message` 的 Pydantic 默认值就是 `"Let us learn one small thing together."`。当 LLM 返回的 JSON 中 `message` 字段为空时，Pydantic 填入该默认值（非空字符串），导致 `not raw_message.strip()` 为 `False`，fallback 永远不触发。

**一行修复：** 将 `llm_outputs.py` 中的默认值改为空字符串：

```python
# 改前
message: str = "Let us learn one small thing together."

# 改后
message: str = ""
```

这样 LLM 返回空 message 时，Pydantic 填入 `""`，`raw_message.strip()` 为空，fallback 正常触发。

---

## 最终结论

| 维度 | 结论 |
|------|------|
| state_update 改动（Memory 空载） | ✅ 正确，已验证 |
| reason prompt 修改（read_memory 触发） | ✅ 生效，工具被调用 |
| Education 模式 profile 自动注入 | ✅ 正常 |
| respond 节点 plain invoke fallback | ❌ 未生效，需改 `ReactResponse.message` 默认值为 `""` |

**按需记忆读取的核心逻辑已正确工作**，剩余失败点是 respond 节点的独立 Bug（改一行 Pydantic 默认值）。
