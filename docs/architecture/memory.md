# Memory 架构

## 概览

小智当前的 Memory 体系是一个分层记忆设计，目标是在“短期对话上下文”和“长期用户画像/历史积累”之间做明确分工，而不是把所有历史都塞进 prompt。

当前 Memory 入口由以下组件组成：

- `app/memory/manager.py`
- `app/memory/tool.py`
- `app/memory/storage.py`
- `app/memory/vector_store.py`
- `app/memory/types/*`

在服务装配阶段，`ChatService.create_chat_service()` 会创建：

1. `MemoryConfig`
2. `MemoryManager`
3. `MemoryTool`

然后把 `MemoryTool` 注入到 `BasicTools`，供 Graph 中的 `state_update`、`memory_update` 等节点使用。

## 分层设计

当前共有四类记忆：

| 类型 | 持久化方式 | 主要用途 | 当前默认状态 |
| --- | --- | --- | --- |
| `working` | 进程内内存 | 会话内短期上下文 | 默认启用 |
| `episodic` | SQLite + 本地向量索引 | 长期事件记忆 | 默认启用 |
| `semantic` | SQLite + 本地向量索引 | 长期抽象知识/偏好 | 已实现，主链路默认不主动写 |
| `perceptual` | SQLite + 本地向量索引 | 多模态/感知记忆预留 | 已实现，主链路默认不写 |

核心思路是：

- `working` 负责当前 session 的近场上下文
- `episodic` 负责“发生过什么”
- `semantic` 负责“抽象出了什么”
- `perceptual` 为图片等多模态长期记忆预留

## 核心数据结构

### 1. MemoryItem

统一的数据结构定义在 `app/memory/base.py`：

- `id`
- `user_id`
- `session_id`
- `memory_type`
- `content`
- `importance`
- `timestamp`
- `metadata`
- `last_accessed`
- `access_count`

其中：

- `user_id` 当前在业务上等价于 `profile_id`
- `session_id` 用于区分同一用户的不同会话
- `importance` 用于遗忘、检索排序、整合与压缩
- `metadata` 用来保留 turn、topic、mode、来源范围等业务信息

### 2. MemoryConfig

`MemoryConfig` 定义了当前记忆系统的关键运行参数：

- `db_path`
- `index_dir`
- `working_memory_capacity`
- `working_memory_ttl_minutes`
- `default_search_limit`
- `default_min_importance`
- `consolidate_working_threshold`
- `consolidate_episodic_threshold`
- `forget_default_max_age_days`
- `vector_dim`

这些值最终来源于 `Settings`，由环境变量控制。

## 组件分工

### 1. MemoryManager

`MemoryManager` 是统一调度层，负责：

- 根据 `memory_type` 路由到对应子存储
- 添加/检索/更新/删除记忆
- 执行遗忘
- 执行 consolidate
- 生成 session/profile 快照
- 生成 `read_memory_bundle`
- 提供长期记忆压缩候选

它内部持有：

- `HashingTextEmbedder`
- `SQLiteMemoryStore`
- `LocalVectorStore`
- `WorkingMemory`
- `EpisodicMemory`
- `SemanticMemory`
- `PerceptualMemory`

### 2. MemoryTool

`MemoryTool` 是对外统一 action 接口，Graph 并不直接调 `MemoryManager`，而是通过 `MemoryTool.execute(action, **kwargs)` 访问。

当前支持的 action 包括：

- `add`
- `search`
- `summary`
- `stats`
- `update`
- `remove`
- `forget`
- `consolidate`
- `clear_all`
- `read_bundle`
- `read_session`
- `read_profile`

返回值统一是：

```json
{
  "success": true,
  "data": {},
  "error": null
}
```

### 3. SQLiteMemoryStore

`SQLiteMemoryStore` 负责长期记忆的结构化持久化：

- 表 `memory_items`
- 表 `schema_version`

`memory_items` 当前保存：

- 主键 `memory_id`
- `user_id`
- `session_id`
- `memory_type`
- `content`
- `importance`
- `timestamp`
- `metadata`
- `last_accessed`
- `access_count`
- `archived`

删除默认是软删除：

- `delete_item()` 将 `archived=1`
- `hard_delete_item()` 才会物理删除

### 4. LocalVectorStore

`LocalVectorStore` 负责长期记忆向量索引，按 `user_id + memory_type` 拆分存储。

磁盘文件是两类：

- `*.npz`
  - 存向量矩阵
- `*.ids.json`
  - 存向量行与 `memory_id` 的映射

检索方式是：

- 对 query 向量归一化
- 用内积近似余弦相似度
- 按分数降序返回

### 5. HashingTextEmbedder

当前 Memory 不依赖在线 embedding，而是使用本地 `HashingTextEmbedder`。

特点：

- 无需外部服务
- 维度默认 256
- 对英文词、中文单字、中文二元片段做轻量分词
- 使用哈希桶累计词频，再做归一化

这个设计的目标不是追求高精度语义向量，而是保证本地可运行和稳定。

## 各类记忆的实现细节

### 1. WorkingMemory

`WorkingMemory` 位于 `app/memory/types/working.py`，是纯内存结构：

- 按 `user_id::session_id` 分桶
- 写入前先做 TTL 清理
- 超过容量后按 `importance + timestamp` 裁剪

它的检索分数是混合分：

- 向量相似度
- 关键词重叠
- 时间衰减
- 重要性权重

`get_snapshot()` 会返回：

- `recent_turns`
- `last_topic`
- `last_agent_question`

这是 `state_update` 恢复上下文时直接使用的结构。

### 2. EpisodicMemory

`EpisodicMemory` 表示情景记忆，强调：

- 事件顺序
- 时间近因性
- 上下文回放

评分逻辑：

- 向量相似度占 0.8
- 时间近因占 0.2
- 再乘以重要性权重

当前主链路会在每轮对话结束时默认写入 1 条合并后的 `episodic`：

```text
user: ...
assistant: ...
```

这样做是为了避免把同一轮的 user/assistant 各自重复落成两条长期记忆。

### 3. SemanticMemory

`SemanticMemory` 表示抽象语义记忆，适合沉淀：

- 用户偏好
- 稳定事实
- 抽象结论

它会在写入时自动提取轻量实体并放入 `metadata.entities`，检索分数由两部分组成：

- 向量相似度占 0.7
- 实体交集图关系占 0.3

当前主链路没有默认写 semantic，但保留了 consolidate 能力。

### 4. PerceptualMemory

`PerceptualMemory` 是为多模态长期记忆预留的层。

它支持：

- 默认写入 `metadata.modality`
- 检索时按 `target_modality` 过滤

当前代码已经实现，但主链路默认关闭写入，需要 `MEMORY_WRITE_PERCEPTUAL_ENABLED=true` 才会在 `memory_update` 中写入。

## 当前读写链路

### 1. 每轮开头：读取记忆包

`state_update` 节点固定调用 `read_memory_bundle`，返回结构为：

- `session`
  - 来自 `get_session_snapshot()`
- `profile`
  - 来自 `get_profile_snapshot()`
- `tool_success`

其中：

- `session` 主要服务于上下文恢复
- `profile` 主要服务于 prompt 中的历史背景补充

### 2. 每轮结束：写回记忆

`memory_update` 会做以下事情：

1. 写入 2 条 `working`
   - user turn
   - assistant turn
2. 写入 1 条 `episodic`
   - 合并后的对话对
3. 可选写入 1 条 `perceptual`
   - 默认关闭
4. 执行 `forget`
5. 可选执行自动 `consolidate`
   - 默认关闭

### 3. 长期压缩：memory_compact

`memory_compact` 不是 MemoryTool 的通用 action，而是 Graph 的末尾节点。

它会：

- 收集最旧的 10 条未压缩 `episodic + semantic`
- 调用 LLM 做摘要
- 写回 1 条新的 `episodic`
- 记录来源 ID、来源类型、时间范围、topic、key points
- 物理删除原始条目

这部分的元数据非常关键，因为后续检索与调试都依赖这些压缩来源信息。

## 快照与画像的当前语义

### 1. Session Snapshot

`get_session_snapshot()` 当前返回最近工作记忆窗口，主要用途是恢复这一次对话上下文，而不是生成持久画像。

### 2. Profile Snapshot

`get_profile_snapshot()` 当前实现比较轻量：

- 从 `episodic` 和 `semantic` 读长期记忆
- 截取内容前 500 字
- 拼成 `Memory.memory_summaries`

也就是说，当前的 `profile` 更像“长期记忆摘要拼接结果”，还不是结构化的用户画像系统。

## 关键配置项

当前与 Memory 最相关的配置位于 `app/core/config.py`：

| 配置项 | 作用 |
| --- | --- |
| `MEMORY_DB_PATH` | SQLite 数据库路径 |
| `MEMORY_INDEX_DIR` | 本地向量索引目录 |
| `MEMORY_WORKING_CAPACITY` | working 容量上限 |
| `MEMORY_WORKING_TTL_MINUTES` | working TTL |
| `MEMORY_CONSOLIDATE_WORKING_THRESHOLD` | working -> episodic 阈值 |
| `MEMORY_CONSOLIDATE_EPISODIC_THRESHOLD` | episodic -> semantic 阈值 |
| `MEMORY_AUTO_CONSOLIDATE_ENABLED` | 是否自动 consolidate |
| `MEMORY_WRITE_PERCEPTUAL_ENABLED` | 是否写 perceptual |
| `MEMORY_FORGET_MAX_AGE_DAYS` | forget 最大年龄 |
| `MEMORY_RESET_ON_START` | 启动时是否重置 DB |

其中最需要注意的是：

- `MEMORY_RESET_ON_START=True` 时，服务启动会删除现有 memory DB

这会导致长期记忆虽然“支持持久化”，但默认运行体验仍然是不稳定持久。

## 当前边界与限制

### 1. WorkingMemory 不落盘

`working` 只存在进程内存中：

- 服务重启即丢失
- 无法跨实例共享

### 2. Long-term memory 默认不稳定持久

虽然 `episodic / semantic / perceptual` 已经用 SQLite 落盘，但如果开启 `memory_reset_on_start`，启动时仍会被清空。

### 3. Profile Snapshot 仍偏拼接式

目前 `get_profile_snapshot()` 还没有做：

- 偏好字段抽取
- 标签结构化
- 用户画像 schema

因此 Memory 对 prompt 的支撑仍以“摘要文本拼接”为主。

### 4. PerceptualMemory 已实现但未进入默认主链路

多模态长期记忆目前只是预留能力，不应误解为项目已经具备成熟的图片跨轮记忆。

### 5. Update / access_count 使用较少

虽然 `MemoryItem` 保留了 `last_accessed`、`access_count` 等字段，但当前主链路更多是写入和检索，访问统计并没有被深度使用。

## 调试建议

遇到 Memory 问题时，建议优先检查以下位置：

1. `app/agent/nodes/state_update.py`
2. `app/agent/nodes/memory_update.py`
3. `app/agent/nodes/memory_compact.py`
4. `app/tools/basic_tools.py` 中的 `_run_memory_tool()`
5. `app/memory/manager.py`

排查重点可以放在：

- `profile_id` / `session_id` 是否正确
- `memory_reset_on_start` 是否误清库
- `working` 是否因 TTL 或容量被清掉
- `episodic` 是否写入成功
- `memory_compact` 是否过早压缩了数据
