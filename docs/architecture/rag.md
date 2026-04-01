# RAG 架构

## 概览

小智当前的 RAG 体系是“本地知识库 + 向量检索”的轻量方案，主要用来回答稳定知识类问题，例如百科、常识、教学知识点，不负责时间敏感信息。

真实入口在：

- `app/rag/retriever.py`
- `app/tools/basic_tools.py`

装配入口在 `ChatService.create_chat_service()`：

1. 根据 `Settings` 初始化 `LocalKnowledgeRetriever`
2. 注入到 `BasicTools`
3. 由 Graph 在需要时通过 `retrieve_knowledge` 工具调用

## 设计目标

当前 RAG 的定位比较明确：

- 为儿童教育场景提供本地稳定知识检索
- 尽量减少 prompt 中硬编码知识
- 与主 Graph 的单轮受控 ReAct 配合
- 在 embedding 服务不可用时允许降级为空结果，而不是拖垮主服务

它不是一个通用知识中台，也不负责联网实时搜索。

## 数据源与索引产物

### 1. 知识源目录

知识源目录默认来自 `Settings.kg_dir`，即项目根目录下的 `KG/`。

当前支持的原始文件类型：

- `.txt`
- `.pdf`

`_collect_corpus_files()` 会递归扫描整个 `KG/` 目录，只收录这两类文件。

### 2. 自动 bootstrap

如果 `KG/` 目录不存在且 `KG_AUTO_BOOTSTRAP=true`，`LocalKnowledgeRetriever` 会自动：

- 创建 `KG/`
- 生成一个 `bootstrap_science.txt`

这保证了首次运行时至少有最小可用语料。

### 3. 索引缓存目录

RAG 索引缓存保存在 `DEFAULT_INDEX_DIR`，当前定义在 `app/rag/retriever.py` 中，对应 `app/rag/data/rag_index`。

每套索引会写出三类文件：

- `*.faiss.index`
  - FAISS 向量索引
- `*.chunks.json`
  - chunk 元数据与正文
- `*.meta.json`
  - 索引元数据

索引文件名基于：

- `kg_dir`
- embedding backend
- embedding model

计算出的哈希 key 生成。

## 检索器：LocalKnowledgeRetriever

### 1. 初始化流程

`LocalKnowledgeRetriever` 初始化时会依次执行：

1. 确认 `faiss` 可用
2. 读取配置
3. 初始化 embedding 客户端
4. 检查或创建 KG 目录
5. 加载缓存索引或构建新索引

### 2. Embedding 后端

当前 RAG 使用在线 embedding，而不是本地哈希向量。

实现特征：

- 客户端使用 `OpenAI`
- 默认模型是 `hunyuan-embedding`
- 默认 base URL 是 `https://api.hunyuan.cloud.tencent.com/v1`
- API Key 读取自 `RAG_embedding_model_key`

这部分目前直接写在 `app/rag/retriever.py` 顶部常量里，而不是完全由 `Settings` 托管。

### 3. Embedding 探活与降级

初始化阶段会执行 `_probe_embedding_service()` 做启动探活。

如果出现以下情况，会把检索器标记为不可用：

- 模型名为空
- base URL 为空
- API key 为空
- embedding 接口请求失败
- embedding 返回维度非法
- 向量建库结果异常

一旦 `embedding_available=False`：

- `retrieve()` 会直接返回空结果
- `force_refresh()` 会跳过重建
- `index_status.embedding_error` 会记录原因

这种设计是“可降级，不阻塞主服务”。

## 文档加载与切块

### 1. 文档加载

`_load_documents()` 的行为如下：

- `.txt`
  - 直接读取文本
  - 统一做空白清理
- `.pdf`
  - 尝试通过 `pypdf.PdfReader` 提取逐页文本
  - 每一页单独作为一个 `SourceDocument`
  - 若未安装 `pypdf` 或 PDF 解析失败，则跳过并记录 warning

### 2. 文本清洗

`_clean_text()` 会把连续空白折叠成单个空格，并去掉首尾空白。

### 3. 切块策略

`_build_chunks()` 对每个文档按字符窗口切块：

- `chunk_size` 默认 520
- `chunk_overlap` 默认 80

每个 chunk 会保留：

- `chunk_id`
- `source`
- `source_type`
- `page`
- `text`
- `length`
- `term_freq`

`chunk_id` 格式为：

```text
{source}:p{page}:{chunk_no}
```

当前切块是简单字符窗口，不是按语义段落或标题结构切分。

## 向量建库与缓存

### 1. 建库流程

首次建库或强制重建时，会按以下顺序执行：

1. 收集语料文件
2. 加载文档
3. 切块
4. 对所有 chunk 文本做 embedding
5. 构建 `faiss.IndexFlatIP`
6. 校验 `index.ntotal == len(chunks)`
7. 落盘索引文件与元数据

### 2. 缓存加载

重启后会优先尝试 `_load_cached_index()`。

当前缓存校验只检查几个关键条件：

- `kg_dir` 是否一致
- `embedding_model` 是否一致
- `chunks.json` 是否可解析
- FAISS 行数是否与 chunk 数量一致

注意：

- 当前不会自动比较 `KG/` 中源文件是否有新增或修改
- 也不会基于文件更新时间自动刷新

也就是说，缓存命中后默认信任旧索引。

### 3. 强制刷新

`force_refresh()` 提供了显式重建入口，但当前主链路不会自动调用。

如果后续调整了 `KG/` 内容，通常需要手动触发刷新或删除缓存文件后重启。

## 查询链路

### 1. Graph 中的调用方式

Graph 不直接操作 retriever，而是通过 `BasicTools._retrieve_knowledge()` 间接调用。

当前工具名为 `retrieve_knowledge`，由 `BasicTools.as_langgraph_tools()` 注册到 LangGraph。

### 2. 检索入参

当前检索主要使用：

- `query`
- `top_k`
- `min_score`

默认值来源于 `Settings`：

- `RAG_TOP_K`
- `RAG_MIN_SCORE`

### 3. 检索过程

`LocalKnowledgeRetriever.retrieve()` 会：

1. 检查 embedding 是否可用
2. 检查 query 是否为空
3. 检查当前是否已加载可用索引
4. 对 query 做 embedding
5. 在 FAISS 中搜索 top-k
6. 根据 `min_score` 过滤
7. 生成结果列表

返回的每条结果包含：

- `chunk_id`
- `source`
- `source_type`
- `page`
- `score`
- `snippet`

`snippet` 会被截断到 280 字左右，用于下游 prompt 和响应展示。

### 4. Graph 中的消费方式

工具返回结果进入 `ObserveNode` 后，会被写入：

- `retrieved_chunks`
- `tool_result`
- `tool_success`

随后 `respond` 阶段会把这些 chunks 带进回复 prompt，最终 `response` 节点会把它们映射到 `grounding.sources`。

## 配置项

当前与 RAG 直接相关的配置位于 `app/core/config.py`：

| 配置项 | 作用 |
| --- | --- |
| `KG_DIR` | 知识库目录 |
| `RAG_ENABLED` | 是否开启 RAG |
| `RAG_TOP_K` | 默认召回条数 |
| `RAG_MIN_SCORE` | 最低分阈值 |
| `RAG_CHUNK_SIZE` | 切块大小 |
| `RAG_CHUNK_OVERLAP` | 切块重叠 |
| `RAG_REFRESH_INTERVAL_SECONDS` | 兼容保留字段，当前未真正使用 |
| `KG_AUTO_BOOTSTRAP` | KG 不存在时是否自动生成 bootstrap |

额外要注意的是：

- embedding 模型、base URL、API key 目前仍在 `app/rag/retriever.py` 顶部常量与环境变量中定义
- 这部分还没有完全并入统一 `Settings`

## 与 Tavily 的边界

当前项目同时有两类外部知识来源：

### 1. 本地 RAG

适合：

- 稳定知识
- 教材/百科类内容
- 项目内可控语料

特点：

- 本地缓存
- 可重复
- 可解释来源
- 不依赖实时联网结果

### 2. Tavily 联网搜索

适合：

- 时间敏感问题
- 当下新闻、天气、政策、近期活动
- 本地 KG 不覆盖的开放信息

在 Graph 中，两者都归类为“知识检索动作”，但语义不同：

- `retrieve_knowledge` 是本地知识
- `tavily_search` 是联网搜索

## 当前边界与限制

### 1. Embedding 可用性是单点前置条件

如果在线 embedding 配置不可用：

- 本地 RAG 会直接退化为空结果
- 但主服务仍可继续回答，只是失去 grounding

### 2. 没有自动增量更新

当前缓存命中后不会自动比较源文件变化，因此 KG 改动后索引可能不是最新的。

### 3. 切块策略比较粗粒度

当前是按固定字符窗口切块，没有：

- 标题感知
- 段落感知
- 表格感知
- 语义边界切块

### 4. PDF 支持依赖 `pypdf`

若环境未安装 `pypdf`，PDF 会被直接跳过。

### 5. 代码里仍存在调试输出

`retrieve()` 当前保留了 `print()` 调试输出，后续如果面向生产环境，建议收敛到结构化日志。

## 调试建议

排查 RAG 问题时，建议优先看：

1. `index_status`
2. `embedding_available`
3. `embedding_error`
4. `chunks_total`
5. `grounding.sources`

常见排查路径：

1. 确认 `RAG_ENABLED=true`
2. 确认 `KG/` 下存在 `.txt` 或 `.pdf`
3. 确认 `RAG_embedding_model_key` 已配置
4. 确认本机安装了 `faiss-cpu`
5. 若有 PDF，确认安装了 `pypdf`
6. 若 KG 已变更，确认是否需要手动刷新缓存
