# RAGAS Evaluation - XiaoZ RAG Pipeline

目标：对比「直接检索」与「查询优化 + 多查询扩展」两种策略的 RAG 效果。

评估维度：

- `ragas_eval.py`: 计算生成质量指标 `Faithfulness / AnswerRelevancy`
- `chunk_precision@5 / chunk_recall@5 / mrr@5`

## Step 1: 模糊化问题

将规范问题改写为儿童口语化表述，使评估更贴近真实用户提问场景。

```bash
# 改写动物卷 animals.json（默认）
python eval/gen_fuzzy_queries.py

# 等价写法
python eval/gen_fuzzy_queries.py \
  --queries eval/queries/animals.json \
  --backup eval/queries/animals_original.json

# 改写植物卷 plants.json
python eval/gen_fuzzy_queries.py \
  --queries eval/queries/plants.json \
  --backup eval/queries/plants_original.json

# 改写人体卷 human.json
python eval/gen_fuzzy_queries.py \
  --queries eval/queries/human.json \
  --backup eval/queries/human_original.json

# 改写生活卷 life.json
python eval/gen_fuzzy_queries.py \
  --queries eval/queries/life.json \
  --backup eval/queries/life_original.json
```

## Step 2: 运行 RAGAS 生成质量 A/B 评估

```bash
# 快速冒烟测试（5 条，省 API 费）
python eval/ragas_eval.py --mode both --subset 5

# 完整评估（默认 animals.json，全部 50 条）
python eval/ragas_eval.py --mode both

# 单独跑某一模式（可开两个终端并行跑，比 --mode both 更快）
python eval/ragas_eval.py --mode baseline
python eval/ragas_eval.py --mode multi_query
```

显式指定评估集和评判模型：

```bash
# 动物卷（当前建议只跑这一卷，RAGAS 比 chunk_metrics 更耗 token）
python eval/ragas_eval.py \
  --mode baseline \
  --queries eval/queries/animals.json \
  --reference-queries eval/queries/animals_original.json \
  --gen-model MiniMax-M2.1 \
  --judge-model qwen-plus-2025-12-01 \
  --request-timeout 60 \
  --metric-timeout 180 \
  --result-tag animals

python eval/ragas_eval.py \
  --mode multi_query \
  --queries eval/queries/animals.json \
  --reference-queries eval/queries/animals_original.json \
  --gen-model MiniMax-M2.1 \
  --judge-model qwen-plus-2025-12-01 \
  --request-timeout 60 \
  --metric-timeout 180 \
  --result-tag animals

# 植物卷
python eval/ragas_eval.py \
  --mode baseline \
  --queries eval/queries/plants.json \
  --judge-model qwen3.6-flash \
  --result-tag plants

python eval/ragas_eval.py \
  --mode multi_query \
  --queries eval/queries/plants.json \
  --judge-model qwen3.6-flash \
  --result-tag plants

# 人体卷
python eval/ragas_eval.py \
  --mode baseline \
  --queries eval/queries/human.json \
  --judge-model qwen3.6-flash \
  --result-tag human

python eval/ragas_eval.py \
  --mode multi_query \
  --queries eval/queries/human.json \
  --judge-model qwen3.6-flash \
  --result-tag human

# 生活卷
python eval/ragas_eval.py \
  --mode baseline \
  --queries eval/queries/life.json \
  --judge-model qwen3.6-flash \
  --result-tag life

python eval/ragas_eval.py \
  --mode multi_query \
  --queries eval/queries/life.json \
  --judge-model qwen3.6-flash \
  --result-tag life
```

## 模式说明

- `baseline`: 直接将用户问题送入检索器。
- `multi_query`: 使用「原问题 + LLM 额外扩展的 2 个精准检索短语」共 3 个查询，逐一检索后按 `chunk_id` 去重合并，再排序取 `top-k`。
- `both`: 依次运行 `baseline` 和 `multi_query`，最终打印对比表格。

做 A/B 对比时，`baseline` 和 `multi_query` 应使用同一份 queries 文件。例如都用 `animals.json`，或都用 `plants.json`。如果一个用 `animals.json`，另一个用 `plants.json`，最终均值不能公平比较。

## 输出文件

`eval/results/` 已被 `.gitignore` 忽略。

- `result_baseline.json`: 默认动物卷 baseline 报告
- `result_multi_query.json`: 默认动物卷 multi-query 报告
- `result_baseline_animals.json`: 加 `--result-tag animals` 后的动物卷 baseline 报告
- `result_multi_query_animals.json`: 加 `--result-tag animals` 后的动物卷 multi-query 报告
- `result_baseline_plants.json`: 加 `--result-tag plants` 后的植物卷 baseline 报告
- `result_multi_query_plants.json`: 加 `--result-tag plants` 后的植物卷 multi-query 报告
- `result_baseline_human.json`: 加 `--result-tag human` 后的人体卷 baseline 报告
- `result_multi_query_human.json`: 加 `--result-tag human` 后的人体卷 multi-query 报告
- `result_baseline_life.json`: 加 `--result-tag life` 后的生活卷 baseline 报告
- `result_multi_query_life.json`: 加 `--result-tag life` 后的生活卷 multi-query 报告

终端最终会打印类似：

```text
══════════════════════════════════════════════════════════════
  A/B COMPARISON  (50 queries)
  baseline  vs  multi-query expansion
──────────────────────────────────────────────────────────────
  metric          baseline  multi_query         Δ
──────────────────────────────────────────────────────────────
  faithful          0.7500        0.8200    +0.0700  ▲
  relevant          0.8100        0.8500    +0.0400  ▲
══════════════════════════════════════════════════════════════
```

## Chunk ID 快速检索指标

用 `animals_original.json` 的规范问题先检索出 ground-truth chunk，再用 `animals.json` 的口语化问题评估 `baseline / multi_query`：

```bash
python eval/chunk_metrics.py \
  --gold-queries eval/queries/animals_original.json \
  --queries eval/queries/animals.json \
  --mode baseline

python eval/chunk_metrics.py \
  --gold-queries eval/queries/animals_original.json \
  --queries eval/queries/animals.json \
  --mode multi_query
```

其它卷只需要替换文件名：

```bash
# 植物卷
python eval/chunk_metrics.py \
  --gold-queries eval/queries/plants_original.json \
  --queries eval/queries/plants.json \
  --mode baseline

python eval/chunk_metrics.py \
  --gold-queries eval/queries/plants_original.json \
  --queries eval/queries/plants.json \
  --mode multi_query

# 人体卷
python eval/chunk_metrics.py \
  --gold-queries eval/queries/human_original.json \
  --queries eval/queries/human.json \
  --mode baseline

python eval/chunk_metrics.py \
  --gold-queries eval/queries/human_original.json \
  --queries eval/queries/human.json \
  --mode multi_query

# 生活卷
python eval/chunk_metrics.py \
  --gold-queries eval/queries/life_original.json \
  --queries eval/queries/life.json \
  --mode baseline

python eval/chunk_metrics.py \
  --gold-queries eval/queries/life_original.json \
  --queries eval/queries/life.json \
  --mode multi_query
```

如果 queries 文件中已经手动标注 `reference_chunk_ids`，也可以基于 `ragas_eval.py` 的结果文件直接计算：

```bash
python eval/chunk_metrics.py \
  --results eval/results/result_baseline.json \
  --queries eval/queries/animals.json
```

## 文件结构

```text
eval/
├── README.md                本说明文件
├── queries/                 评估问答集
│   ├── animals.json          动物卷问答对（50 条，Step 1 后为口语化版本）
│   ├── animals_original.json 动物卷原始规范问题备份
│   ├── plants.json           植物卷问答对（50 条）
│   ├── plants_original.json  植物卷原始规范问题备份
│   ├── human.json            人体卷问答对（50 条）
│   ├── human_original.json   人体卷原始规范问题备份
│   ├── life.json             生活卷问答对（50 条）
│   └── life_original.json    生活卷原始规范问题备份
├── gen_fuzzy_queries.py     Step 1 脚本：改写问题为儿童口语
├── ragas_eval.py            Step 2 脚本：RAGAS 生成质量评估 + A/B 对比
├── chunk_metrics.py         Chunk ID 快速检索指标计算
└── results/                 运行后自动生成（已 gitignore）
    ├── result_baseline.json
    └── result_multi_query.json
```

## 依赖

```bash
pip install -r requirements.txt
```

所需 `.env` 变量：

- `LLM_API_KEY / LLM_BASE_URL / LLM_MODEL`: 答案生成 + 查询扩展
- `RAG_embedding_model_key`: FAISS 向量检索

业务回答生成和查询扩展默认使用 `.env` 里的 `LLM_MODEL`。如果要单独指定生成模型，可加：

```bash
python eval/ragas_eval.py \
  --mode baseline \
  --queries eval/queries/animals.json \
  --reference-queries eval/queries/animals_original.json \
  --gen-model your-answer-model \
  --judge-model qwen-flash \
  --request-timeout 60 \
  --metric-timeout 180 \
  --result-tag animals
```

评判模型默认是 `qwen-max`，但完整评估建议显式指定更快的 `qwen-flash` 或 `qwen3.5-flash`。脚本会对 `qwen*` 模型默认关闭思考模式；如果确实需要思考模式，可加 `--enable-thinking`。注意：如果模型控制台开启了 “free tier only” 且免费额度耗尽，会出现 `AllocationQuota.FreeTierOnly`，这时需要换模型或在控制台关闭该限制。
