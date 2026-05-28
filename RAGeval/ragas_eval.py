"""
RAGAS evaluation for XiaoZ RAG pipeline.
"""

from __future__ import annotations

import sys
import json
import asyncio
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ragas.embeddings import BaseRagasEmbeddings

ROOT = Path(__file__).resolve().parents[1]
# 允许直接用 `python eval/ragas_eval.py` 运行，同时还能 import app 目录下的项目代码。
sys.path.insert(0, str(ROOT))

# 默认评估模型：专门用于 RAGAS 打分，和业务回答/查询扩展模型分开统计成本。
DEFAULT_JUDGE_MODEL = "qwen-max"

# 每跑完多少条样本，在终端打印一次阶段性均值。
BATCH_SIZE = 50

# baseline：原问题直接检索；multi_query：先扩展问题再检索；both：依次跑二者做 A/B 对比。
MODES = ("baseline", "multi_query", "both")

# RAGAS 原始指标名较长，终端打印时用短名便于观察。
SHORT = {
    "faithfulness": "faithful",
    "answer_relevancy": "relevant",
}


def log(message: str = "") -> None:
    """统一立即刷新日志，避免长时间 API 调用时终端看起来没有输出。"""
    print(message, flush=True)


def qwen_extra_body(model: str, enable_thinking: bool) -> dict | None:
    """支持混合思考的千问模型默认关闭思考模式，评估打分会更快。"""
    lower = model.lower()
    thinking_prefixes = ("qwen3", "qwen-flash", "qwen-turbo", "qwen-plus")
    return {"enable_thinking": enable_thinking} if lower.startswith(thinking_prefixes) else None


class RetrieverRagasEmbeddings(BaseRagasEmbeddings):
    """复用本项目检索器的 embedding 服务，供 AnswerRelevancy 计算问题相似度。"""

    def __init__(self, retriever: LocalKnowledgeRetriever):
        super().__init__()
        self.retriever = retriever

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.retriever._embed_texts(texts)
        if matrix.shape[0] != len(texts):
            raise RuntimeError("retriever embedding returned invalid shape")
        return matrix.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


def expand_query(question: str, llm: ChatOpenAI) -> list[str]:
    """为儿童口语问题额外生成最多 2 个更适合向量检索的短语。"""
    # 这里用 `|||` 做分隔符，是为了让模型输出容易 split，避免中文逗号/顿号不稳定。
    prompt = (
        "把儿童模糊问题改写成一个最可能的十万个为什么标题，用|||分隔，"
        "不要重复原问题，可以近似"
        "只输出标题，不要序号或其他内容。\n\n"
        f"问题：{question}\n可能的标题："
    )
    try:
        # 这一步属于“查询优化”，会额外消耗一次生成模型调用。
        raw = llm.invoke(prompt).content.strip()
        parts = [p.strip() for p in raw.split("|||") if p.strip()]
        # 原问题会在 retrieve_contexts() 里固定保留；这里最多只返回 2 个额外查询。
        return parts[:2]
    except Exception as exc:
        # 扩展失败时不追加额外查询；retrieve_contexts() 仍会使用原问题检索。
        log(f"    [warn] query expansion failed: {type(exc).__name__}: {exc}")
        return []


def retrieve_contexts(
    question: str,
    retriever: LocalKnowledgeRetriever,
    gen_llm: ChatOpenAI,
    top_k: int,
    min_score: float,
    mode: str,
) -> tuple[list[dict], list[str]]:
    """执行检索，返回「检索结果记录」和「实际使用的查询短语」。"""
    if mode == "baseline":
        # baseline 是对照组：完全不做查询扩展，直接拿用户问题去检索。
        results = retriever.retrieve(question, top_k=top_k, min_score=min_score)
        # RAGAS 要求 retrieved_contexts 是字符串列表；没有结果时放占位文本，方便后续流程继续跑。
        contexts = results if results else [{"snippet": "（无检索结果）"}]
        return contexts, [question]

    # multi_query 实验组：原问题 + 2 个 LLM 扩展问题 -> 分别检索 -> 按 chunk_id 去重合并。
    # 保留原问题很重要：儿童口语问法本身可能已经能召回正确 chunk，扩展查询只作为补充。
    sub_queries = [question]
    for expanded in expand_query(question, gen_llm):
        if expanded not in sub_queries:
            sub_queries.append(expanded)
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for sq in sub_queries:
        for r in retriever.retrieve(sq, top_k=top_k, min_score=min_score):
            cid = str(r.get("chunk_id", ""))
            if cid not in seen_ids:
                seen_ids.add(cid)
                merged.append(r)
    # 多个子查询的结果合并后，按向量相似度重新排序，再统一截断到 top_k。
    merged.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return merged[:top_k] or [{"snippet": "（无检索结果）"}], sub_queries


def build_answer(question: str, contexts: list[str], llm: ChatOpenAI) -> str:
    """用检索上下文生成待评估回答。"""
    # 只把前 5 个 context 放进 prompt，和检索 top_k 保持一致，控制输入长度和费用。
    ctx_text = "\n---\n".join(contexts[:5])
    prompt = (
        "根据以下参考资料，简洁回答问题。如果资料不够充分，结合你的知识作答。\n\n"
        f"参考资料：\n{ctx_text}\n\n"
        f"问题：{question}\n\n答："
    )
    return llm.invoke(prompt).content


async def score_sample(
    sample: SingleTurnSample,
    metrics: list,
    metric_timeout: float,
) -> dict[str, float | None]:
    """对单条问答样本并发计算 RAGAS 指标。"""

    async def score_metric(metric) -> tuple[str, float | None]:
        try:
            log(f"    scoring {metric.name} ...")
            # single_turn_ascore 是异步接口；同一样本内的多个指标可以并发等待。
            val = await asyncio.wait_for(
                metric.single_turn_ascore(sample),
                timeout=metric_timeout,
            )
            log(f"    scoring {metric.name} done")
            return metric.name, round(float(val), 4) if val is not None else None
        except asyncio.TimeoutError:
            log(f"    [warn] {metric.name} timed out after {metric_timeout:.0f}s")
            return metric.name, None
        except Exception as exc:
            # RAGAS 指标依赖 LLM 判断，偶发超时/解析失败时记为 None，均值计算会自动跳过。
            log(f"    [warn] {metric.name} failed: {type(exc).__name__}: {exc}")
            return metric.name, None

    metric_scores = await asyncio.gather(*(score_metric(metric) for metric in metrics))
    return dict(metric_scores)


def _write_result(path: Path, n_completed: int, n_total: int,
                  totals: dict, per_sample: list) -> None:
    """把当前进度写入 JSON；每条样本后都会覆盖一次，方便中断后查看已完成结果。"""
    # totals 只保存成功得到数值的指标，因此均值不会被 None 污染。
    mean = {k: round(sum(v) / len(v), 4) for k, v in totals.items() if v}
    report = {
        "last_updated": datetime.now().isoformat(),
        "n_completed": n_completed,
        "n_total": n_total,
        "mean_scores": mean,
        "per_sample": per_sample,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_batch_summary(batch_rows: list, batch_num: int, start: int, end: int, mode: str) -> None:
    """打印一个批次内的平均分，长任务运行时用于观察趋势。"""
    log("\n" + "═" * 60)
    log(f"  [{mode.upper()}] BATCH #{batch_num}  (queries {start}–{end})")
    metric_names = [k for k in batch_rows[0] if k in SHORT]
    for name in metric_names:
        vals = [r[name] for r in batch_rows if r.get(name) is not None]
        if vals:
            log(f"  {SHORT[name]:<12} {sum(vals)/len(vals):.4f}")
    log("═" * 60)


def _print_comparison(
    baseline_totals: dict[str, list[float]],
    mq_totals: dict[str, list[float]],
    n: int,
) -> None:
    """both 模式结束后，打印 baseline 和 multi_query 的最终均值差异。"""
    bm = {k: sum(v) / len(v) for k, v in baseline_totals.items() if v}
    mm = {k: sum(v) / len(v) for k, v in mq_totals.items() if v}
    w = 62

    log("\n\n" + "═" * w)
    log(f"  A/B COMPARISON  ({n} queries)")
    log(f"  baseline  vs  multi-query expansion")
    log("─" * w)
    log(f"  {'metric':<14}{'baseline':>10}{'multi_query':>14}{'Δ':>10}  ")
    log("─" * w)
    for name, short in SHORT.items():
        bv = bm.get(name)
        mv = mm.get(name)
        if bv is not None and mv is not None:
            delta = mv - bv
            arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "─")
            log(f"  {short:<14}{bv:>10.4f}{mv:>14.4f}{delta:>+10.4f}  {arrow}")
    log("═" * w)


async def run_mode(
    queries: list,
    retriever: LocalKnowledgeRetriever,
    gen_llm: ChatOpenAI,
    judge_llm: LangchainLLMWrapper,
    result_path: Path,
    mode: str,
    metric_timeout: float,
) -> dict[str, list[float]]:
    settings = get_settings()
    # RAGAS 在这里仅评估生成质量。
    # - Faithfulness：回答是否忠于检索上下文，主要看有没有幻觉。
    # - AnswerRelevancy：回答是否切题，主要看有没有正面回应用户问题。
    # 检索类指标（precision/recall/MRR）由 chunk_metrics.py 单独计算，避免重复消耗评判模型 token。
    ragas_embeddings = RetrieverRagasEmbeddings(retriever)
    metrics = [
        Faithfulness(llm=judge_llm),
        # RAGAS 默认 strictness=3，会为每条回答生成 3 个反推问题；这里用 1 控制评估耗时。
        AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings, strictness=1),
    ]
    metric_names = [m.name for m in metrics]

    totals: dict[str, list[float]] = defaultdict(list)
    per_sample: list[dict] = []
    n_total = len(queries)

    log(f"\n{'─' * 60}")
    log(f"  MODE: {mode.upper()}  ({n_total} queries)  →  {result_path.name}")
    log(f"{'─' * 60}")

    for i, item in enumerate(queries, 1):
        q = item["question"]
        log(f"\n[{i}/{n_total}] {q[:50]}")

        # Step 1：检索。baseline 和 multi_query 的区别只发生在这个函数里。
        log("  step1 retrieve contexts ...")
        context_records, sub_queries = retrieve_contexts(
            q, retriever, gen_llm,
            top_k=5, min_score=settings.rag_min_score, mode=mode,
        )
        # RAGAS 只需要纯文本 contexts；完整检索记录会另存到结果 JSON，方便排查低分样本。
        contexts = [str(r.get("snippet") or "") for r in context_records]
        if mode == "multi_query":
            log(f"  ↳ {' | '.join(sub_queries)}")
        log(f"  contexts: {len(contexts)}")

        # Step 2：基于检索结果生成回答。这个回答会被生成质量指标评估。
        log("  step2 generate answer ...")
        answer = build_answer(q, contexts, gen_llm)
        log(f"  answer preview: {answer[:80]}")

        # Step 3：组装 RAGAS 的单轮样本。
        # user_input 是问题，retrieved_contexts 是检索文本，response 是模型回答，reference 是标准答案。
        sample = SingleTurnSample(
            user_input=q,
            retrieved_contexts=contexts,
            response=answer,
            reference=item["ground_truth"],
        )

        # Step 4：调用评估模型打分。
        log("  step3 score generated answer ...")
        scores = await score_sample(sample, metrics, metric_timeout=metric_timeout)

        score_str = "  ".join(
            f"{SHORT[n]}={scores[n]:.4f}" if scores.get(n) is not None else f"{SHORT[n]}=N/A"
            for n in metric_names
        )
        log(f"  {score_str}")

        # 只把非 None 分数放进 totals；某个指标失败不会影响其它指标的均值。
        for name in metric_names:
            if scores.get(name) is not None:
                totals[name].append(scores[name])
        avg_str = "  ".join(
            f"{SHORT[n]}={sum(totals[n])/len(totals[n]):.3f}" if totals[n] else f"{SHORT[n]}=N/A"
            for n in metric_names
        )
        log(f"  avg → {avg_str}")

        # 保存单条样本详情：
        # - response_preview：只保留回答前 120 字，避免结果文件过大。
        # - retrieved_contexts：保留 chunk_id/source/score/snippet，用来排查生成回答是否依赖了正确上下文。
        per_sample.append({
            "index": i,
            "question": q,
            **({"reference_question": item["reference_question"]} if item.get("reference_question") else {}),
            "sub_queries": sub_queries,
            "response_preview": answer[:120],
            "n_contexts": len(contexts),
            "retrieved_contexts": context_records,
            **({"reference_chunk_ids": item["reference_chunk_ids"]} if item.get("reference_chunk_ids") else {}),
            **{n: scores.get(n) for n in metric_names},
        })

        # 每条样本完成后立即落盘，完整评估中断时也能拿到已完成部分。
        _write_result(result_path, i, n_total, totals, per_sample)

        if i % BATCH_SIZE == 0:
            batch_start = i - BATCH_SIZE + 1
            _print_batch_summary(
                per_sample[batch_start - 1 : i],
                i // BATCH_SIZE, batch_start, i, mode,
            )

    remainder = n_total % BATCH_SIZE
    if remainder:
        _print_batch_summary(
            per_sample[n_total - remainder :],
            (n_total // BATCH_SIZE) + 1,
            n_total - remainder + 1, n_total,
            mode,
        )

    return totals


async def run(
    queries: list,
    retriever: LocalKnowledgeRetriever,
    gen_llm: ChatOpenAI,
    judge_llm: LangchainLLMWrapper,
    result_dir: Path,
    mode: str,
    metric_timeout: float,
    result_tag: str = "",
) -> None:
    suffix = f"_{result_tag}" if result_tag else ""
    if mode == "both":
        # A/B 对比：同一批 queries 先跑 baseline，再跑 multi_query，最后比较均值。
        baseline_path = result_dir / f"result_baseline{suffix}.json"
        mq_path = result_dir / f"result_multi_query{suffix}.json"
        baseline_totals = await run_mode(
            queries, retriever, gen_llm, judge_llm, baseline_path, "baseline", metric_timeout
        )
        mq_totals = await run_mode(
            queries, retriever, gen_llm, judge_llm, mq_path, "multi_query", metric_timeout
        )
        _print_comparison(baseline_totals, mq_totals, len(queries))
        log(f"\nDone. Reports:\n  {baseline_path}\n  {mq_path}")
    else:
        # 单模式运行时只生成一个 result_{mode}.json。
        result_path = result_dir / f"result_{mode}{suffix}.json"
        await run_mode(queries, retriever, gen_llm, judge_llm, result_path, mode, metric_timeout)
        log(f"\nDone. Full report at: {result_path}")


def main() -> None:
    # 命令行参数：
    # - queries 指定评估集文件，默认 eval/queries/animals.json。
    # - subset 用于省钱快速试跑。
    # - mode 控制检索策略。
    # - gen-model 指定回答生成/查询扩展模型，默认读取 .env 的 LLM_MODEL。
    # - judge-model 指定 RAGAS 评判模型，默认 qwen-max。
    # - result-tag 给结果文件名加后缀，避免不同评估集互相覆盖结果。
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on XiaoZ RAG pipeline")
    parser.add_argument("--queries", type=Path, default=Path(__file__).parent / "queries" / "animals.json",
                        help="Path to evaluation queries JSON (default: eval/queries/animals.json)")
    parser.add_argument("--reference-queries", type=Path, default=None,
                        help="Optional JSON file providing ground_truth/reference by matching index")
    parser.add_argument("--result-tag", default="",
                        help="Suffix tag for result files, e.g. plants -> result_baseline_plants.json")
    parser.add_argument("--subset", type=int, default=None, metavar="N",
                        help="Evaluate only the first N queries")
    parser.add_argument("--mode", choices=MODES, default="baseline",
                        help="Retrieval mode: baseline | multi_query | both (default: baseline)")
    parser.add_argument("--gen-model", default=None,
                        help="LLM used for answer generation and query expansion (default: .env LLM_MODEL)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"LLM used by RAGAS metrics (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--request-timeout", type=float, default=None,
                        help="Timeout in seconds for one LLM HTTP request (default: REQUEST_TIMEOUT_SECONDS)")
    parser.add_argument("--metric-timeout", type=float, default=180.0,
                        help="Timeout in seconds for one RAGAS metric on one sample (default: 180)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen thinking mode for gen/judge models (default: disabled for speed)")
    
    args = parser.parse_args()

    global ChatOpenAI
    global Faithfulness, AnswerRelevancy
    global LangchainLLMWrapper, SingleTurnSample
    global LocalKnowledgeRetriever, get_settings

    from langchain_openai import ChatOpenAI
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper

    from app.core.config import get_settings
    from app.rag.retriever import LocalKnowledgeRetriever

    settings = get_settings()

    # queries 文件中每条数据需要包含 question 和 ground_truth。
    # gen_fuzzy_queries.py 跑完后，question 会变成儿童口语化问法，ground_truth 保持标准答案。
    queries_path = args.queries
    if not queries_path.is_absolute():
        queries_path = Path.cwd() / queries_path
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    reference_queries_path = args.reference_queries
    if reference_queries_path is not None and not reference_queries_path.is_absolute():
        reference_queries_path = Path.cwd() / reference_queries_path
    if reference_queries_path is not None:
        reference_queries = json.loads(reference_queries_path.read_text(encoding="utf-8"))
        if len(reference_queries) < len(queries):
            raise SystemExit(
                f"--reference-queries has fewer rows ({len(reference_queries)}) "
                f"than --queries ({len(queries)})"
            )
        merged_queries = []
        for item, reference_item in zip(queries, reference_queries):
            merged = dict(item)
            merged["ground_truth"] = reference_item["ground_truth"]
            if reference_item.get("question"):
                merged["reference_question"] = reference_item["question"]
            if reference_item.get("reference_chunk_ids"):
                merged["reference_chunk_ids"] = reference_item["reference_chunk_ids"]
            merged_queries.append(merged)
        queries = merged_queries

    if args.subset:
        queries = queries[: args.subset]
    gen_model = args.gen_model or settings.llm_model
    request_timeout = args.request_timeout or settings.request_timeout_seconds

    log(
        f"Loaded {len(queries)} queries from {queries_path}  |  "
        f"reference: {reference_queries_path or queries_path}  |  "
        f"gen: {gen_model}  |  judge: {args.judge_model}  |  "
        f"mode: {args.mode}  |  request_timeout: {request_timeout}s  |  "
        f"metric_timeout: {args.metric_timeout}s  |  "
        f"qwen_thinking: {'on' if args.enable_thinking else 'off'}"
    )

    # 复用项目内的本地知识库检索器；auto_bootstrap=False 表示 KG 缺失时不自动塞示例数据。
    log("Initializing retriever ...")
    retriever = LocalKnowledgeRetriever.from_kg_dir(
        kg_dir=settings.kg_dir,
        auto_bootstrap=False,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        auto_refresh_interval_seconds=0,
    )
    log("Retriever ready.")

    # 业务生成模型：用于 multi_query 查询扩展，也用于根据 context 生成回答。
    log("Initializing LLM clients ...")
    gen_llm = ChatOpenAI(
        model=gen_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
        request_timeout=request_timeout,
        max_retries=1,
        extra_body=qwen_extra_body(gen_model, args.enable_thinking),
    )

    # RAGAS 评估模型：只负责给指标打分，不参与回答生成。
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=args.judge_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            temperature=0,
            request_timeout=request_timeout,
            max_retries=1,
            extra_body=qwen_extra_body(args.judge_model, args.enable_thinking),
        )
    )
    log("LLM clients ready.")

    # 结果目录在 .gitignore 中被忽略，避免把评估输出和 API 生成内容提交到仓库。
    result_dir = Path(__file__).parent / "results"
    result_dir.mkdir(exist_ok=True)

    asyncio.run(run(
        queries,
        retriever,
        gen_llm,
        judge_llm,
        result_dir,
        args.mode,
        args.metric_timeout,
        args.result_tag.strip(),
    ))


if __name__ == "__main__":
    main()
