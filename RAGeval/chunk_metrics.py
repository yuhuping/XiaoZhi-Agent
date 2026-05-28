from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"reference_chunk_ids must be a string or list, got {type(value).__name__}")

    ids: list[str] = []
    seen: set[str] = set()
    for item in value:
        chunk_id = str(item).strip()
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ids.append(chunk_id)
    return ids


def default_result_output_path(results_path: Path, k: int) -> Path:
    return results_path.with_name(f"{results_path.stem}_chunk_metrics_at_{k}.json")


def default_direct_output_path(eval_queries_path: Path, mode: str, k: int) -> Path:
    return ROOT / "eval" / "results" / f"chunk_metrics_{eval_queries_path.stem}_{mode}_at_{k}.json"


def score_row(
    row: dict[str, Any],
    query_item: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    gold_ids = normalize_ids(row.get("reference_chunk_ids") or query_item.get("reference_chunk_ids"))
    if not gold_ids:
        raise ValueError("missing reference_chunk_ids")

    retrieved_ids = [
        str(ctx.get("chunk_id") or "").strip()
        for ctx in row.get("retrieved_contexts", [])
        if str(ctx.get("chunk_id") or "").strip()
    ]
    top_ids = retrieved_ids[:k]

    gold_set = set(gold_ids)
    hit_ids = []
    seen_hits: set[str] = set()
    for chunk_id in top_ids:
        if chunk_id in gold_set and chunk_id not in seen_hits:
            seen_hits.add(chunk_id)
            hit_ids.append(chunk_id)

    first_hit_rank = next(
        (rank for rank, chunk_id in enumerate(top_ids, start=1) if chunk_id in gold_set),
        None,
    )

    hit_count = len(hit_ids)
    return {
        "index": row.get("index"),
        "question": row.get("question") or query_item.get("question"),
        "reference_chunk_ids": gold_ids,
        "retrieved_chunk_ids_at_k": top_ids,
        f"chunk_precision@{k}": round(hit_count / k, 4),
        f"chunk_recall@{k}": round(hit_count / len(gold_set), 4),
        f"mrr@{k}": round((1 / first_hit_rank) if first_hit_rank else 0.0, 4),
        f"hit@{k}": 1.0 if first_hit_rank else 0.0,
    }


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def metric_keys(k: int) -> tuple[str, str, str, str]:
    return f"chunk_precision@{k}", f"chunk_recall@{k}", f"mrr@{k}", f"hit@{k}"


def build_report(
    rows: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    k: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    precision_key, recall_key, mrr_key, hit_key = metric_keys(k)
    return {
        "last_updated": datetime.now().isoformat(),
        **metadata,
        "k": k,
        "n_evaluated": len(rows),
        "n_missing_reference_chunk_ids": len(missing),
        "mean_scores": {
            precision_key: mean(rows, precision_key),
            recall_key: mean(rows, recall_key),
            mrr_key: mean(rows, mrr_key),
            hit_key: mean(rows, hit_key),
        },
        "per_sample": rows,
        "missing_reference_chunk_ids": missing,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_path}")
    for key, value in report["mean_scores"].items():
        print(f"{key}: {value}")
    if report.get("n_missing_reference_chunk_ids"):
        print(f"missing_reference_chunk_ids: {report['n_missing_reference_chunk_ids']}")


def run_from_results(args: argparse.Namespace) -> None:
    results_path = args.results.resolve()
    queries_path = args.queries.resolve()
    output_path = (args.output.resolve() if args.output else default_result_output_path(results_path, args.k))

    result = load_json(results_path)
    queries = load_json(queries_path)
    if not isinstance(queries, list):
        raise SystemExit("--queries must point to a JSON list")

    samples = result.get("per_sample", []) if isinstance(result, dict) else []
    if not samples:
        raise SystemExit("--results must contain a non-empty per_sample list")

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in samples:
        index = int(row.get("index") or 0)
        if index < 1 or index > len(queries):
            raise SystemExit(f"result row has invalid index: {index}")

        query_item = queries[index - 1]
        try:
            rows.append(score_row(row, query_item, args.k))
        except ValueError as exc:
            if "missing reference_chunk_ids" not in str(exc):
                raise
            missing.append({
                "index": index,
                "question": row.get("question") or query_item.get("question"),
            })

    if missing and not args.allow_missing:
        preview = "\n".join(f"  #{item['index']} {item['question']}" for item in missing[:10])
        raise SystemExit(
            "Missing reference_chunk_ids. Add reference_chunk_ids to the query JSON first.\n"
            f"Missing rows: {len(missing)}\n{preview}"
        )

    report = build_report(rows, missing, args.k, {
        "results_path": str(results_path),
        "queries_path": str(queries_path),
        "source": "ragas_result",
    })
    write_report(report, output_path)


def create_retriever(settings):
    from app.rag.retriever import LocalKnowledgeRetriever

    return LocalKnowledgeRetriever.from_kg_dir(
        kg_dir=settings.kg_dir,
        auto_bootstrap=False,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        auto_refresh_interval_seconds=0,
    )


def create_query_llm(settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="qwen-max",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0,
    )


def run_direct_retrieval(args: argparse.Namespace) -> None:
    from app.core.config import get_settings
    from ragas_eval import retrieve_contexts

    if args.mode not in {"baseline", "multi_query"}:
        raise SystemExit("--mode is required and must be baseline or multi_query in direct retrieval mode")

    gold_queries_path = args.gold_queries.resolve()
    eval_queries_path = args.queries.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else default_direct_output_path(eval_queries_path, args.mode, args.k)
    )

    gold_queries = load_json(gold_queries_path)
    eval_queries = load_json(eval_queries_path)
    if not isinstance(gold_queries, list) or not isinstance(eval_queries, list):
        raise SystemExit("--gold-queries and --queries must point to JSON lists")
    if len(gold_queries) != len(eval_queries):
        raise SystemExit(
            f"query count mismatch: gold={len(gold_queries)}, eval={len(eval_queries)}"
        )

    settings = get_settings()
    retriever = create_retriever(settings)
    gen_llm = create_query_llm(settings) if args.mode == "multi_query" else None
    min_score = settings.rag_min_score if args.min_score is None else args.min_score

    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for index, (gold_item, eval_item) in enumerate(zip(gold_queries, eval_queries), start=1):
        gold_question = str(gold_item.get("question") or "")
        eval_question = str(eval_item.get("question") or "")

        gold_contexts, _ = retrieve_contexts(
            gold_question,
            retriever,
            gen_llm,
            top_k=args.gold_top_k,
            min_score=min_score,
            mode="baseline",
        )
        gold_ids = [
            str(ctx.get("chunk_id") or "").strip()
            for ctx in gold_contexts
            if str(ctx.get("chunk_id") or "").strip()
        ]

        eval_contexts, sub_queries = retrieve_contexts(
            eval_question,
            retriever,
            gen_llm,
            top_k=args.k,
            min_score=min_score,
            mode=args.mode,
        )

        row = {
            "index": index,
            "question": eval_question,
            "gold_question": gold_question,
            "sub_queries": sub_queries,
            "retrieved_contexts": eval_contexts,
            "reference_chunk_ids": gold_ids,
        }

        if not gold_ids:
            missing.append({
                "index": index,
                "question": eval_question,
                "gold_question": gold_question,
            })
            continue

        scored = score_row(row, {"question": eval_question, "reference_chunk_ids": gold_ids}, args.k)
        scored.update({
            "gold_question": gold_question,
            "sub_queries": sub_queries,
            "gold_contexts": gold_contexts,
        })
        rows.append(scored)

        score_preview = " ".join(
            f"{key}={scored[key]:.4f}" for key in metric_keys(args.k)[:3]
        )
        print(f"[{index}/{len(eval_queries)}] {eval_question[:40]}  {score_preview}")

    if missing and not args.allow_missing:
        preview = "\n".join(f"  #{item['index']} {item['gold_question']}" for item in missing[:10])
        raise SystemExit(
            "Some gold queries did not retrieve any chunk ids.\n"
            f"Missing rows: {len(missing)}\n{preview}"
        )

    report = build_report(rows, missing, args.k, {
        "source": "direct_retrieval",
        "mode": args.mode,
        "gold_queries_path": str(gold_queries_path),
        "eval_queries_path": str(eval_queries_path),
        "gold_top_k": args.gold_top_k,
        "min_score": min_score,
    })
    write_report(report, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute chunk-id retrieval metrics"
    )
    parser.add_argument("--results", type=Path, default=None,
                        help="Optional result_*.json generated by eval/ragas_eval.py")
    parser.add_argument("--queries", type=Path, required=True,
                        help="Evaluation queries JSON")
    parser.add_argument("--gold-queries", type=Path, default=None,
                        help="Original/clear queries JSON used to retrieve ground-truth chunks")
    parser.add_argument("--mode", choices=("baseline", "multi_query"), default=None,
                        help="Retrieval mode for --queries when using --gold-queries")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k cutoff for chunk metrics (default: 5)")
    parser.add_argument("--gold-top-k", type=int, default=1,
                        help="Top-k chunks from --gold-queries treated as ground truth (default: 1)")
    parser.add_argument("--min-score", type=float, default=None,
                        help="Retriever min score (default: settings.rag_min_score)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path")
    parser.add_argument("--allow-missing", action="store_true",
                        help="Skip rows without gold/reference chunk ids instead of failing")
    args = parser.parse_args()

    if args.k <= 0:
        raise SystemExit("--k must be positive")
    if args.gold_top_k <= 0:
        raise SystemExit("--gold-top-k must be positive")

    if args.results and args.gold_queries:
        raise SystemExit("Use either --results or --gold-queries, not both")
    if args.results:
        run_from_results(args)
    elif args.gold_queries:
        run_direct_retrieval(args)
    else:
        raise SystemExit("Provide either --results or --gold-queries")


if __name__ == "__main__":
    main()
