import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Aliyun DashScope RPM limit: slow down to avoid 429
REQUEST_INTERVAL_SECONDS = 2

_PROMPT = (
    "你是一个8岁的好奇小孩。\n"
    "将下面这个关于自然科学的问题，改写为你会自然说出的口语问法。\n"
    "要求：\n"
    "- 用词简单，口语化，15字以内\n"
    "- 保留核心主题，但可以换表述方式\n"
    "- 可以加语气词（呀、啊、嘛、呢）\n"
    "- 不要解释，只输出改写后的问题\n\n"
    "原始问题：{question}\n"
    "改写后："
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite evaluation questions into child-like phrasing")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).parent / "queries" / "animals.json",
        help="Path to queries JSON to rewrite (default: eval/queries/animals.json)",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="Backup path before rewriting (default: <queries_stem>_original.json)",
    )
    args = parser.parse_args()

    from langchain_openai import ChatOpenAI
    from app.core.config import get_settings

    settings = get_settings()
    queries_path = args.queries
    if not queries_path.is_absolute():
        queries_path = Path.cwd() / queries_path
    backup_path = args.backup or queries_path.with_name(f"{queries_path.stem}_original.json")
    if not backup_path.is_absolute():
        backup_path = Path.cwd() / backup_path

    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    backup_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"原始问题已备份 → {backup_path}\n")

    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=0.5,
    )

    for i, item in enumerate(queries, 1):
        original = item["question"]
        fuzzy = llm.invoke(_PROMPT.format(question=original)).content.strip()
        print(f"[{i}/{len(queries)}] {original!r}\n         → {fuzzy!r}")
        item["question"] = fuzzy
        if i < len(queries):
            time.sleep(REQUEST_INTERVAL_SECONDS)

    queries_path.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成！{len(queries)} 条问题已改写，{queries_path} 已更新。")


if __name__ == "__main__":
    main()
