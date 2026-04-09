from __future__ import annotations

PARENT_SUMMARY_INSTRUCTION = (
    "你是一位专业的儿童发展助手，正在为家长撰写孩子的学习报告。"
    "用中文生成结构化摘要，语气专业、友好。"
    "Output plain text only. Do not output JSON, markdown, or code fences."
)


def build_parent_summary_user_prompt(
    has_memory: bool,
    memory_texts: list[str],
    child_profile_id: str,
) -> str:
    if not has_memory:
        return f"孩子（{child_profile_id}）目前还没有学习记录，请告知家长暂时无法生成摘要。"
    joined = "\n".join(f"- {t}" for t in memory_texts)
    return (
        f"以下是孩子（{child_profile_id}）与小智互动的记忆记录：\n\n"
        f"{joined}\n\n"
        "请生成结构化摘要，包含：\n"
        "1. 近期感兴趣的话题\n"
        "2. 常问的问题类型\n"
        "3. 学习观察与建议"
    )
