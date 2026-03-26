#!/usr/bin/env python
"""Simple connectivity tester for OpenAI-compatible LLM APIs (SCNet recommended).

Examples:
  python scripts/test_llm_api.py
  python scripts/test_llm_api.py --model Qwen3-235B-A22B --timeout 60
  python scripts/test_llm_api.py --base-url https://api.scnet.cn/api/llm/v1 --api-key sk-xxxx
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from urllib import error, request


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_local_dotenv() -> None:
    dotenv_path = ROOT_DIR / ".env"
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _first_nonempty(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _extract_output_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                chunks.append(part["text"].strip())
    return "\n".join(chunk for chunk in chunks if chunk)


def _http_json(method: str, url: str, api_key: str, timeout: int, payload: dict | None = None) -> tuple[int, dict, float]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        },
    )

    started = time.time()
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        elapsed = time.time() - started
        return resp.status, json.loads(body), elapsed


def main() -> int:
    _load_local_dotenv()

    parser = argparse.ArgumentParser(description="Test an OpenAI-compatible LLM API endpoint.")
    parser.add_argument(
        "--base-url",
        default=_first_nonempty("LLM_BASE_URL", "OPENAI_API_BASE") or "https://api.scnet.cn/api/llm/v1",
        help="API base URL, e.g. https://api.scnet.cn/api/llm/v1",
    )
    parser.add_argument(
        "--api-key",
        default=_first_nonempty("LLM_API_KEY", "OPENAI_API_KEY"),
        help="API key. If omitted, reads LLM_API_KEY then OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=_first_nonempty("LLM_MODEL", "OPENAI_MODEL") or "Qwen3-235B-A22B",
        help="Model name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="HTTP timeout seconds for each request.",
    )
    parser.add_argument(
        "--prompt",
        default="Explain what an apple is for a 5-year-old child in 2 short sentences.",
        help="Prompt text for response test.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: missing API key. Set --api-key or LLM_API_KEY (or OPENAI_API_KEY).", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    print(f"[config] base_url={base_url}")
    print(f"[config] model={args.model}")
    print(f"[config] timeout={args.timeout}s")

    try:
        status, models_payload, elapsed = _http_json(
            method="GET",
            url=f"{base_url}/models",
            api_key=args.api_key,
            timeout=args.timeout,
        )
        model_count = len(models_payload.get("data", [])) if isinstance(models_payload, dict) else 0
        print(f"[models] status={status} elapsed={elapsed:.2f}s count={model_count}")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"[models] HTTP {exc.code}: {detail[:300]}")
        return 1
    except Exception as exc:
        print(f"[models] error: {exc}")
        return 1

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": args.prompt}
        ],
        "max_tokens": 256,
        "stream": False,
    }
    url = f"{base_url}/chat/completions"

    try:
        status, response_payload, elapsed = _http_json(
            method="POST",
            url=url,
            api_key=args.api_key,
            timeout=args.timeout,
            payload=payload,
        )
        print(response_payload["choices"][0]["message"]["content"])
        text = _extract_output_text(response_payload)
        print(f"[responses] status={status} elapsed={elapsed:.2f}s")
        print(f"[responses] id={response_payload.get('id', 'unknown')} status={response_payload.get('status', 'unknown')}")
        
        if text:
            print("[responses] output_preview=")
            print(text[:400])
        else:
            print("[responses] output_preview=<empty>")
        return 0
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        print(f"[responses] HTTP {exc.code}: {detail[:500]}")
        return 1
    except Exception as exc:
        print(f"[responses] error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
