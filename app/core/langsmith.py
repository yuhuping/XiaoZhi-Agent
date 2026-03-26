from __future__ import annotations

import os

from app.core.config import Settings


def configure_langsmith(settings: Settings) -> None:
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"


def is_langsmith_enabled(settings: Settings) -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)
