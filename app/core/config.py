from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


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


_load_local_dotenv()


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_first_nonempty(*names: str) -> str | None:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None:
            continue
        value = raw_value.strip()
        if value:
            return value
    return None


def _read_int(default: int, *names: str) -> int:
    raw_value = _read_first_nonempty(*names)
    if raw_value is None:
        return default
    return int(raw_value)


def _read_openai_planning_model() -> str:
    planning_model = _read_first_nonempty("LLM_PLANNING_MODEL", "OPENAI_PLANNING_MODEL")
    if planning_model:
        return planning_model
    return _read_first_nonempty("LLM_MODEL", "OPENAI_MODEL") or "Qwen3-235B-A22B"


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "XiaoZhi API"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    langsmith_tracing: bool = field(
        default_factory=lambda: _read_bool("LANGSMITH_TRACING", False)
    )
    langsmith_api_key: str | None = field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY")
    )
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "XiaoZhi")
    )
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com",
        )
    )
    openai_api_key: str | None = field(
        default_factory=lambda: _read_first_nonempty("LLM_API_KEY", "OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: _read_first_nonempty("LLM_MODEL", "OPENAI_MODEL")
        or "Qwen3-235B-A22B"
    )
    openai_planning_model: str = field(default_factory=_read_openai_planning_model)
    openai_api_base: str = field(
        default_factory=lambda: _read_first_nonempty("LLM_BASE_URL", "OPENAI_API_BASE")
        or "https://api.scnet.cn/api/llm/v1"
    )
    openai_max_concurrency: int = field(
        default_factory=lambda: _read_int(1, "LLM_MAX_CONCURRENCY", "OPENAI_MAX_CONCURRENCY")
    )
    openai_image_request_timeout_seconds: int = field(
        default_factory=lambda: _read_int(
            120,
            "LLM_IMAGE_REQUEST_TIMEOUT_SECONDS",
            "OPENAI_IMAGE_REQUEST_TIMEOUT_SECONDS",
        )
    )
    max_upload_image_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_IMAGE_BYTES", str(4 * 1024 * 1024)))
    )
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )
    vllm_api_key: str | None = field(
        default_factory=lambda: _read_first_nonempty("vllm_api_key", "LLM_API_KEY")
    )
    @property
    def llm_api_key(self) -> str | None:
        return self.openai_api_key

    @property
    def llm_model(self) -> str:
        return self.openai_model

    @property
    def llm_planning_model(self) -> str:
        return self.openai_planning_model

    @property
    def llm_base_url(self) -> str:
        return self.openai_api_base

    @property
    def llm_max_concurrency(self) -> int:
        return self.openai_max_concurrency

    @property
    def llm_image_request_timeout_seconds(self) -> int:
        return self.openai_image_request_timeout_seconds

    # Backward compatibility for temporary uppercase references.
    @property
    def LLM_API_KEY(self) -> str | None:
        return self.llm_api_key

    @property
    def LLM_MODEL(self) -> str:
        return self.llm_model

    @property
    def LLM_PLANNING_MODEL(self) -> str:
        return self.llm_planning_model

    @property
    def LLM_BASE_URL(self) -> str:
        return self.llm_base_url

    @property
    def LLM_MAX_CONCURRENCY(self) -> int:
        return self.llm_max_concurrency

    @property
    def LLM_IMAGE_REQUEST_TIMEOUT_SECONDS(self) -> int:
        return self.llm_image_request_timeout_seconds

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
