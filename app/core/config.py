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


def _read_float(default: float, *names: str) -> float:
    raw_value = _read_first_nonempty(*names)
    if raw_value is None:
        return default
    return float(raw_value)


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
    llm_api_key: str | None = field(
        default_factory=lambda: _read_first_nonempty("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_model: str = field(
        default_factory=lambda: _read_first_nonempty("LLM_MODEL", "OPENAI_MODEL")
        or "MiniMax-M2.5"
    )
    # openai_planning_model: str = field(default_factory=_read_openai_planning_model)
    llm_base_url: str = field(
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

    vllm_base_url: str | None = field(
        default_factory=lambda: _read_first_nonempty("vllm_base_url", "LLM_BASE_URL")
    )
    vllm_api_key: str | None = field(
        default_factory=lambda: _read_first_nonempty("vllm_api_key", "LLM_API_KEY")
    )
    vllm_model: str = field(
        default_factory=lambda: _read_first_nonempty("vllm_model", "LLM_MODEL") or "hunyuan-t1-vision-20250916"
    )
    
    # RAG相关参数
    kg_dir: str = field(default_factory=lambda: os.getenv("KG_DIR", str(ROOT_DIR / "KG")))
    rag_enabled: bool = field(default_factory=lambda: _read_bool("RAG_ENABLED", True))
    rag_top_k: int = field(default_factory=lambda: _read_int(3, "RAG_TOP_K"))
    rag_min_score: float = field(default_factory=lambda: _read_float(0.08, "RAG_MIN_SCORE"))
    rag_chunk_size: int = field(default_factory=lambda: _read_int(520, "RAG_CHUNK_SIZE"))
    rag_chunk_overlap: int = field(default_factory=lambda: _read_int(80, "RAG_CHUNK_OVERLAP"))
    rag_refresh_interval_seconds: int = field(
        default_factory=lambda: _read_int(2, "RAG_REFRESH_INTERVAL_SECONDS")
    )
    kg_auto_bootstrap: bool = field(default_factory=lambda: _read_bool("KG_AUTO_BOOTSTRAP", True))
    profile_db_path: str = field(
        default_factory=lambda: os.getenv(
            "PROFILE_DB_PATH",
            str(ROOT_DIR / "data" / "profile_memory.sqlite3"),
        )
    )
    memory_db_path: str = field(
        default_factory=lambda: _read_first_nonempty("MEMORY_DB_PATH", "PROFILE_DB_PATH")
        or str(ROOT_DIR / "data" / "memory.sqlite3")
    )
    memory_index_dir: str = field(
        default_factory=lambda: os.getenv(
            "MEMORY_INDEX_DIR",
            str(ROOT_DIR / "data" / "memory_index"),
        )
    )
    memory_working_capacity: int = field(
        default_factory=lambda: _read_int(50, "MEMORY_WORKING_CAPACITY")
    )
    memory_working_ttl_minutes: int = field(
        default_factory=lambda: _read_int(60, "MEMORY_WORKING_TTL_MINUTES")
    )
    memory_consolidate_working_threshold: float = field(
        default_factory=lambda: _read_float(0.7, "MEMORY_CONSOLIDATE_WORKING_THRESHOLD")
    )
    memory_consolidate_episodic_threshold: float = field(
        default_factory=lambda: _read_float(0.8, "MEMORY_CONSOLIDATE_EPISODIC_THRESHOLD")
    )
    memory_auto_consolidate_enabled: bool = field(
        default_factory=lambda: _read_bool("MEMORY_AUTO_CONSOLIDATE_ENABLED", False)
    )
    memory_write_perceptual_enabled: bool = field(
        default_factory=lambda: _read_bool("MEMORY_WRITE_PERCEPTUAL_ENABLED", False)
    )
    memory_forget_max_age_days: int = field(
        default_factory=lambda: _read_int(30, "MEMORY_FORGET_MAX_AGE_DAYS")
    )
    memory_reset_on_start: bool = field(
        default_factory=lambda: _read_bool("MEMORY_RESET_ON_START", True)
    )

    # Tavily搜索相关参数
    tavily_api_key: str | None = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    tavily_base_url: str = field(
        default_factory=lambda: os.getenv("TAVILY_BASE_URL", "https://api.tavily.com")
    )
    tavily_max_results: int = field(default_factory=lambda: _read_int(3, "TAVILY_MAX_RESULTS"))
    tavily_search_depth: str = field(
        default_factory=lambda: os.getenv("TAVILY_SEARCH_DEPTH", "basic")
    )
    tavily_timeout_seconds: int = field(
        default_factory=lambda: _read_int(15, "TAVILY_TIMEOUT_SECONDS")
    )
    # @property
    # def llm_api_key(self) -> str | None:
    #     return self.openai_api_key

    # @property
    # def llm_model(self) -> str:
    #     return self.openai_model

    # @property
    # def llm_planning_model(self) -> str:
    #     return self.openai_planning_model

    # @property
    # def llm_base_url(self) -> str:
    #     return self.openai_api_base

    # @property
    # def llm_max_concurrency(self) -> int:
    #     return self.openai_max_concurrency

    # @property
    # def llm_image_request_timeout_seconds(self) -> int:
    #     return self.openai_image_request_timeout_seconds

    # # Backward compatibility for temporary uppercase references.
    # @property
    # def LLM_API_KEY(self) -> str | None:
    #     return self.llm_api_key

    # @property
    # def LLM_MODEL(self) -> str:
    #     return self.llm_model

    # @property
    # def LLM_PLANNING_MODEL(self) -> str:
    #     return self.llm_planning_model

    # @property
    # def LLM_BASE_URL(self) -> str:
    #     return self.llm_base_url

    # @property
    # def LLM_MAX_CONCURRENCY(self) -> int:
    #     return self.llm_max_concurrency

    # @property
    # def LLM_IMAGE_REQUEST_TIMEOUT_SECONDS(self) -> int:
    #     return self.llm_image_request_timeout_seconds

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
