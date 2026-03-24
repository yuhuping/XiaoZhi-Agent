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


@dataclass(frozen=True)
class Settings:
    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "XiaoZhi API"))
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    host: str = field(default_factory=lambda: os.getenv("APP_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    )
    openai_api_base: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_API_BASE",
            "https://api2.codexcn.com/v1",
        )
    )
    mock_mode: bool = field(default_factory=lambda: _read_bool("MOCK_MODEL", True))
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
