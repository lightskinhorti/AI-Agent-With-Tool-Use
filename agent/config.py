from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    chroma_persist_dir: Path = Path("./data/chroma")
    sqlite_db_path: Path = Path("./data/agent.db")
    memory_db_path: Path = Path("./data/memory.db")
    log_level: str = "INFO"
    agent_model: str = "claude-3-5-sonnet-20241022"
    agent_max_steps: int = 15
    hitl_timeout_seconds: int = 300

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
