from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.config import settings


def _ensure_data_dir() -> None:
    Path(settings.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)


def get_sync_checkpointer() -> SqliteSaver:
    _ensure_data_dir()
    return SqliteSaver.from_conn_string(str(settings.sqlite_db_path))


def get_async_checkpointer() -> AsyncSqliteSaver:
    _ensure_data_dir()
    return AsyncSqliteSaver.from_conn_string(str(settings.sqlite_db_path))
