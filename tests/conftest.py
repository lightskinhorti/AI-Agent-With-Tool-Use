from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())
os.environ.setdefault("SQLITE_DB_PATH", str(Path(tempfile.mkdtemp()) / "test_agent.db"))
os.environ.setdefault("MEMORY_DB_PATH", str(Path(tempfile.mkdtemp()) / "test_memory.db"))


@pytest.fixture
def tmp_chroma_dir(tmp_path):
    d = tmp_path / "chroma"
    d.mkdir()
    return str(d)


@pytest.fixture
def tmp_memory_db(tmp_path):
    return str(tmp_path / "memory.db")
