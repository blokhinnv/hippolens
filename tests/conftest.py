"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEMO_INDEX = _REPO_ROOT / "examples" / "demo_index"
_DEMO_GRAPH = (
    _DEMO_INDEX
    / "openai_gpt-4o-mini_openai_text-embedding-3-small"
    / "graph.pickle"
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: tests requiring API key and demo index")


@pytest.fixture(scope="module")
def demo_hr():
    if not _DEMO_GRAPH.exists():
        pytest.skip("demo index not built — run examples/index_corpus.py first")

    from examples.demo_config import load_demo_env, make_demo_config

    load_demo_env()
    from hipporag import HippoRAG

    return HippoRAG(global_config=make_demo_config(str(_DEMO_INDEX)))


@pytest.fixture(scope="module")
def lens(demo_hr):
    from hippolens import HippoLens

    return HippoLens(demo_hr)
