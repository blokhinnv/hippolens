"""HippoLens wrapper (core implementation in Phase A)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hipporag import HippoRAG


class HippoLens:
    """Wrapper over HippoRAG that exposes full PageRank scores for visualization."""

    def __init__(self, hipporag: HippoRAG) -> None:
        self._hr = hipporag
