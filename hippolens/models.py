"""Data models for HippoLens retrieval and graph visualization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass
class GraphNode:
    id: str
    node_type: Literal["phrase", "passage"]
    label: str
    content: str
    pagerank: float
    seed_weight: float
    is_seed: bool
    x: float | None = None
    y: float | None = None
    style: dict[str, Any] | None = None


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float
    edge_type: Literal["phrase_phrase", "phrase_passage"] | None = None
    style: dict[str, Any] | None = None


@dataclass
class RankedItem:
    id: str
    node_type: Literal["phrase", "passage"]
    content: str
    score: float
    rank: int


@dataclass
class LensQueryResult:
    question: str
    graph_mode: Literal["full", "subgraph"]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    ranked_passages: list[RankedItem]
    ranked_phrases: list[RankedItem]
    seed_node_ids: list[str]
    linking_score_map: dict[str, float]
    facts_used: list[tuple[Any, ...]]
    retrieval_mode: Literal["ppr", "dpr_fallback"]
    timings: dict[str, float]
    default_top_k: int = 20

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "graph_mode": self.graph_mode,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "ranked_passages": [asdict(r) for r in self.ranked_passages],
            "ranked_phrases": [asdict(r) for r in self.ranked_phrases],
            "seed_node_ids": self.seed_node_ids,
            "linking_score_map": self.linking_score_map,
            "facts_used": [list(f) for f in self.facts_used],
            "retrieval_mode": self.retrieval_mode,
            "timings": self.timings,
            "default_top_k": self.default_top_k,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LensQueryResult:
        return cls(
            question=data["question"],
            graph_mode=data["graph_mode"],
            nodes=[GraphNode(**{k: v for k, v in n.items() if k in GraphNode.__dataclass_fields__}) for n in data["nodes"]],
            edges=[
                GraphEdge(**{k: v for k, v in e.items() if k in GraphEdge.__dataclass_fields__})
                for e in data["edges"]
            ],
            ranked_passages=[RankedItem(**r) for r in data["ranked_passages"]],
            ranked_phrases=[RankedItem(**r) for r in data["ranked_phrases"]],
            seed_node_ids=data["seed_node_ids"],
            linking_score_map={k: float(v) for k, v in data["linking_score_map"].items()},
            facts_used=[tuple(f) for f in data["facts_used"]],
            retrieval_mode=data["retrieval_mode"],
            timings={k: float(v) for k, v in data["timings"].items()},
            default_top_k=int(data.get("default_top_k", 20)),
        )
