"""Export HippoRAG graph structures for visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hippolens.models import GraphEdge, GraphNode
from hippolens.path_highlight import find_paths_to_passage

if TYPE_CHECKING:
    from hipporag import HippoRAG

MAX_SUBGRAPH_NODES = 200


def build_all_nodes(
    hipporag: HippoRAG,
    ppr_scores: np.ndarray | None,
    node_weights: np.ndarray | None,
) -> list[GraphNode]:
    hr = hipporag
    weights = node_weights if node_weights is not None else np.zeros(hr.graph.vcount())
    passage_ids = set(hr.passage_node_keys)
    nodes: list[GraphNode] = []

    for vertex_idx, name in enumerate(hr.graph.vs["name"]):
        seed_weight = float(weights[vertex_idx])
        is_seed = seed_weight > 0
        pagerank = float(ppr_scores[vertex_idx]) if ppr_scores is not None else 0.0

        if name in passage_ids:
            content = hr.chunk_embedding_store.get_row(name)["content"]
            node_type = "passage"
        else:
            content = hr.entity_embedding_store.get_row(name)["content"]
            node_type = "phrase"

        label = content[:40] + ("…" if len(content) > 40 else "")
        nodes.append(
            GraphNode(
                id=name,
                node_type=node_type,
                label=label,
                content=content,
                pagerank=pagerank,
                seed_weight=seed_weight,
                is_seed=is_seed,
            )
        )

    return nodes


def build_edges(hipporag: HippoRAG, node_ids: set[str]) -> list[GraphEdge]:
    """Build edges from the igraph backbone (includes passage↔phrase links)."""
    hr = hipporag
    seen: set[tuple[str, str]] = set()
    edges: list[GraphEdge] = []

    for edge in hr.graph.es:
        source = hr.graph.vs[edge.source]["name"]
        target = hr.graph.vs[edge.target]["name"]
        if source not in node_ids or target not in node_ids:
            continue
        key = tuple(sorted((source, target)))
        if key in seen:
            continue
        seen.add(key)
        weight = float(edge["weight"]) if "weight" in edge.attributes() else 1.0
        edges.append(GraphEdge(source=source, target=target, weight=weight))

    return edges


def export_subgraph(
    all_nodes: list[GraphNode],
    all_edges: list[GraphEdge],
    *,
    hipporag: HippoRAG,
    seed_node_ids: list[str],
    top_n_phrases: int = 30,
    top_m_passages: int = 20,
    max_nodes: int = MAX_SUBGRAPH_NODES,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    selected_ids: set[str] = set(seed_node_ids)

    phrases = sorted(
        (n for n in all_nodes if n.node_type == "phrase"),
        key=lambda n: n.pagerank,
        reverse=True,
    )
    passages = sorted(
        (n for n in all_nodes if n.node_type == "passage"),
        key=lambda n: n.pagerank,
        reverse=True,
    )

    for node in phrases[:top_n_phrases]:
        selected_ids.add(node.id)
    top_passage_ids = [n.id for n in passages[:top_m_passages]]
    for node_id in top_passage_ids:
        selected_ids.add(node_id)

    phrase_seed_ids = [
        node_id
        for node_id in seed_node_ids
        if any(n.id == node_id and n.node_type == "phrase" for n in all_nodes)
    ]
    for passage_id in top_passage_ids:
        for path in find_paths_to_passage(hipporag, passage_id, phrase_seed_ids):
            selected_ids.update(path)

    if len(selected_ids) > max_nodes:
        ranked = sorted(
            (n for n in all_nodes if n.id in selected_ids),
            key=lambda n: (n.is_seed, n.pagerank),
            reverse=True,
        )
        selected_ids = {n.id for n in ranked[:max_nodes]}

    nodes = [n for n in all_nodes if n.id in selected_ids]
    edges = [e for e in all_edges if e.source in selected_ids and e.target in selected_ids]
    return nodes, edges


def export_full(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    return nodes, edges


def warn_large(nodes: list[GraphNode]) -> bool:
    return len(nodes) > 500
