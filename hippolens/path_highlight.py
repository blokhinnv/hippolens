"""Shortest-path highlighting from seed phrases to passages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hipporag import HippoRAG


def find_paths_to_passage(
    hipporag: HippoRAG,
    passage_id: str,
    seed_ids: list[str],
) -> list[list[str]]:
    hr = hipporag
    if passage_id not in hr.node_name_to_vertex_idx:
        return []

    passage_idx = hr.node_name_to_vertex_idx[passage_id]
    paths: list[list[str]] = []

    for seed_id in seed_ids:
        if seed_id not in hr.node_name_to_vertex_idx:
            continue
        seed_idx = hr.node_name_to_vertex_idx[seed_id]
        if seed_idx == passage_idx:
            paths.append([seed_id])
            continue
        try:
            vertex_path = hr.graph.get_shortest_paths(
                seed_idx,
                to=passage_idx,
                mode="undirected",
                output="vpath",
            )[0]
        except Exception:
            continue
        if not vertex_path:
            continue
        paths.append([hr.graph.vs[idx]["name"] for idx in vertex_path])

    return paths


def path_elements(
    paths: list[list[str]],
) -> tuple[set[str], set[tuple[str, str]]]:
    nodes: set[str] = set()
    edges: set[tuple[str, str]] = set()

    for path in paths:
        nodes.update(path)
        for left, right in zip(path, path[1:]):
            edges.add(tuple(sorted((left, right))))

    return nodes, edges
