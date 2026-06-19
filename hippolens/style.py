"""Visual encoding for graph nodes and edges (paper-style)."""

from __future__ import annotations

from typing import Literal

from hippolens.models import GraphEdge, GraphNode

NODE_BORDER_WIDTH = 1.0
NODE_BORDER_COLOR = "#000000"
SEED_BORDER_WIDTH = 2.0
PHRASE_SIZE_MIN = 10.0
PHRASE_SIZE_MAX = 26.0
PASSAGE_SIZE_MIN = 12.0
PASSAGE_SIZE_MAX = 30.0

EdgeType = Literal["phrase_phrase", "phrase_passage"]


def _normalize(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return (value - vmin) / (vmax - vmin)


def _gradient_color(t: float, light: tuple[int, int, int], dark: tuple[int, int, int]) -> str:
    t = max(0.0, min(1.0, t))
    r = int(light[0] + (dark[0] - light[0]) * t)
    g = int(light[1] + (dark[1] - light[1]) * t)
    b = int(light[2] + (dark[2] - light[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def phrase_ppr_color(t: float) -> str:
    return _gradient_color(t, (0xE3, 0xF2, 0xFD), (0x0D, 0x47, 0xA1))


def passage_ppr_color(t: float) -> str:
    return _gradient_color(t, (0xFF, 0xE0, 0xB2), (0xC6, 0x28, 0x28))


def _size_in_group(
    value: float,
    vmin: float,
    vmax: float,
    size_min: float,
    size_max: float,
) -> float:
    t = _normalize(value, vmin, vmax)
    return size_min + t * (size_max - size_min)


def encode_node_style(
    node: GraphNode,
    phrase_ppr_min: float,
    phrase_ppr_max: float,
    passage_ppr_min: float,
    passage_ppr_max: float,
    phrase_seed_min: float,
    phrase_seed_max: float,
    passage_seed_min: float,
    passage_seed_max: float,
) -> dict[str, float | str]:
    if node.node_type == "phrase":
        ppr_t = _normalize(node.pagerank, phrase_ppr_min, phrase_ppr_max)
        color = phrase_ppr_color(ppr_t)
        size = _size_in_group(
            node.seed_weight,
            phrase_seed_min,
            phrase_seed_max,
            PHRASE_SIZE_MIN,
            PHRASE_SIZE_MAX,
        )
    else:
        ppr_t = _normalize(node.pagerank, passage_ppr_min, passage_ppr_max)
        color = passage_ppr_color(ppr_t)
        size = _size_in_group(
            node.seed_weight,
            passage_seed_min,
            passage_seed_max,
            PASSAGE_SIZE_MIN,
            PASSAGE_SIZE_MAX,
        )

    border_width = SEED_BORDER_WIDTH if node.is_seed else NODE_BORDER_WIDTH
    return {
        "width": size,
        "height": size,
        "background-color": color,
        "border-width": border_width,
        "border-color": NODE_BORDER_COLOR,
    }


def classify_edge(
    source: GraphNode,
    target: GraphNode,
) -> EdgeType:
    if source.node_type == "phrase" and target.node_type == "phrase":
        return "phrase_phrase"
    return "phrase_passage"


def encode_edge_style(
    edge_type: EdgeType,
    *,
    source: GraphNode | None = None,
    target: GraphNode | None = None,
) -> dict[str, float | str | list[int]]:
    if edge_type == "phrase_phrase":
        style: dict[str, float | str | list[int]] = {
            "line-color": "#616161",
            "line-style": "dashed",
            "width": 1,
            "opacity": 0.7,
        }
    else:
        style = {
            "line-color": "#424242",
            "line-style": "dotted",
            "width": 1.2,
            "opacity": 0.75,
        }

    if source and target and (source.is_seed or target.is_seed):
        style["line-style"] = "solid"
        style["width"] = float(style["width"]) + 0.5
        style["line-color"] = "#212121"
        style["opacity"] = 0.9

    return style


def style_edges(
    edges: list[GraphEdge],
    nodes_by_id: dict[str, GraphNode],
) -> list[GraphEdge]:
    styled: list[GraphEdge] = []
    for edge in edges:
        source = nodes_by_id.get(edge.source)
        target = nodes_by_id.get(edge.target)
        if source is None or target is None:
            styled.append(edge)
            continue
        edge_type = classify_edge(source, target)
        styled.append(
            GraphEdge(
                source=edge.source,
                target=edge.target,
                weight=edge.weight,
                edge_type=edge_type,
                style=encode_edge_style(edge_type, source=source, target=target),
            )
        )
    return styled
