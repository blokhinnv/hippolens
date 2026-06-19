"""Two-orbit radial layout for phrase and passage nodes."""

from __future__ import annotations

import math

from hippolens.models import GraphNode

R1 = 100.0
R2 = 200.0


def two_orbit_layout(nodes: list[GraphNode]) -> dict[str, tuple[float, float]]:
    phrases = [n for n in nodes if n.node_type == "phrase"]
    passages = [n for n in nodes if n.node_type == "passage"]
    positions: dict[str, tuple[float, float]] = {}

    for idx, node in enumerate(phrases):
        angle = 2 * math.pi * idx / max(len(phrases), 1)
        positions[node.id] = (R1 * math.cos(angle), R1 * math.sin(angle))

    for idx, node in enumerate(passages):
        angle = 2 * math.pi * idx / max(len(passages), 1)
        positions[node.id] = (R2 * math.cos(angle), R2 * math.sin(angle))

    return positions
