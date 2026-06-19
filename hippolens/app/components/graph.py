"""Cytoscape.js graph widget for Streamlit."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from hippolens.models import GraphEdge, GraphNode, LensQueryResult


def nodes_to_payload(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    path_nodes: set[str] | None = None,
    path_edges: set[tuple[str, str]] | None = None,
    top_k_passage_ids: list[str] | None = None,
    selected_node_id: str | None = None,
) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "node_type": n.node_type,
                "content": n.content,
                "pagerank": n.pagerank,
                "seed_weight": n.seed_weight,
                "is_seed": n.is_seed,
                "x": n.x,
                "y": n.y,
                "style": n.style or {},
            }
            for n in nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "weight": e.weight,
                "edge_type": e.edge_type,
                "style": e.style or {},
            }
            for e in edges
        ],
        "path_nodes": sorted(path_nodes or []),
        "path_edges": [list(pair) for pair in (path_edges or set())],
        "top_k_passage_ids": top_k_passage_ids or [],
        "selected_node_id": selected_node_id,
    }


def _build_html(payload: dict[str, Any], height: int) -> str:
    data = json.dumps(payload).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; overflow: hidden; background: #ffffff; }}
    #cy {{ width: 100%; height: {height}px; }}
    #tooltip {{
      position: fixed; z-index: 10; display: none;
      background: rgba(30,30,30,0.92); color: #fff;
      padding: 6px 10px; border-radius: 4px;
      font: 12px/1.4 -apple-system, BlinkMacSystemFont, sans-serif;
      max-width: 280px; pointer-events: none;
    }}
    #err {{ color: #b71c1c; padding: 12px; font: 14px sans-serif; }}
  </style>
</head>
<body>
  <div id="cy"></div>
  <div id="tooltip"></div>
  <script>
    const payload = {data};
    const tooltip = document.getElementById("tooltip");

    function buildElements(p) {{
      const nodes = (p.nodes || []).map((n) => ({{
        group: "nodes",
        data: {{
          id: n.id, label: n.label, node_type: n.node_type,
          content: n.content, pagerank: n.pagerank,
          seed_weight: n.seed_weight, is_seed: n.is_seed,
        }},
        position: {{ x: n.x || 0, y: n.y || 0 }},
      }}));
      const edges = (p.edges || []).map((e) => ({{
        group: "edges",
        data: {{
          id: e.source + "__" + e.target,
          source: e.source, target: e.target, weight: e.weight,
          edge_type: e.edge_type || "phrase_phrase",
        }},
      }}));
      return nodes.concat(edges);
    }}

    function applyHighlights(cy, p) {{
      const pathNodes = new Set(p.path_nodes || []);
      const pathEdges = new Set((p.path_edges || []).map((e) => e[0] + "__" + e[1]));
      const topK = new Set(p.top_k_passage_ids || []);
      const hasPath = pathNodes.size > 0;
      const hasTopK = topK.size > 0;
      const selected = p.selected_node_id;

      cy.nodes().forEach((node) => {{
        const id = node.id();
        const isPath = pathNodes.has(id);
        const isTopK = topK.has(id);
        const isSelected = id === selected;
        let opacity = 1;
        if (hasPath && !isPath) opacity = 0.15;
        else if (hasTopK && node.data("node_type") === "passage" && !isTopK) opacity = 0.25;
        node.style("opacity", opacity);

        const nodeStyle = (p.nodes || []).find((n) => n.id === id)?.style || {{}};
        const baseBorderWidth = nodeStyle["border-width"] ?? 1;
        const baseBorderColor = nodeStyle["border-color"] ?? "#000000";
        if (isSelected) {{
          node.style("border-width", 3);
          node.style("border-color", "#D32F2F");
        }} else {{
          node.style("border-width", baseBorderWidth);
          node.style("border-color", baseBorderColor);
        }}
      }});

      cy.edges().forEach((edge) => {{
        const a = edge.source().id(), b = edge.target().id();
        const onPath = pathEdges.has(a + "__" + b) || pathEdges.has(b + "__" + a);
        if (onPath) {{
          edge.style("opacity", hasPath ? 0.95 : 0.6);
          edge.style("width", 3);
          edge.style("line-color", "#D32F2F");
          edge.style("line-style", "solid");
        }} else if (hasPath) {{
          edge.style("opacity", 0.1);
        }}
      }});
    }}

    try {{
      if (typeof cytoscape === "undefined") throw new Error("Cytoscape failed to load");

      const cy = cytoscape({{
        container: document.getElementById("cy"),
        elements: buildElements(payload),
        style: [
          {{
            selector: "node",
            style: {{
              label: "data(label)",
              "font-size": 8,
              "text-valign": "bottom",
              "text-margin-y": 4,
              "text-max-width": 80,
              "border-width": 1,
              "border-color": "#000000",
            }},
          }},
          {{
            selector: 'node[node_type = "phrase"]',
            style: {{ shape: "ellipse" }},
          }},
          {{
            selector: 'node[node_type = "passage"]',
            style: {{ shape: "ellipse" }},
          }},
          {{
            selector: 'edge[edge_type = "phrase_phrase"]',
            style: {{
              width: 1, "line-color": "#616161",
              "line-style": "dashed", "curve-style": "bezier", opacity: 0.7,
            }},
          }},
          {{
            selector: 'edge[edge_type = "phrase_passage"]',
            style: {{
              width: 1.2, "line-color": "#424242",
              "line-style": "dotted", "curve-style": "bezier", opacity: 0.75,
            }},
          }},
          {{
            selector: "edge",
            style: {{
              "curve-style": "bezier", opacity: 0.6,
            }},
          }},
        ],
        layout: {{ name: "preset" }},
        wheelSensitivity: 0.2,
      }});

      (payload.nodes || []).forEach((n) => {{
        const node = cy.getElementById(n.id);
        if (!node.length) return;
        Object.entries(n.style || {{}}).forEach(([k, v]) => node.style(k, v));
      }});

      (payload.edges || []).forEach((e) => {{
        const edge = cy.getElementById(e.source + "__" + e.target);
        if (!edge.length) return;
        Object.entries(e.style || {{}}).forEach(([k, v]) => edge.style(k, v));
      }});

      applyHighlights(cy, payload);

      cy.on("mouseover", "node", (evt) => {{
        const n = evt.target.data();
        tooltip.style.display = "block";
        tooltip.innerHTML =
          "<b>" + n.node_type + "</b><br/>PPR: " + Number(n.pagerank).toFixed(4) +
          "<br/>Seed: " + Number(n.seed_weight).toFixed(4) +
          "<br/>" + (n.content || "").slice(0, 120);
      }});
      cy.on("mousemove", (evt) => {{
        tooltip.style.left = (evt.originalEvent.clientX + 12) + "px";
        tooltip.style.top = (evt.originalEvent.clientY + 12) + "px";
      }});
      cy.on("mouseout", "node", () => {{ tooltip.style.display = "none"; }});

      cy.fit(undefined, 40);
    }} catch (err) {{
      document.getElementById("cy").innerHTML =
        '<p id="err">Graph render error: ' + err.message + "</p>";
    }}
  </script>
</body>
</html>"""


def render_graph(
    graph_payload: dict[str, Any],
    height: int = 600,
) -> None:
    """Render Cytoscape graph (display-only; selection via rank panel)."""
    st.iframe(_build_html(graph_payload, height), height=height)


def render_lens_graph(
    result: LensQueryResult,
    *,
    top_k: int,
    selected_node_id: str | None,
    path_nodes: set[str],
    path_edges: set[tuple[str, str]],
    height: int = 600,
) -> None:
    top_k_ids = [item.id for item in result.ranked_passages[:top_k]]
    payload = nodes_to_payload(
        result.nodes,
        result.edges,
        path_nodes=path_nodes,
        path_edges=path_edges,
        top_k_passage_ids=top_k_ids,
        selected_node_id=selected_node_id,
    )
    render_graph(payload, height=height)
