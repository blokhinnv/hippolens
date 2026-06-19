"""Ranked passages and phrases panel."""

from __future__ import annotations

import streamlit as st

from hippolens.models import LensQueryResult, RankedItem


def _render_ranked_items(
    items: list[RankedItem],
    selected_node_id: str | None,
    key_prefix: str,
    content_limit: int,
) -> str | None:
    new_selection: str | None = None
    for item in items:
        label = f"**{item.rank}.** [{item.score:.4f}] {item.content[:content_limit]}"
        if item.id == selected_node_id:
            st.markdown(f"▶ {label}")
        elif st.button(label, key=f"{key_prefix}-{item.id}", use_container_width=True):
            new_selection = item.id
    return new_selection


def render_rank_panel(
    result: LensQueryResult,
    top_k: int,
    selected_node_id: str | None,
    *,
    section_height: int,
) -> str | None:
    """Render passage/phrase lists; return newly selected node id if any."""
    new_selection: str | None = None

    st.subheader("Ranked Passages")
    with st.container(height=section_height, border=True):
        picked = _render_ranked_items(
            result.ranked_passages[:top_k],
            selected_node_id,
            "passage",
            120,
        )
        if picked:
            new_selection = picked

    st.subheader("Phrases (by PPR)")
    with st.container(height=section_height, border=True):
        picked = _render_ranked_items(
            result.ranked_phrases[:20],
            selected_node_id,
            "phrase",
            80,
        )
        if picked:
            new_selection = picked

    return new_selection


def node_by_id(result: LensQueryResult, node_id: str) -> RankedItem | None:
    for item in result.ranked_passages + result.ranked_phrases:
        if item.id == node_id:
            return item
    for node in result.nodes:
        if node.id == node_id:
            return RankedItem(
                id=node.id,
                node_type=node.node_type,
                content=node.content,
                score=node.pagerank,
                rank=0,
            )
    return None
