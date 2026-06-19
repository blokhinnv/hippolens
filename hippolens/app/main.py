"""Streamlit entrypoint for HippoLens visualization."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_SAVE_DIR = _REPO_ROOT / "examples" / "demo_index"


def _parse_save_dir() -> Path:
    args = sys.argv[1:]
    if "--save-dir" in args:
        idx = args.index("--save-dir")
        if idx + 1 < len(args):
            return Path(args[idx + 1]).expanduser().resolve()
    return _DEFAULT_SAVE_DIR


def _init_lens(save_dir: Path):
    from examples.demo_config import load_demo_env, make_demo_config
    from hippolens import HippoLens
    from hipporag import HippoRAG

    load_demo_env()
    config = make_demo_config(str(save_dir))
    hr = HippoRAG(global_config=config)
    return HippoLens(hr)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")

    st.set_page_config(
        page_title="HippoLens",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    save_dir = _parse_save_dir()

    if not save_dir.is_dir():
        st.error(f"Index directory not found: `{save_dir}`. Run `examples/index_corpus.py` first.")
        st.stop()

    st.title("HippoLens")
    st.caption(f"Index: `{save_dir}`")

    if "lens" not in st.session_state:
        with st.spinner("Loading index…"):
            try:
                st.session_state.lens = _init_lens(save_dir)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

    lens = st.session_state.lens

    if "selected_node_id" not in st.session_state:
        st.session_state.selected_node_id = None
    if "path_nodes" not in st.session_state:
        st.session_state.path_nodes = set()
    if "path_edges" not in st.session_state:
        st.session_state.path_edges = set()
    if "path_phrases" not in st.session_state:
        st.session_state.path_phrases = []

    with st.sidebar:
        st.header("Graph")
        graph_mode = st.radio("Graph view", ["subgraph", "full"], horizontal=True)
        top_n_phrases = st.number_input("Subgraph phrases (N)", min_value=5, max_value=100, value=30)
        top_m_passages = st.number_input("Subgraph passages (M)", min_value=5, max_value=100, value=20)

        st.divider()
        st.markdown("**Legend**")
        st.markdown("- **Phrases** (inner): blue by PPR")
        st.markdown("- **Passages** (outer): red/orange by PPR")
        st.markdown("- **Size** = seed weight")
        st.markdown("- Dashed edges: phrase↔phrase · Dotted: phrase↔passage")

    query = st.text_input("Query", placeholder="What is HippoRAG?")
    top_k = st.slider("Top-k passages", min_value=1, max_value=50, value=20)
    retrieve_clicked = st.button("Retrieve", type="primary")

    if retrieve_clicked:
        if not query.strip():
            st.error("Enter a query.")
        else:
            with st.spinner("Retrieving…"):
                try:
                    result = lens.retrieve_lens(
                        query.strip(),
                        num_to_retrieve=top_k,
                        graph_mode=graph_mode,  # type: ignore[arg-type]
                        top_n_phrases=int(top_n_phrases),
                        top_m_passages=int(top_m_passages),
                    )
                except Exception as exc:
                    st.error(f"Retrieve failed: {exc}")
                    st.stop()
                st.session_state.lens_result = result
                st.session_state.selected_node_id = None
                st.session_state.path_nodes = set()
                st.session_state.path_edges = set()
                st.session_state.path_phrases = []

    result = st.session_state.get("lens_result")
    if result is None:
        st.info("Enter a query and click **Retrieve**.")
        st.stop()

    from hippolens.graph_export import warn_large

    if warn_large(result.nodes):
        st.warning(f"Large graph: {len(result.nodes)} nodes — rendering may be slow.")

    if result.retrieval_mode == "dpr_fallback":
        st.warning("DPR fallback — no PPR graph; path highlight disabled.")

    graph_col, panel_col = st.columns([3, 2])
    graph_height = 620
    # Two subheaders above scroll areas; split remaining height evenly.
    panel_section_height = (graph_height - 56) // 2

    with graph_col:
        from hippolens.app.components.graph import render_lens_graph

        path_enabled = result.retrieval_mode == "ppr"
        render_lens_graph(
            result,
            top_k=top_k,
            selected_node_id=st.session_state.selected_node_id,
            path_nodes=st.session_state.path_nodes if path_enabled else set(),
            path_edges=st.session_state.path_edges if path_enabled else set(),
            height=graph_height,
        )

    with panel_col:
        from hippolens.app.components.rank_panel import node_by_id, render_rank_panel

        picked = render_rank_panel(
            result,
            top_k,
            st.session_state.selected_node_id,
            section_height=panel_section_height,
        )
        if picked:
            if picked == st.session_state.selected_node_id:
                st.session_state.selected_node_id = None
                st.session_state.path_nodes = set()
                st.session_state.path_edges = set()
                st.session_state.path_phrases = []
            else:
                st.session_state.selected_node_id = picked
                _update_path_highlight(lens, result, picked)
            st.rerun()

        if st.session_state.path_phrases:
            st.subheader("Path phrases")
            for phrase in st.session_state.path_phrases:
                st.markdown(f"- {phrase}")

        selected = st.session_state.selected_node_id
        if selected:
            node = node_by_id(result, selected)
            if node:
                st.subheader("Selected node")
                st.markdown(f"**{node.node_type}** · PPR `{node.score:.4f}`")
                st.write(node.content)


def _update_path_highlight(lens, result, node_id: str | None) -> None:
    from hippolens.path_highlight import find_paths_to_passage, path_elements

    st.session_state.path_nodes = set()
    st.session_state.path_edges = set()
    st.session_state.path_phrases = []

    if not node_id or result.retrieval_mode != "ppr":
        return

    node = next((n for n in result.nodes if n.id == node_id), None)
    if node is None or node.node_type != "passage":
        return

    phrase_seeds = [
        sid
        for sid in result.seed_node_ids
        if any(n.id == sid and n.node_type == "phrase" for n in result.nodes)
    ]
    paths = find_paths_to_passage(lens._hr, node_id, phrase_seeds)
    nodes, edges = path_elements(paths)
    st.session_state.path_nodes = nodes
    st.session_state.path_edges = edges
    st.session_state.path_phrases = [
        next((n.content for n in result.nodes if n.id == nid), nid)
        for nid in nodes
        if any(n.id == nid and n.node_type == "phrase" for n in result.nodes)
    ]


def _render_metadata(result) -> None:
    st.subheader("Metadata")
    mode_label = "PPR" if result.retrieval_mode == "ppr" else "DPR fallback"
    if result.retrieval_mode == "dpr_fallback":
        st.error(f"Mode: {mode_label}")
    else:
        st.markdown(f"Mode: **{mode_label}**")
    st.markdown(f"Graph: **{result.graph_mode}** ({len(result.nodes)} nodes)")
    st.markdown(f"Seeds: **{len(result.seed_node_ids)}**")
    st.markdown(f"Facts: **{len(result.facts_used)}**")
    timing_parts = [f"{k}: {v * 1000:.0f}ms" for k, v in result.timings.items()]
    st.markdown("Timings: " + ", ".join(timing_parts))


if __name__ == "__main__":
    main()
