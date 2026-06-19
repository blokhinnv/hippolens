"""Phase A — HippoLens core wrapper tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hippolens.models import GraphEdge, GraphNode, LensQueryResult, RankedItem

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEMO_GRAPH = (
    _REPO_ROOT
    / "examples"
    / "demo_index"
    / "openai_gpt-4o-mini_openai_text-embedding-3-small"
    / "graph.pickle"
)


def test_lens_query_result_roundtrip() -> None:
    original = LensQueryResult(
        question="What is HippoRAG?",
        graph_mode="subgraph",
        nodes=[
            GraphNode(
                id="entity-abc",
                node_type="phrase",
                label="hippocampus",
                content="hippocampus",
                pagerank=0.05,
                seed_weight=1.0,
                is_seed=True,
            )
        ],
        edges=[GraphEdge(source="entity-abc", target="chunk-xyz", weight=1.0)],
        ranked_passages=[
            RankedItem(
                id="chunk-xyz",
                node_type="passage",
                content="The hippocampus stores memories.",
                score=0.12,
                rank=1,
            )
        ],
        ranked_phrases=[],
        seed_node_ids=["entity-abc"],
        linking_score_map={"hippocampus": 0.9},
        facts_used=[("hippocampus", "is", "brain region")],
        retrieval_mode="ppr",
        timings={"embedding": 0.1, "rerank": 0.2, "retrieval": 0.3},
        default_top_k=20,
    )

    restored = LensQueryResult.from_dict(json.loads(original.to_json()))
    assert restored == original


def test_hippolens_init_on_demo_index() -> None:
    if not _DEMO_GRAPH.exists():
        pytest.skip("demo index not built")

    from examples.demo_config import load_demo_env, make_demo_config
    from hippolens import HippoLens

    load_demo_env()

    from hipporag import HippoRAG

    hr = HippoRAG(global_config=make_demo_config(str(_REPO_ROOT / "examples" / "demo_index")))
    lens = HippoLens(hr)

    assert lens._last_ppr_scores is None
    assert lens._last_node_weights is None
    assert lens._last_linking_score_map == {}
    assert hr.ready_to_retrieve


@pytest.mark.integration
def test_retrieve_compat(lens, demo_hr) -> None:
    query = "What is HippoRAG and how does it retrieve information?"
    direct = demo_hr.retrieve([query])
    wrapped = lens.retrieve([query])

    assert wrapped[0].docs == direct[0].docs
    assert np.allclose(wrapped[0].doc_scores, direct[0].doc_scores)


@pytest.mark.integration
def test_ppr_scores_after_retrieve_lens(lens) -> None:
    lens.retrieve_lens("What is the hippocampus used for in the brain?")

    hr = lens._hr
    assert lens._last_ppr_scores is not None
    assert lens._last_ppr_scores.shape[0] == hr.graph.vcount()


@pytest.mark.integration
def test_seed_nodes_match_node_weights(lens) -> None:
    result = lens.retrieve_lens("What is HippoRAG and how does it retrieve information?")

    assert result.retrieval_mode == "ppr"
    assert lens._last_node_weights is not None

    for seed_id in result.seed_node_ids:
        idx = lens._hr.node_name_to_vertex_idx[seed_id]
        assert lens._last_node_weights[idx] > 0


@pytest.mark.integration
def test_retrieve_lens_json_and_passage_count(lens) -> None:
    result = lens.retrieve_lens("What is HippoRAG?")

    payload = json.loads(result.to_json())
    assert payload["question"] == "What is HippoRAG?"
    assert payload["retrieval_mode"] in ("ppr", "dpr_fallback")
    assert len(result.ranked_passages) == len(lens._hr.passage_node_keys)
    assert result.default_top_k == lens._hr.global_config.retrieval_top_k
