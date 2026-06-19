"""HippoLens wrapper over HippoRAG with PPR instrumentation."""

from __future__ import annotations

import time
import types
from typing import TYPE_CHECKING, Literal

import numpy as np
from hipporag.utils.misc_utils import compute_mdhash_id, min_max_normalize

from hippolens.graph_export import build_all_nodes, build_edges, export_full, export_subgraph
from hippolens.layout import two_orbit_layout
from hippolens.models import GraphNode, LensQueryResult, RankedItem
from hippolens.style import encode_node_style, style_edges

if TYPE_CHECKING:
    from hipporag import HippoRAG
    from hipporag.utils.misc_utils import QuerySolution


class HippoLens:
    """Wrapper over HippoRAG that exposes full PageRank scores for visualization."""

    def __init__(self, hipporag: HippoRAG) -> None:
        self._hr = hipporag
        self._last_ppr_scores: np.ndarray | None = None
        self._last_node_weights: np.ndarray | None = None
        self._last_linking_score_map: dict[str, float] = {}

        if not self._hr.ready_to_retrieve:
            self._hr.prepare_retrieval_objects()

        self._apply_patches()

    def _apply_patches(self) -> None:
        hr = self._hr
        if getattr(hr, "_hippolens_patched", False):
            return

        lens = self
        original_run_ppr = hr.run_ppr
        original_graph_search = hr.graph_search_with_fact_entities

        def wrapped_run_ppr(
            self, reset_prob: np.ndarray, damping: float = 0.5
        ) -> tuple[np.ndarray, np.ndarray]:
            if damping is None:
                damping = 0.5
            reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
            pagerank_scores = self.graph.personalized_pagerank(
                vertices=range(len(self.node_name_to_vertex_idx)),
                damping=damping,
                directed=False,
                weights="weight",
                reset=reset_prob,
                implementation="prpack",
            )
            lens._last_ppr_scores = np.array(pagerank_scores, dtype=float)
            lens._last_node_weights = np.array(reset_prob, dtype=float)

            doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
            sorted_doc_ids = np.argsort(doc_scores)[::-1]
            sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
            return sorted_doc_ids, sorted_doc_scores

        def wrapped_graph_search(
            self,
            query: str,
            link_top_k: int,
            query_fact_scores: np.ndarray,
            top_k_facts: list[tuple],
            top_k_fact_indices: list[str],
            passage_node_weight: float = 0.05,
        ) -> tuple[np.ndarray, np.ndarray]:
            # Mirrors hipporag==2.0.0a3 graph_search_with_fact_entities; captures seed state.
            linking_score_map: dict[str, float] = {}
            phrase_scores: dict[str, list[float]] = {}
            phrase_weights = np.zeros(len(self.graph.vs["name"]))
            passage_weights = np.zeros(len(self.graph.vs["name"]))

            for rank, fact in enumerate(top_k_facts):
                subject_phrase = fact[0].lower()
                object_phrase = fact[2].lower()
                fact_score = (
                    query_fact_scores[top_k_fact_indices[rank]]
                    if query_fact_scores.ndim > 0
                    else query_fact_scores
                )
                for phrase in [subject_phrase, object_phrase]:
                    phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                    phrase_id = self.node_name_to_vertex_idx.get(phrase_key)

                    if phrase_id is not None:
                        phrase_weights[phrase_id] = fact_score
                        if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                            phrase_weights[phrase_id] /= len(
                                self.ent_node_to_chunk_ids[phrase_key]
                            )

                    phrase_scores.setdefault(phrase, []).append(fact_score)

            for phrase, scores in phrase_scores.items():
                linking_score_map[phrase] = float(np.mean(scores))

            if link_top_k:
                phrase_weights, linking_score_map = self.get_top_k_weights(
                    link_top_k, phrase_weights, linking_score_map
                )

            dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
            normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

            for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
                passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
                passage_dpr_score = normalized_dpr_sorted_scores[i]
                passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
                passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
                passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
                linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

            node_weights = phrase_weights + passage_weights

            if len(linking_score_map) > 30:
                linking_score_map = dict(
                    sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30]
                )

            lens._last_linking_score_map = dict(linking_score_map)
            lens._last_node_weights = node_weights.copy()

            ppr_start = time.time()
            ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
                node_weights, damping=self.global_config.damping
            )
            self.ppr_time += time.time() - ppr_start

            return ppr_sorted_doc_ids, ppr_sorted_doc_scores

        hr.run_ppr = types.MethodType(wrapped_run_ppr, hr)
        hr.graph_search_with_fact_entities = types.MethodType(wrapped_graph_search, hr)
        hr._hippolens_patched = True
        hr._hippolens_original_run_ppr = original_run_ppr
        hr._hippolens_original_graph_search = original_graph_search

    def retrieve(
        self,
        queries: list[str],
        num_to_retrieve: int | None = None,
        gold_docs: list[list[str]] | None = None,
    ) -> list[QuerySolution] | tuple[list[QuerySolution], dict]:
        return self._hr.retrieve(
            queries=queries,
            num_to_retrieve=num_to_retrieve,
            gold_docs=gold_docs,
        )

    def retrieve_lens(
        self,
        query: str,
        num_to_retrieve: int | None = None,
        graph_mode: Literal["subgraph", "full"] = "subgraph",
        top_n_phrases: int = 30,
        top_m_passages: int = 20,
    ) -> LensQueryResult:
        hr = self._hr
        timings: dict[str, float] = {}
        default_top_k = num_to_retrieve or hr.global_config.retrieval_top_k

        if not hr.ready_to_retrieve:
            hr.prepare_retrieval_objects()

        t0 = time.time()
        hr.get_query_embeddings([query])
        timings["embedding"] = time.time() - t0

        t1 = time.time()
        query_fact_scores = hr.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, _rerank_log = hr.rerank_facts(query, query_fact_scores)
        timings["rerank"] = time.time() - t1

        if len(top_k_facts) == 0:
            retrieval_mode = "dpr_fallback"
            t2 = time.time()
            sorted_doc_ids, sorted_doc_scores = hr.dense_passage_retrieval(query)
            timings["retrieval"] = time.time() - t2
            self._last_ppr_scores = None
            self._last_node_weights = None
            self._last_linking_score_map = {}
            ranked_passages = self._build_ranked_from_dpr(sorted_doc_ids, sorted_doc_scores)
            ranked_phrases = self._build_ranked_phrases_zero()
        else:
            retrieval_mode = "ppr"
            t2 = time.time()
            hr.graph_search_with_fact_entities(
                query=query,
                link_top_k=hr.global_config.linking_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=hr.global_config.passage_node_weight,
            )
            timings["retrieval"] = time.time() - t2
            ranked_passages = self._build_ranked_passages_ppr()
            ranked_phrases = self._build_ranked_phrases_ppr()

        seed_node_ids = self._seed_node_ids()
        nodes, edges = self._build_graph_payload(
            graph_mode=graph_mode,
            seed_node_ids=seed_node_ids,
            top_n_phrases=top_n_phrases,
            top_m_passages=top_m_passages,
        )

        return LensQueryResult(
            question=query,
            graph_mode=graph_mode,
            nodes=nodes,
            edges=edges,
            ranked_passages=ranked_passages,
            ranked_phrases=ranked_phrases,
            seed_node_ids=seed_node_ids,
            linking_score_map=dict(self._last_linking_score_map),
            facts_used=list(top_k_facts),
            retrieval_mode=retrieval_mode,
            timings=timings,
            default_top_k=default_top_k,
        )

    def _build_graph_payload(
        self,
        *,
        graph_mode: Literal["subgraph", "full"],
        seed_node_ids: list[str],
        top_n_phrases: int,
        top_m_passages: int,
    ) -> tuple[list[GraphNode], list]:
        hr = self._hr
        all_nodes = build_all_nodes(hr, self._last_ppr_scores, self._last_node_weights)
        all_node_ids = {n.id for n in all_nodes}
        all_edges = build_edges(hr, all_node_ids)

        if graph_mode == "full":
            nodes, edges = export_full(all_nodes, all_edges)
        else:
            nodes, edges = export_subgraph(
                all_nodes,
                all_edges,
                hipporag=hr,
                seed_node_ids=seed_node_ids,
                top_n_phrases=top_n_phrases,
                top_m_passages=top_m_passages,
            )

        if not nodes:
            return nodes, edges

        positions = two_orbit_layout(nodes)
        phrase_pprs = [n.pagerank for n in nodes if n.node_type == "phrase"]
        passage_pprs = [n.pagerank for n in nodes if n.node_type == "passage"]
        phrase_ppr_min = min(phrase_pprs) if phrase_pprs else 0.0
        phrase_ppr_max = max(phrase_pprs) if phrase_pprs else 1.0
        passage_ppr_min = min(passage_pprs) if passage_pprs else 0.0
        passage_ppr_max = max(passage_pprs) if passage_pprs else 1.0
        phrase_seed_weights = [n.seed_weight for n in nodes if n.node_type == "phrase"]
        passage_seed_weights = [n.seed_weight for n in nodes if n.node_type == "passage"]
        phrase_seed_min = min(phrase_seed_weights) if phrase_seed_weights else 0.0
        phrase_seed_max = max(phrase_seed_weights) if phrase_seed_weights else 1.0
        passage_seed_min = min(passage_seed_weights) if passage_seed_weights else 0.0
        passage_seed_max = max(passage_seed_weights) if passage_seed_weights else 1.0

        styled_nodes: list[GraphNode] = []
        for node in nodes:
            x, y = positions[node.id]
            style = encode_node_style(
                node,
                phrase_ppr_min,
                phrase_ppr_max,
                passage_ppr_min,
                passage_ppr_max,
                phrase_seed_min,
                phrase_seed_max,
                passage_seed_min,
                passage_seed_max,
            )
            styled_nodes.append(
                GraphNode(
                    id=node.id,
                    node_type=node.node_type,
                    label=node.label,
                    content=node.content,
                    pagerank=node.pagerank,
                    seed_weight=node.seed_weight,
                    is_seed=node.is_seed,
                    x=x,
                    y=y,
                    style=style,
                )
            )

        nodes_by_id = {n.id: n for n in styled_nodes}
        styled_edges = style_edges(edges, nodes_by_id)

        return styled_nodes, styled_edges

    def _seed_node_ids(self) -> list[str]:
        if self._last_node_weights is None:
            return []
        hr = self._hr
        return [
            hr.graph.vs[idx]["name"]
            for idx, weight in enumerate(self._last_node_weights)
            if weight > 0
        ]

    def _build_ranked_passages_ppr(self) -> list[RankedItem]:
        hr = self._hr
        assert self._last_ppr_scores is not None
        items: list[tuple[float, str, str]] = []
        for key in hr.passage_node_keys:
            vertex_idx = hr.node_name_to_vertex_idx[key]
            score = float(self._last_ppr_scores[vertex_idx])
            content = hr.chunk_embedding_store.get_row(key)["content"]
            items.append((score, key, content))
        items.sort(key=lambda x: x[0], reverse=True)
        return [
            RankedItem(id=node_id, node_type="passage", content=content, score=score, rank=rank)
            for rank, (score, node_id, content) in enumerate(items, start=1)
        ]

    def _build_ranked_phrases_ppr(self) -> list[RankedItem]:
        hr = self._hr
        assert self._last_ppr_scores is not None
        items: list[tuple[float, str, str]] = []
        for key in hr.entity_node_keys:
            vertex_idx = hr.node_name_to_vertex_idx[key]
            score = float(self._last_ppr_scores[vertex_idx])
            content = hr.entity_embedding_store.get_row(key)["content"]
            items.append((score, key, content))
        items.sort(key=lambda x: x[0], reverse=True)
        return [
            RankedItem(id=node_id, node_type="phrase", content=content, score=score, rank=rank)
            for rank, (score, node_id, content) in enumerate(items, start=1)
        ]

    def _build_ranked_from_dpr(
        self, sorted_doc_ids: np.ndarray, sorted_doc_scores: np.ndarray
    ) -> list[RankedItem]:
        hr = self._hr
        id_to_score = dict(zip(sorted_doc_ids.tolist(), sorted_doc_scores.tolist(), strict=True))
        items: list[tuple[float, str, str]] = []
        for doc_idx, key in enumerate(hr.passage_node_keys):
            score = float(id_to_score.get(doc_idx, 0.0))
            content = hr.chunk_embedding_store.get_row(key)["content"]
            items.append((score, key, content))
        items.sort(key=lambda x: x[0], reverse=True)
        return [
            RankedItem(id=node_id, node_type="passage", content=content, score=score, rank=rank)
            for rank, (score, node_id, content) in enumerate(items, start=1)
        ]

    def _build_ranked_phrases_zero(self) -> list[RankedItem]:
        hr = self._hr
        items = [
            (
                0.0,
                key,
                hr.entity_embedding_store.get_row(key)["content"],
            )
            for key in hr.entity_node_keys
        ]
        return [
            RankedItem(id=node_id, node_type="phrase", content=content, score=score, rank=rank)
            for rank, (score, node_id, content) in enumerate(items, start=1)
        ]
