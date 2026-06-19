# HippoLens

Interactive visualization layer on top of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG). Run a query, explore the knowledge graph with full Personalized PageRank (PPR) scores, inspect ranked passages and phrases, and highlight shortest paths from seed phrases to a selected passage.

**Workflow:** index a corpus once (LLM + embeddings via OpenRouter) → explore retrieval locally in a Streamlit UI with no repeated API calls.

## Features

- **Full PPR export** — every phrase and passage gets a score, not just top-k
- **Two-orbit graph** — phrases on the inner ring, passages on the outer ring (Cytoscape.js)
- **Visual encoding** — node size = seed weight, color = PPR score
- **Path highlight** — click a passage to show shortest paths from query seeds
- **Dynamic top-k** — change the passage slider without re-running retrieval
- **Subgraph / full graph** — toggle between a focused view and the entire indexed graph

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- OpenRouter API key ([openrouter.ai/keys](https://openrouter.ai/keys)) — required only for **indexing**

## Installation

```bash
git clone <repo-url>
cd hippolens
uv sync
```

Verify:

```bash
uv run python -c "import hippolens"
```

## OpenRouter setup

Indexing uses OpenRouter as an OpenAI-compatible endpoint for OpenIE (LLM) and embeddings. Demo models:

| Role | Model |
|------|-------|
| LLM (OpenIE, reranking) | `openai/gpt-4o-mini` |
| Embeddings | `openai/text-embedding-3-small` |

1. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. Copy the env template:

```bash
cp .env.example .env
```

3. Set **the same key** for both variables (HippoRAG reads `OPENAI_API_KEY`):

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-or-v1-...
```

Config lives in `examples/demo_config.py` (`make_demo_config`). Model names must match between indexing and the UI — they determine the on-disk working directory under `save_dir`.

## Step 1 — Index demo corpus (uses API)

The demo corpus is 7 short passages in `examples/demo_corpus.json` (hippos, hippocampus, HippoRAG, etc.).

```bash
uv run python examples/index_corpus.py
```

Options:

```bash
uv run python examples/index_corpus.py \
  --corpus examples/demo_corpus.json \
  --save-dir examples/demo_index
```



After success, check `examples/demo_index/` for `graph.pickle` and embedding stores under a model-specific subdirectory (e.g. `openai_gpt-4o-mini_openai_text-embedding-3-small/`).

## Step 2 — Streamlit UI (retrieval is local)

Once the index exists, retrieval and graph layout run locally — no LLM or embedding API calls per query. You still need `.env` with a valid key so HippoRAG can load the saved index.

```bash
uv run streamlit run hippolens/app/main.py
```

Custom index path:

```bash
uv run streamlit run hippolens/app/main.py -- --save-dir /path/to/index
```

Open [http://localhost:8501](http://localhost:8501), enter a query (e.g. *What is HippoRAG?*), and click **Retrieve**.

### UI layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HippoLens                                              Index: demo_index   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Query [________________________________]  Top-k passages [====●====]  [Retrieve] │
├──────────────────────────────┬──────────────────────────────────────────────┤
│                              │  Ranked passages (top-k)                   │
│   Knowledge graph            │  ┌────────────────────────────────────────┐  │
│   (Cytoscape.js)             │  │ #1  passage …  score 0.42              │  │
│                              │  │ #2  passage …  score 0.31              │  │
│   inner ring  ○ phrases      │  │ …                                      │  │
│   outer ring  ● passages     │  └────────────────────────────────────────┘  │
│                              │  Ranked phrases (top 20)                     │
│   size  = seed weight        │  ┌────────────────────────────────────────┐  │
│   color = PPR score          │  │ phrase …  score 0.18                     │  │
│                              │  └────────────────────────────────────────┘  │
│   click passage → path       │  Path phrases / Selected node (on click)   │
│   highlight from seeds       │                                              │
└──────────────────────────────┴──────────────────────────────────────────────┘

Sidebar (collapsed by default):
  Graph view:  (●) subgraph  ( ) full
  Subgraph phrases (N), passages (M)
  Legend: phrases (blue), passages (red/orange), edge styles
```

**Tips:**

- Change **Top-k passages** after retrieve — the graph and list update without a new API call.
- Click a passage in the graph or rank list to highlight paths from seed phrases; click again to clear.
- **Subgraph** shows top-N phrases and top-M passages; **full** shows the entire indexed graph (warning if >500 nodes).
- If retrieval falls back to dense passage retrieval (no matching facts), a **DPR fallback** badge appears and path highlight is disabled.

## Python API

```python
from hipporag import HippoRAG
from hippolens import HippoLens
from examples.demo_config import make_demo_config

hr = HippoRAG(global_config=make_demo_config("examples/demo_index"))
lens = HippoLens(hr)

# Standard HippoRAG retrieval (unchanged)
docs, scores = lens.retrieve(["What is HippoRAG?"])

# Extended result with full PPR, graph nodes, layout
result = lens.retrieve_lens("What is HippoRAG?", graph_mode="subgraph")
print(result.ranked_passages[:5])
print(result.to_json())
```

## Tests

```bash
uv run pytest tests/
uv run ruff check
```

Integration tests need a built demo index and `.env` with a valid key:

```bash
uv run pytest tests/test_lens_core.py -m integration
```

## Project layout

```
hippolens/
  lens.py           # HippoLens wrapper (retrieve, retrieve_lens)
  models.py         # GraphNode, RankedItem, LensQueryResult
  graph_export.py   # Subgraph / full graph export
  layout.py         # Two-orbit coordinates
  path_highlight.py # Shortest paths seed → passage
  style.py          # PPR / seed visual encoding
  app/
    main.py         # Streamlit entrypoint
    components/     # Cytoscape graph + rank panel
examples/
  demo_corpus.json  # Demo passages
  demo_config.py    # OpenRouter config
  index_corpus.py   # One-time indexing script
  demo_index/       # Built index (gitignored artifacts)
```

## License

See [LICENSE](LICENSE).
