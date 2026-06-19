# HippoLens

Visualization wrapper around [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG): interactive knowledge-graph retrieval with full PageRank scores.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## OpenRouter API key

Indexing calls OpenRouter for OpenIE (LLM) and embeddings. Retrieval on a pre-built index is local.

1. Create a key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. Copy the env template and paste your key:

```bash
cp .env.example .env
```

Set the same key for both variables in `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
OPENAI_API_KEY=sk-or-v1-...   # same key; HippoRAG reads OPENAI_API_KEY
```

Demo models: `openai/gpt-4o-mini` (LLM) and `openai/text-embedding-3-small` (embeddings).

## Step 1 — Index demo corpus (uses API)

```bash
uv run python examples/index_corpus.py
```

Options: `--corpus examples/demo_corpus.json`, `--save-dir examples/demo_index` (defaults).

Expected cost for the 7-passage demo: roughly **$0.05–0.15** for a full first run. HippoRAG caches LLM responses under the save dir; re-running indexing reuses the cache when passages are unchanged.

After indexing, check `examples/demo_index/` for `graph.pickle` and embedding stores under the model-specific working subdirectory.

## Step 2 — Streamlit UI (no API)

Coming in Phase C:

```bash
uv run streamlit run hippolens/app/main.py
```

## Verify install

```bash
uv run python -c "import hippolens"
```
