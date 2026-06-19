# HippoLens — технический план

## 0. Цель продукта

**HippoLens** — обёртка над `HippoRAG`, которая:

1. Сохраняет стандартный retrieval pipeline (facts → rerank → PPR).
2. **Дополнительно** экспонирует полный PageRank по графу для визуализации и анализа.
3. Даёт локальное Streamlit-приложение: интерактивный KG в стиле статьи (2 орбиты) + панель ранжирования с динамическим top-k.

**Успех:** пользователь вводит query → видит граф с раскраской по PPR → кликает passage → видит phrase-путь от seeds → меняет top-k без повторного запуска PPR.

---

## 1. Принятые решения

| Вопрос | Решение |
|--------|---------|
| Индексация | **Отдельно.** `index()` запускается вне HippoLens; на вход — уже настроенный `HippoRAG` с готовым `save_dir` |
| Масштаб графа в UI | **Оба режима:** переключатель «Full graph» / «Activated subgraph» |
| API retrieval | **Два метода:** `retrieve()` (как hipporag) + `retrieve_lens()` (расширенный результат) |
| top-k | Только **passages**; phrases — informational panel без отдельного slider |
| QA (`rag_qa`) | **Нет** в v1 |
| Deployment | **Локальный** инструмент |
| UI stack | **Streamlit** + Cytoscape.js (`components.html`) |
| Визуальное кодирование узлов | **size** = `seed_weight`, **color** = final PPR (как в статье) |
| Path highlight | **В scope v1:** клик на passage → подсветка phrase-пути от seeds |
| Название | Пакет `hippolens`, класс `HippoLens` |
| LLM / embeddings | **OpenRouter** — один API-ключ, OpenAI-compatible endpoint |
| Модели (demo) | LLM: `openai/gpt-4o-mini`; embeddings: `openai/text-embedding-3-small` |

---

## 2. Конфигурация API — OpenRouter

HippoRAG использует OpenAI SDK (`OpenAI(base_url=..., api_key=...)`). OpenRouter совместим с этим интерфейсом: достаточно сменить `base_url` и передать ключ OpenRouter.

### 2.1 Переменные окружения

Файл `.env` (не коммитить; шаблон — `.env.example`):

```bash
# Ключ OpenRouter (https://openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-v1-...

# HippoRAG читает OPENAI_API_KEY — алиас для совместимости
OPENAI_API_KEY=${OPENROUTER_API_KEY}
```

Опционально для OpenRouter (рекомендуется при публичном использовании):

```bash
OPENROUTER_HTTP_REFERER=http://localhost:8501
OPENROUTER_APP_NAME=hippolens
```

> **Примечание:** hipporag из коробки не прокидывает `HTTP-Referer` / `X-Title` в запросы. Для локальной демки обычно не требуется. Если OpenRouter вернёт 403 — добавить заголовки в wrapper или форкнуть `CacheOpenAI`.

### 2.2 Конфиг HippoRAG для демо

Единый модуль `examples/demo_config.py`:

```python
import os
from hipporag.utils.config_utils import BaseConfig

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

def make_demo_config(save_dir: str) -> BaseConfig:
  return BaseConfig(
      save_dir=save_dir,
      # LLM: OpenIE + recognition memory (DSPy rerank) — тот же endpoint
      llm_name="openai/gpt-4o-mini",
      llm_base_url=OPENROUTER_BASE,
      # Embeddings: retrieval + synonymy edges (через OpenAIEmbeddingModel)
      embedding_model_name="openai/text-embedding-3-small",
      embedding_base_url=OPENROUTER_BASE,
      # Демо: меньше топ-k для UI
      retrieval_top_k=20,
      linking_top_k=5,
  )
```

**Почему эти модели:**

| Компонент | Модель | Зачем |
|-----------|--------|-------|
| OpenIE (index) | `openai/gpt-4o-mini` | извлечение triples из passages; дёшево, JSON mode |
| Recognition memory (retrieve) | тот же LLM | фильтрация facts через DSPy prompt |
| Embeddings | `openai/text-embedding-3-small` | dense passage retrieval + phrase linking; дешевле `3-large` |

Альтернатива для embeddings без API-расходов: `embedding_model_name="facebook/contriever"` (локально, без GPU тяжело на больших корпусах). Для демо на 5–10 passages OpenRouter embeddings проще.

### 2.3 Что дергает API при index + retrieve

```
index(docs)
  ├── OpenIE (LLM)          → N вызовов × passages
  ├── Embeddings (API)      → chunks + entities + facts
  └── Synonymy KNN          → локально по embeddings

retrieve(query)
  ├── Fact embedding match  → локально
  ├── DSPy rerank (LLM)     → 1 вызов
  ├── Dense passage scores  → локально
  └── PPR                   → локально
```

Streamlit UI **не вызывает API** — только читает готовый индекс.

### 2.4 Загрузка `.env` в скриптах

```bash
# examples/index_corpus.py и app/main.py
# uv add python-dotenv  (или читать os.environ напрямую)
from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env"
```

### 2.5 Оценка стоимости demo

Corpus 5–10 коротких passages: порядка **$0.05–0.30** на полную индексацию + копейки за retrieve. Кэш LLM hipporag (`save_dir/llm_cache/*.sqlite`) ускоряет повторные запуски.

---

## 3. Архитектура

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit app (локально)                               │
│  - query, top-k slider, graph mode toggle, rank panel   │
└───────────────────────┬─────────────────────────────────┘
                        │ LensQueryResult (JSON)
┌───────────────────────▼─────────────────────────────────┐
│  hippolens/                                             │
│  - HippoLens (wrapper)                                  │
│  - retrieve() / retrieve_lens()                         │
│  - graph_export, layout, path_highlight                 │
└───────────────────────┬─────────────────────────────────┘
                        │ monkey patch (instrumentation)
┌───────────────────────▼─────────────────────────────────┐
│  hipporag.HippoRAG (pre-indexed, передан снаружи)       │
└─────────────────────────────────────────────────────────┘
```

**Структура пакета:**

```
hippolens/
  __init__.py
  lens.py              # HippoLens wrapper + patches
  models.py            # LensQueryResult, GraphNode, GraphEdge, RankedItem
  graph_export.py      # igraph → payload; full vs subgraph modes
  layout.py            # two-orbit radial layout
  path_highlight.py    # shortest paths seed phrase → passage
  app/
    main.py            # Streamlit entrypoint
    components/
      graph.py         # Cytoscape HTML widget
      rank_panel.py    # passage ranking + phrase info
examples/
  demo_config.py       # OpenRouter + BaseConfig для демо
  index_corpus.py      # скрипт индексации (вне UI)
  demo_index/          # готовый save_dir для демо
.env.example           # шаблон OPENROUTER_API_KEY
docs/
  PLAN.md
  EXECUTION.md         # пошаговый план для исполнителя
```

**Workflow пользователя:**

```bash
# 0. Ключ OpenRouter
cp .env.example .env   # вписать OPENROUTER_API_KEY

# 1. Индексация (один раз, отдельно; тратит API credits)
uv run python examples/index_corpus.py --save-dir examples/demo_index

# 2. Визуализация (без API, только готовый индекс)
uv run streamlit run hippolens/app/main.py -- --save-dir examples/demo_index
```

---

## 4. Monkey patch (instrumentation)

Точка патча — **`run_ppr`**, не смена return type у `retrieve`.

```python
# hipporag: full pagerank считается, но наружу уходят только passage scores
pagerank_scores = self.graph.personalized_pagerank(...)
doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
return sorted_doc_ids, sorted_doc_scores  # phrase scores теряются
```

| Метод | Действие при init `HippoLens` |
|-------|-------------------------------|
| `run_ppr` | Кэш `pagerank_scores` (полный вектор длины `vcount()`) |
| `graph_search_with_fact_entities` | Кэш `node_weights`, `linking_score_map` |
| `retrieve` | Не менять сигнатуру; делегировать в `hipporag.retrieve` |

**Публичный API:**

```python
class HippoLens:
    def __init__(self, hipporag: HippoRAG): ...

    def retrieve(self, queries, num_to_retrieve=None) -> list[QuerySolution]:
        """Прозрачная обёртка hipporag.retrieve."""

    def retrieve_lens(self, query: str, num_to_retrieve: int | None = None) -> LensQueryResult:
        """Retrieve + полный PPR, граф, seeds, paths."""
```

---

## 5. Модель данных

```python
@dataclass
class GraphNode:
    id: str
    node_type: Literal["phrase", "passage"]
    label: str
    content: str
    pagerank: float          # → color
    seed_weight: float       # → size (0 если не seed)
    is_seed: bool

@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float

@dataclass
class RankedItem:
    id: str
    node_type: Literal["phrase", "passage"]
    content: str
    score: float
    rank: int

@dataclass
class LensQueryResult:
    question: str
    graph_mode: Literal["full", "subgraph"]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    ranked_passages: list[RankedItem]   # все passages, desc — для динамического top-k
    ranked_phrases: list[RankedItem]    # informational, без top-k slider
    seed_node_ids: list[str]
    linking_score_map: dict[str, float]
    facts_used: list[tuple]
    retrieval_mode: Literal["ppr", "dpr_fallback"]
    timings: dict[str, float]
```

---

## 6. Граф: два режима отображения

### 6.1 Activated subgraph (default)

Узлы:

- все seeds (`node_weights > 0`);
- top-N phrases по PPR (default N=30);
- top-M passages по PPR (default M=20);
- узлы на кратчайших путях seed → top passages (для path highlight).

Рёбра — только между включёнными узлами. Лимит ~200 узлов.

### 6.2 Full graph

Все phrase + passage узлы из `hipporag.graph`. Предупреждение в UI при `vcount() > 500`. Для больших графов — снижение edge opacity, упрощённые labels, lazy render.

Переключатель в sidebar: `Graph view: Subgraph | Full`.

---

## 7. Визуализация

### 7.1 Layout — 2 орбиты

- **Внутренняя орбита** (`r1`): phrase nodes
- **Внешняя орбита** (`r2 > r1`): passage nodes
- Углы — равномерно внутри типа

### 7.2 Кодирование узлов (как в статье)

| Канал | Источник | Реализация |
|-------|----------|------------|
| **Color** | final PPR | min-max normalize → gradient (светлый → тёмный) |
| **Size** | `seed_weight` | `base_size + scale * normalized_seed_weight`; min size для non-seed |
| **Border** | `is_seed` | контрастная обводка у seed nodes |

### 7.3 Path highlight

При клике на **passage** node:

1. Найти все **seed phrase** nodes (`is_seed and node_type == "phrase"`).
2. Для каждого seed — кратчайший путь в igraph до выбранного passage.
3. Подсветить: узлы и рёбра на объединении путей; остальное — dim (opacity 0.15).
4. В панели — список phrase nodes на пути + их PPR/seed_weight.

Реализация: `path_highlight.py`, вызов из Cytoscape click handler через `postMessage` → Streamlit session state.

### 7.4 Интерактивность

| Событие | Действие |
|---------|----------|
| Hover | tooltip: type, PPR, seed_weight, truncated content |
| Click passage | path highlight + content panel |
| Click phrase | content panel + связанные passages |
| top-k slider | highlight top-k passages, dim остальные (без re-retrieve) |

---

## 8. Web UI (Streamlit)

```
┌──────────────────────────────────────────────────────────────┐
│ save_dir: [examples/demo_index]  Graph: ○ Subgraph ● Full   │
│ [Query input________________]  [Retrieve]   top-k: ──●── 10 │
├───────────────────────────────┬──────────────────────────────┤
│                               │  Ranked Passages (top-k)   │
│   Graph (2 orbits)            │  1. [0.142] passage...       │
│   size=seed, color=PPR        │  2. [0.098] ...            │
│                               │                              │
│   click passage → path        │  Phrases (by PPR, info)      │
│                               │  1. [0.055] "thomas"         │
├───────────────────────────────┴──────────────────────────────┤
│  Selected / Path: nodes, edges, metadata                     │
│  Mode: PPR | Seeds: 5 | Facts: 3 | PPR: 12ms                │
└──────────────────────────────────────────────────────────────┘
```

**Динамический top-k (passages only):**

1. `Retrieve` → один PPR run → `st.session_state.lens_result`.
2. Slider меняет k → фильтрация `ranked_passages[:k]` + style update на графе.
3. Без повторного вызова `retrieve_lens`.

---

## 9. Зависимости

```toml
dependencies = [
    "hipporag>=2.0.0a3",
    "streamlit>=1.30",
    "numpy",
    "python-dotenv>=1.0",
]
```

Cytoscape.js — CDN в HTML component.

---

## 10. Тестирование

| Тест | Проверка |
|------|----------|
| `test_run_ppr_cache` | full vector длины `vcount()` |
| `test_retrieve_lens_shape` | nodes, ranked_passages после retrieve на demo index |
| `test_retrieve_compat` | `retrieve()` ≡ hipporag |
| `test_top_k_filter` | client filter, порядок сохранён |
| `test_subgraph_vs_full` | оба режима export |
| `test_path_highlight` | passage click → non-empty phrase path |
| `test_node_encoding` | size correlates with seed_weight, color with PPR |

---

## 11. Риски

| Риск | Митигация |
|------|-----------|
| Patch ломается при обновлении hipporag | patch только `run_ppr`; pin версию |
| Full graph тормозит UI | warning >500 nodes; subgraph default |
| Streamlit rerun | session_state; style-only update на top-k |
| DPR fallback (no facts) | badge «DPR mode»; path highlight недоступен |
| Большой save_dir | документировать минимальный demo corpus |
| OpenRouter rate limits / 402 | маленький demo corpus; LLM cache hipporag |
| Slug модели не поддерживается | зафиксировать slugs в `demo_config.py`; smoke test index |
| Embeddings slug на OpenRouter | имя должно содержать `text-embedding` для `OpenAIEmbeddingModel` |

---

## 12. Вне scope v1

- `rag_qa` / answer generation
- Индексация внутри UI
- top-k slider для phrases
- Публичный deploy / HuggingFace Spaces
- Анимация PPR по итерациям
- Side-by-side DPR vs PPR

## 13. Идеи на v2

- CLI: `hippolens retrieve --query "..." --json`
- Export results JSON
- Side-by-side embedding vs PPR ranking
