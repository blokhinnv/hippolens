# HippoLens — план исполнения

Пошаговый чеклист для разработчика. Каждый пункт — атомарная задача с критерием приёмки.

**Предусловия:** Python 3.12, `uv sync`, ключ [OpenRouter](https://openrouter.ai/keys) в `.env` (только для фазы 0 — индексация).

**Порядок:** пункты внутри фазы выполняются последовательно; фазы 0→A→B→C→D.

---

## Фаза 0 — Подготовка окружения и demo index

### 0.1 Структура пакета

- [x] Создать `hippolens/` с `__init__.py`, экспорт `HippoLens`
- [x] Перенести/удалить заглушку `main.py` в корне или оставить как thin CLI wrapper
- [x] Добавить в `pyproject.toml`: `streamlit`, `python-dotenv`, documented entrypoint

**Приёмка:** `uv run python -c "import hippolens"` без ошибок.

### 0.2 OpenRouter — конфиг и секреты

- [x] `.env.example`:
  ```bash
  OPENROUTER_API_KEY=sk-or-v1-...
  OPENAI_API_KEY=sk-or-v1-...   # тот же ключ; hipporag читает OPENAI_API_KEY
  ```
- [x] `examples/demo_config.py` — `make_demo_config(save_dir)` (см. PLAN.md §2.2):
  - `llm_name="openai/gpt-4o-mini"`
  - `llm_base_url="https://openrouter.ai/api/v1"`
  - `embedding_model_name="openai/text-embedding-3-small"`
  - `embedding_base_url="https://openrouter.ai/api/v1"`
- [x] Хелпер `load_demo_env()` — `load_dotenv()` + assert `OPENAI_API_KEY`

**Приёмка:** без `.env` скрипт падает с понятной ошибкой; с ключом — config создаётся.

### 0.3 Demo corpus и скрипт индексации

- [x] `examples/demo_corpus.json` — 5–10 коротких passages (текст + id)
- [x] `examples/index_corpus.py`:
  - `load_demo_env()`
  - `HippoRAG(global_config=make_demo_config(save_dir))`
  - `index(docs)` из corpus
  - CLI: `--corpus`, `--save-dir` (default `examples/demo_index`)
- [x] README: получение ключа OpenRouter, `cp .env.example .env`, команда индексации, ориентир по стоимости (~$0.1)

**Приёмка:** `uv run python examples/index_corpus.py` с валидным ключом создаёт `examples/demo_index/` с graph.pickle и embeddings; повторный запуск использует LLM cache.

### 0.4 Зафиксировать версию hipporag

- [x] Pin `hipporag==2.0.0a3` (или текущая рабочая) в `pyproject.toml`
- [x] Записать в комментарии к patch: какие методы monkey-patch зависят от внутреннего API

**Приёмка:** `uv lock` актуален.

---

## Фаза A — Core wrapper

### A.1 `models.py`

- [ ] `GraphNode`, `GraphEdge`, `RankedItem`, `LensQueryResult` — dataclasses из PLAN.md §4
- [ ] Метод `LensQueryResult.to_json()` / `from_dict()` для передачи в Streamlit component

**Приёмка:** unit test сериализации round-trip.

### A.2 `HippoLens.__init__`

- [ ] Принимает готовый `HippoRAG` instance (не создаёт сам)
- [ ] Вызывает `hipporag.prepare_retrieval_objects()` если `not ready_to_retrieve`
- [ ] Инициализирует `_last_ppr_scores: np.ndarray | None`, `_last_node_weights`, `_last_linking_score_map`

**Приёмка:** `HippoLens(hr)` не падает на demo index.

### A.3 Patch `run_ppr`

- [ ] Сохранить `original_run_ppr = hipporag.run_ppr`
- [ ] Wrapped version: вызвать original, но до return вычислить и сохранить full `pagerank_scores` (потребуется либо дублировать вызов `personalized_pagerank`, либо patch на уровне тела `run_ppr` — предпочтительно wrap с доступом к `self.graph.personalized_pagerank` результату)
- [ ] Привязать patch к конкретному instance: `hipporag.run_ppr = types.MethodType(wrapped, hipporag)`

**Приёмка:** после одного `retrieve_lens` `_last_ppr_scores.shape[0] == hipporag.graph.vcount()`.

### A.4 Patch `graph_search_with_fact_entities`

- [ ] Wrap: перед return сохранить `node_weights`, `linking_score_map` на instance lens

**Приёмка:** после retrieve `is_seed` корректен для узлов с `node_weights > 0`.

### A.5 `retrieve()` — совместимость

- [ ] `HippoLens.retrieve(queries, num_to_retrieve=None)` → делегирует `self._hr.retrieve(...)`
- [ ] Return type идентичен hipporag

**Приёмка:** `test_retrieve_compat` — docs и scores совпадают с прямым вызовом.

### A.6 `retrieve_lens()`

- [ ] Один query (str) → вызывает внутренний retrieve pipeline или копирует логику одной итерации из `hipporag.retrieve`
- [ ] Собирает `LensQueryResult`:
  - `ranked_passages` — все passages с PPR score, sorted desc
  - `ranked_phrases` — все phrases с PPR score, sorted desc
  - `retrieval_mode`: `"ppr"` или `"dpr_fallback"` (если facts пусты)
  - `facts_used`, `timings`, `seed_node_ids`
- [ ] **Не** обрезать `ranked_passages` по `num_to_retrieve` внутри — обрезка только в UI; поле `default_top_k` = `num_to_retrieve or config.retrieval_top_k`

**Приёмка:** `retrieve_lens("...")` возвращает валидный JSON; `len(ranked_passages) == num_passage_nodes`.

### A.7 Тесты фазы A

- [ ] `tests/test_lens_core.py` — A.3, A.5, A.6 на demo index (pytest marker `@pytest.mark.integration` если нужны keys)

**Приёмка:** `uv run pytest tests/test_lens_core.py` green.

---

## Фаза B — Graph export, layout, path highlight

### B.1 `graph_export.py` — построение узлов

- [ ] `build_all_nodes(hipporag, ppr_scores, node_weights) -> list[GraphNode]`
  - phrase: content из `entity_embedding_store`
  - passage: content из `chunk_embedding_store`
  - `pagerank`, `seed_weight`, `is_seed`
- [ ] `build_edges(hipporag, node_ids: set[str]) -> list[GraphEdge]` из `graph.es` / `node_to_node_stats`

**Приёмка:** типы узлов корректны; каждый edge ссылается на существующие id.

### B.2 Subgraph export

- [ ] `export_subgraph(nodes, edges, *, top_n_phrases=30, top_m_passages=20) -> (nodes, edges)`
- [ ] Логика отбора — PLAN.md §5.1
- [ ] Hard cap 200 nodes

**Приёмка:** `test_subgraph_limits` — ≤200 узлов на demo.

### B.3 Full graph export

- [ ] `export_full(nodes, edges) -> (nodes, edges)` — без фильтрации
- [ ] Флаг `warn_large: bool` если `len(nodes) > 500`

**Приёмка:** `test_subgraph_vs_full` — full ⊃ subgraph по id.

### B.4 `layout.py`

- [ ] `two_orbit_layout(nodes: list[GraphNode]) -> dict[str, tuple[float, float]]`
- [ ] phrase → круг radius `r1=100`, passage → `r2=200` (константы в модуле)
- [ ] Возврат нормализованных координат для Cytoscape

**Приёмка:** `test_layout` — все phrase на ~r1, passage на ~r2 (±tolerance).

### B.5 `path_highlight.py`

- [ ] `find_paths_to_passage(hipporag, passage_id, seed_ids) -> list[list[str]]`
  - кратчайшие пути igraph от каждого seed phrase до passage
  - только undirected shortest path
- [ ] `path_elements(paths) -> tuple[set[str], set[tuple[str,str]]]` — узлы и рёбра для highlight

**Приёмка:** `test_path_highlight` — для demo query и top passage путь не пуст (или skip если DPR fallback).

### B.6 Визуальное кодирование (Python side)

- [ ] `encode_node_style(node: GraphNode, ppr_min, ppr_max, seed_max) -> dict`
  - `color`: interpolate по normalized PPR
  - `size`: `8 + 24 * (seed_weight / seed_max)` если seed, иначе `8`
  - `borderWidth`: 2 если `is_seed` else 0

**Приёмка:** seed nodes визуально крупнее; higher PPR — темнее (unit test на monotonicity).

### B.7 Интеграция в `retrieve_lens`

- [ ] Параметр `graph_mode: Literal["subgraph", "full"] = "subgraph"`
- [ ] Вызов export + layout → записать `x, y, style` в nodes или отдельный `graph_payload` dict

**Приёмка:** `retrieve_lens(..., graph_mode="full")` vs `"subgraph"` — разный `len(nodes)`.

---

## Фаза C — Streamlit app

### C.1 Каркас `app/main.py`

- [ ] `load_dotenv()` не обязателен для UI (индекс уже построен), но `save_dir` должен существовать
- [ ] `HippoRAG(global_config=make_demo_config(save_dir))` + `HippoLens(hr)` в `st.session_state`
  - **важно:** `llm_name` / `embedding_model_name` в config должны совпадать с теми, что использовались при index (имя влияет на `working_dir`)
- [ ] Поле query + кнопка Retrieve
- [ ] Sidebar: graph mode toggle, subgraph params (N, M), top-k slider (passages)

**Приёмка:** `uv run streamlit run hippolens/app/main.py` открывается без crash на demo index.

### C.2 `app/components/graph.py` — Cytoscape widget

- [ ] HTML template с Cytoscape.js (CDN)
- [ ] Python функция `render_graph(graph_payload, height=600) -> selected_node_id | None`
- [ ] Передача данных через `json.dumps` в `st.components.v1.html`
- [ ] Events: `tap` → `postMessage` с node id; hover → qtip/tooltip

**Приёмка:** граф рендерится; клик возвращает id в Streamlit (через query params или component value pattern).

### C.3 Orbit layout в Cytoscape

- [ ] Использовать preset positions из `layout.py` (не force-directed)
- [ ] phrase/passage разные shape или label position

**Приёмка:** визуально 2 кольца.

### C.4 Node style: size + color

- [ ] Применить `encode_node_style` к каждому элементу
- [ ] Легенда в sidebar: «Size = seed weight, Color = PPR»

**Приёмка:** соответствует PLAN.md §6.2.

### C.5 Path highlight в UI

- [ ] При `selected_node` типа passage → вызов `find_paths_to_passage`
- [ ] Передача `highlight_nodes`, `highlight_edges` в component
- [ ] Dim non-path elements (opacity 0.15)
- [ ] Панель «Path phrases»: список phrase на пути

**Приёмка:** клик на passage подсвечивает путь от seeds; повторный клик сбрасывает.

### C.6 `app/components/rank_panel.py`

- [ ] Список `ranked_passages` с score, обрезка по top-k slider
- [ ] Список `ranked_phrases` (top 20 informational, без slider)
- [ ] Клик по строке → select node на графе

**Приёмка:** top-k меняется мгновенно без кнопки Retrieve.

### C.7 Top-k highlight на графе

- [ ] При изменении slider: top-k passage nodes — full opacity + border; остальные passages — dimmed
- [ ] Без rerun полного retrieve — только `st.rerun` от slider с тем же `lens_result`

**Приёмка:** slider 3 → ровно 3 highlighted passages.

### C.8 Metadata panel

- [ ] `retrieval_mode`, `facts_used`, `timings`, count seeds
- [ ] Badge «DPR fallback» если `retrieval_mode == "dpr_fallback"`; path highlight disabled

**Приёмка:** отображается после retrieve.

### C.9 Full graph mode

- [ ] Toggle → при следующем retrieve используется `graph_mode="full"`
- [ ] Warning `st.warning` если nodes > 500

**Приёмка:** переключение работает; subgraph быстрее full на demo.

---

## Фаза D — Финализация

### D.1 README

- [ ] Описание проекта, установка (`uv sync`)
- [ ] OpenRouter: ключ, `.env`, модели (`gpt-4o-mini`, `text-embedding-3-small`)
- [ ] Шаг 1: index (API), Шаг 2: streamlit (без API)
- [ ] Скриншот или ascii layout

**Приёмка:** новый разработчик запускает demo по README.

### D.2 Полный test suite

- [ ] `tests/` — unit + integration (demo index)
- [ ] `uv run pytest`
- [ ] `uv run ruff check`

**Приёмка:** CI-ready locally.

### D.3 Ручной smoke test

- [ ] Index demo corpus
- [ ] Запустить app, 2–3 разных queries
- [ ] Проверить: subgraph/full, top-k, path highlight, DPR fallback query (если есть)

**Приёмка:** чеклист в PR description.

---

## Сводка оценок

| Фаза | Оценка | Зависимости |
|------|--------|-------------|
| 0 — Подготовка | 0.5–1 дн | API keys для index |
| A — Core | 1.5–2 дн | 0 |
| B — Graph | 1–1.5 дн | A |
| C — Streamlit | 2–3 дн | B |
| D — Финализация | 0.5 дн | C |
| **Итого** | **~6–8 дн** | |

---

## Definition of Done (весь проект)

1. `.env.example` + `examples/demo_config.py` — OpenRouter из коробки.
2. `examples/index_corpus.py` + `examples/demo_index/` — индексация отдельно от UI.
3. `HippoLens.retrieve()` и `HippoLens.retrieve_lens()` работают на готовом индексе.
4. Streamlit app локально: 2 орбиты, size=seed/color=PPR, path highlight, top-k passages, subgraph/full toggle.
5. Тесты и README позволяют воспроизвести demo с нуля (с ключом OpenRouter).
