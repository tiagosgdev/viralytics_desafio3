# Search System Overview (LNIAGIA)

This document explains how the clothing search works end-to-end, and which models are used in each step.

---

## 1) High-level architecture

The search system is a **hybrid pipeline**:

1. **Natural-language query parsing (LLM)**  
   Converts user text into structured filters (`include` / `exclude`).
2. **Semantic retrieval (embeddings + vector DB)**  
   Finds similar items by meaning, using vector similarity in Qdrant.
3. **Metadata-aware filtering/reranking**  
   Applies strict or soft constraints over fields like `type`, `color`, `style`, etc.

Main entry point for interactive usage:
- `LNIAGIA/search_app.py`

---

## 2) Models used

### A) LLM for query parsing
- **Model:** `qwen2.5:3b-instruct`
- **Where configured:** `LNIAGIA/llm_query_parser.py` (`OLLAMA_MODEL`)
- **Runtime:** via local Ollama (`ollama.chat`)
- **Role:** transforms free-text queries into structured JSON filters:
  - `include`: values the user wants
  - `exclude`: values the user does not want

Example output shape:

```json
{
  "include": {"type": ["shorts"], "style": ["sporty"]},
  "exclude": {"pattern": ["floral"]}
}
```

The parser validates returned values against the controlled vocabulary in `DB/models.py` + `DB/vector/nl_mappings.py`.

### B) Embedding model for semantic search
- **Model:** `BAAI/bge-base-en-v1.5`
- **Where configured:** `LNIAGIA/DB/vector/VectorDBManager.py` (`EMBEDDING_MODEL_NAME`)
- **Library:** `sentence-transformers`
- **Role:** encodes item descriptions and user queries to vectors for cosine similarity search.

Important detail:
- Queries are prefixed with:
  `Represent this sentence for searching relevant passages: `
  (BGE best-practice for better retrieval quality).

### C) Vector database (retrieval engine)
- **Engine:** Qdrant local storage
- **Where configured:** `LNIAGIA/DB/vector/VectorDBManager.py`
- **Collection:** `clothing_search`
- **Role:** stores embeddings + metadata payload and executes nearest-neighbor search + metadata filters.

---

## 3) Offline indexing flow (how the searchable DB is built)

1. **Structured catalog data** is generated/stored (JSON records with attributes).  
2. `description_generator.py` converts each item into a rich natural-language description (with synonyms).  
3. `VectorDBManager.build_vector_db(...)`:
   - loads descriptions
   - computes embeddings with `BAAI/bge-base-en-v1.5`
   - creates/recreates Qdrant collection
   - uploads vectors + metadata payload

Metadata is kept in payload so the search step can filter by fields like:
`type`, `color`, `style`, `pattern`, `material`, `fit`, `gender`, `age_group`, `season`, `occasion`, plus selected type-specific fields.

---

## 4) Online query flow (what happens when a user searches)

Implemented in `search_app.py`:

1. **User types query** (natural language)
2. **LLM parsing** (`parse_query`) extracts structured filters
3. **User chooses mode**:
   - **Strict** (`strict=True`)
   - **Non-strict / soft** (`strict=False`)
4. **`filtered_search(...)` runs semantic retrieval** and applies constraints
5. **Top results are shown** with metadata and similarity score

---

## 5) Strict vs non-strict behavior

Implemented in `VectorDBManager.filtered_search(...)`.

### Strict mode
- `include` + `exclude` are translated to Qdrant `must` and `must_not` filters.
- Retrieval only returns items satisfying those hard constraints.

### Non-strict mode
- Semantic candidates are retrieved more broadly.
- If `exclude` exists, an additional filtered query is merged with normal results.
- Soft score adjustment is applied:
  - match excluded values → score penalty (`PENALTY_WEIGHT`)
  - (boost weight constant exists but is currently not actively applied)
- Results below `SIMILARITY_THRESHOLD` are dropped.
- Remaining results are re-ranked by final score.

---

## 6) Key search configuration

Defined mainly in `LNIAGIA/DB/vector/VectorDBManager.py`:

- `TOTAL_RESULTS = 10`
- `MAX_QUERY_RESULTS = 50`
- `SIMILARITY_THRESHOLD = 0.25`
- `PENALTY_WEIGHT = 0.2`
- `BOOST_WEIGHT = 0.2`
- `COLLECTION_NAME = "clothing_search"`

LLM model config:
- `OLLAMA_MODEL = "qwen2.5:7b-instruct-q3_K_M"` in `LNIAGIA/llm_query_parser.py`

---

## 7) Controlled vocabulary and filter schema

The parser and filtering layer rely on `LNIAGIA/DB/models.py`:
- `FILTERABLE_FIELDS`: allowed filter fields
- `FREE_TEXT_FILTER_FIELDS = ["brand"]`
- domain values (types, colors, styles, seasons, etc.)

This keeps parser outputs aligned with what Qdrant can actually filter.

---

## 8) Evaluation

`LNIAGIA/tests/run_evaluation.py` evaluates retrieval quality for both modes:
- strict
- non-strict

Metrics include:
- precision
- recall
- F1
- average score

Outputs are saved per query plus an aggregate summary JSON.

---

## 9) Practical summary

In short, the search combines:
- **LLM understanding** of user intent (`qwen2.5:3b-instruct`)
- **Embedding-based semantic retrieval** (`BAAI/bge-base-en-v1.5`)
- **Qdrant metadata filtering/reranking**

This provides both natural-language flexibility and controllable attribute constraints.