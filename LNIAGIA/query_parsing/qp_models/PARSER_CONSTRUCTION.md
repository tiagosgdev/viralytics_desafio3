# Parser Construction and Runtime Behavior

This document explains how the two baseline parsers are built and how they behave at runtime:

- Rule-based parser: `baselines/rule_based.py`
- CRF parser: `baselines/crf_model.py`

It is implementation-driven and matches the current code.

## 1. Shared Input and Output Contract

Both parsers produce the same structure:

```json
{
  "include": {"key": ["value1", "value2"]},
  "exclude": {"key": ["value1"]}
}
```

The benchmark and evaluation flow treat each `(polarity, key, value)` as a structured pair.

## 2. Rule-Based Parser

## 2.1 What Is Learned in `fit`

The rule-based parser is deterministic at inference, but it still uses train+val data to build lexical resources:

1. Collect canonical values per key from `include` and `exclude`.
2. Mine extra surface forms from BIO spans when a span can be unambiguously mapped to one canonical value.
3. Inject type aliases (`TYPE_ALIASES`) only for types that exist in the current dataset.
4. Build phrase variants (`default_phrase_variants`) and normalize text for matching.
5. Build `PhraseMatcher` patterns for every `(key, canonical_value)` pair.
6. Build semantic expansion (`SEMANTIC_HINTS`) filtered by values that really exist in the folder-specific schema.

Main internal tables:

- `known_values_by_key`
- `surface_forms_by_pair`
- `match_id_to_pair`
- `semantic_expansion`

## 2.2 Inference (`parse`)

Given a query:

1. Run spaCy pipeline (`load_nlp`).
2. Generate candidate matches from:
   - `PhraseMatcher` spans
   - token-level semantic hints (for words such as `dark`, `light`, `warm`, `cool`)
3. For each candidate span, classify polarity with `_is_negated`:
   - local left-window negation check (up to 4 tokens)
   - dependency-based checks (`neg` relation and negation among children/head siblings)
4. Aggregate into sorted unique `include` and `exclude` lists per key.

## 2.3 Rule-Based Strengths and Limitations

Strengths:

- Fast and deterministic.
- Easy to inspect and debug.
- Uses explicit lexical and semantic resources.

Limitations:

- Depends heavily on lexical coverage and alias quality.
- Complex negation scope can still be hard for fixed heuristics.
- Semantic hints are hand-curated and intentionally small.

## 2.4 Export/Reload Support

`to_export_state()` stores:

- parser type
- spaCy model metadata
- known values
- surface forms
- semantic expansion
- negation words
- alias/hint config

`from_export_state()` reconstructs matcher state and semantic expansion, then marks parser as fitted.

## 3. CRF Parser

## 3.1 What Is Learned in `fit`

CRF training has two layers:

1. Sequence model training (`sklearn-crfsuite`), using train BIO labels.
2. Lookup table construction from train+val for robust value normalization at inference.

The value lookup layer builds:

- `known_values_by_key`
- `surface_to_keys` (single-token lexicon hints)
- `variant_to_canonical` (surface form -> canonical value)

It also injects `TYPE_ALIASES` where applicable.

## 3.2 Feature Engineering

Per-token feature groups in `_token_features`:

- Lexical: lower, lemma, prefix/suffix, shape
- Linguistic: POS, tag, dependency
- Token type: digit/punct flags
- Lexicon: in-schema indicator and candidate keys
- Negation: token-level and local-window negation signal
- Context: previous/next token lower/POS/dep
- Position: BOS/EOS flags

## 3.3 Training Procedure

1. Normalize labels for training (`-X` stripped; keep `NEG`, `O`, BIO tags).
2. Build train/val feature matrices and label sequences.
3. Grid-search over regularization pairs:
   - default grid: `{"c1": 0.1, "c2": 0.1}`, `{"c1": 0.01, "c2": 0.01}`
4. Select best model by validation weighted F1 over non-`O` labels (fallback to token accuracy when needed).

## 3.4 Inference (`parse`)

Given a query:

1. Predict BIO/NEG sequence with CRF.
2. Convert labels to spans.
3. Resolve each span text to canonical value via `_resolve_value`:
   - exact normalized variant lookup
   - fuzzy fallback (`difflib.get_close_matches`, cutoff `0.82`)
   - canonical normalization fallback
4. Detect negation:
   - predicted `NEG` near span, else
   - lexical negation in nearby tokens
5. Build `include` / `exclude` output.

## 3.5 CRF Strengths and Limitations

Strengths:

- Learns contextual patterns from labels.
- Usually better generalization than strict lexical rules.
- Still interpretable compared to deep black-box models.

Limitations:

- Requires labeled BIO data quality to be good.
- Value resolution still relies on lookup tables and fuzzy matching heuristics.
- Sequence model can be sensitive to annotation noise.

## 3.6 Export/Reload Support

CRF export is split across two artifacts:

- parser state (`parser_state.json`): lookup tables + parser config
- trained CRF object (`crf_model.pkl`)

`CRFParser.from_export_state(..., model=...)` reattaches the trained model and restores lookup tables.

## 4. Practical Difference Between the Parsers

- Rule-based focuses on explicit phrase and negation heuristics.
- CRF focuses on sequence labeling, then canonical value resolution.

Both are benchmarked with the same structured metrics and can be exported as winner artifacts for production use.