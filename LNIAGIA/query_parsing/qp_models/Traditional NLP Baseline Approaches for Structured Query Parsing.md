# Traditional NLP Baseline Approaches for Structured Query Parsing

## Overview

The query parsing task consists of three sub-tasks that must be solved in sequence:

1. **Span Detection** — Identify tokens or phrases in the query that refer to clothing attributes (e.g., "dark", "oversized", "casual").
2. **Slot Mapping** — Map each detected mention to a canonical key-value pair from the attribute schema (e.g., "dark" → `color: [black, navy, burgundy, …]`).
3. **Negation Detection** — Determine the user's intent for each detected attribute — whether they want it included or excluded from results (e.g., "not too oversized" → `exclude: fit: oversized`).

Two traditional NLP models are proposed as baselines to compare against the LLM-based parser. Together they represent a natural progression from fully rule-based to statistical learning.

---

## Baseline 1: Rule-Based Pipeline with spaCy

### Description

The first baseline is a deterministic, rule-based system built on top of [spaCy](https://spacy.io/), a widely used industrial-strength NLP library. This approach requires no training data and operates entirely on hand-crafted rules and lexicon lookups. It serves as the simplest possible baseline and establishes an upper-bound reference for precision on vocabulary that appears verbatim in the attribute schema.

### Pipeline Steps

#### 1. Tokenisation and Linguistic Annotation

The input query is processed by spaCy's pre-trained English pipeline (`en_core_web_sm` or larger), which produces:

- **Tokens** — individual words and punctuation units.
- **Part-of-Speech (POS) tags** — grammatical category of each token (noun, adjective, adverb, etc.).
- **Dependency parse tree** — a directed graph representing the syntactic relationships between tokens (e.g., subject, object, modifier).
- **Lemmas** — the base/dictionary form of each token (e.g., "fitted" → "fit").

#### 2. Lexicon Matching

A lookup table is constructed from the full attribute schema, mapping every known value to its corresponding key:
```python
lexicon = {
    "black": "color",
    "oversized": "fit",
    "casual": "style",
    "t-shirt": "type",
    # ...
}
```

spaCy's `PhraseMatcher` or `EntityRuler` is used to scan the token sequence and flag any token or multi-token phrase that matches a known value. Matching is performed on lowercased lemmas to improve recall (e.g., "fitted" matches "fit").

#### 3. Semantic Expansion

When the query uses an indirect or descriptive reference (e.g., "dark" instead of "black"), the system applies a manually curated semantic expansion table to map the mention to one or more canonical values:
```python
semantic_expansion = {
    "dark": ["black", "navy", "burgundy", "olive", "brown"],
    "light": ["white", "cream", "beige", "yellow"],
    "warm":  ["red", "orange", "coral", "burgundy"],
    # ...
}
```

This table can be constructed semi-automatically using WordNet synsets or word embedding nearest-neighbour lookups, and then manually reviewed for domain accuracy.

#### 4. Negation Scope Detection

The dependency parse tree produced by spaCy is traversed to identify negation. The key signal is the `neg` dependency relation, which spaCy assigns to tokens such as "not", "without", "never", and "avoid". Any attribute span that falls within the syntactic scope of a negation token is marked as `exclude`; all others are marked as `include`.

For example, in the sentence:

> *"I want a T-shirt that is **not** too oversized"*

The token "not" has a `neg` dependency arc to "oversized", so "oversized" is classified as an excluded attribute.

#### 5. Output Construction

The detected spans, their mapped key-value pairs, and their include/exclude polarity are aggregated into the final structured JSON output:
```json
{
  "include": {
    "type": ["short_sleeve_top"],
    "style": ["casual"]
  },
  "exclude": {
    "fit": ["oversized"]
  }
}
```

### Strengths and Limitations

| Aspect | Detail |
|---|---|
| **Strengths** | Fast, fully interpretable, no training data required, high precision on in-vocabulary queries |
| **Limitations** | Zero generalisation to unseen vocabulary, brittle to paraphrasing, semantic expansion table requires manual effort, negation heuristics can fail on complex syntax |

---

## Baseline 2: CRF-Based Sequence Labeler

### Description

The second baseline is a **Conditional Random Field (CRF)** model, which is the canonical pre-deep-learning approach for Named Entity Recognition (NER) and slot filling — tasks that are structurally identical to the attribute extraction problem defined here. Unlike the rule-based system, the CRF learns from annotated examples and can generalise to surface forms not seen during rule construction.

A CRF is a discriminative probabilistic model that, given a sequence of input tokens, predicts a label for every token jointly, taking into account the entire sequence context. This makes it well-suited for tagging tasks where the label of one token depends on the labels of its neighbours.

The implementation uses [`sklearn-crfsuite`](https://sklearn-crfsuite.readthedocs.io/), a lightweight Python wrapper around CRFsuite, combined with spaCy for feature extraction.

### Sequence Labelling Scheme

The model uses the standard **BIO (Begin-Inside-Outside) tagging scheme**, where each token receives one of the following labels:

- `B-<KEY>` — Beginning of an attribute span for the given key (e.g., `B-COLOR`).
- `I-<KEY>` — Inside (continuation of) an attribute span (e.g., `I-COLOR`).
- `O` — Outside any attribute span (not relevant to any key).
- `NEG` — A negation marker token (e.g., "not", "without").

**Example annotation:**

| Token | Label |
|---|---|
| I | O |
| want | O |
| a | O |
| dark | B-COLOR |
| T-shirt | B-TYPE |
| that | O |
| is | O |
| not | NEG |
| too | O |
| oversized | B-FIT |
| , | O |
| casual | B-STYLE |

After inference, a post-processing step reads the `NEG` labels and applies negation scope logic: any `B/I-<KEY>` span that is preceded by (or syntactically linked to) a `NEG` token within a defined window is assigned to `exclude`; all others go to `include`.

### Feature Set

For each token at position `i`, the following features are extracted and passed to the CRF:

| Feature Group | Features |
|---|---|
| **Token-level** | Lowercased token, lemma, is digit, is punctuation, token prefix (3 chars), token suffix (3 chars) |
| **Morphological** | POS tag, coarse POS tag, spaCy dependency label |
| **Lexicon** | Is token in known value list, which key it belongs to (if any) |
| **Contextual window** | Same token-level and POS features for tokens at positions `i-2`, `i-1`, `i+1`, `i+2` |
| **Positional** | Is beginning of sentence, is end of sentence |
| **Negation signal** | Is the token a known negation word ("not", "without", "avoid", "no"), is there a negation word within a window of 3 tokens |

### Training Data Generation

Since the original dataset was synthetically generated using an LLM, the CRF training data can be bootstrapped automatically through the following procedure:

1. **Generate a large set of diverse natural language queries** covering varied phrasings, indirect references, typos, and different combinations of attributes.
2. **Run the existing LLM parser** on each query to obtain the structured JSON output (key-value pairs with include/exclude labels).
3. **Project the structured output back onto the token sequence** using a string matching aligner: locate each value mention in the original query tokens and assign BIO labels accordingly. Negation tokens adjacent to excluded values receive the `NEG` label.
4. **Review a sample manually** to validate label quality before training.

This procedure allows hundreds or thousands of labelled training examples to be produced with minimal human effort.

### Training and Inference

The CRF is trained using L-BFGS optimisation with L1 and L2 regularisation to prevent overfitting on the generated data. Hyperparameters (regularisation weights `c1` and `c2`, maximum iterations) are tuned via cross-validation.

At inference time, the Viterbi algorithm is used to decode the most probable label sequence for a new query. The decoded BIO spans are then:

1. Mapped to canonical values (exact match first, then fuzzy match or embedding-based nearest-neighbour for unseen surface forms).
2. Grouped by key.
3. Partitioned into `include` and `exclude` based on the negation post-processing step.

### Strengths and Limitations

| Aspect | Detail |
|---|---|
| **Strengths** | Learns from data, generalises beyond the lexicon, models sequential dependencies between labels, interpretable feature weights, well-established academic baseline |
| **Limitations** | Requires labelled training data (addressed via bootstrapping), no deep semantic understanding, limited ability to handle long-range dependencies, semantic expansion still requires a lookup table or embedding step |

---

## Comparison Summary

| Property | Rule-Based (spaCy) | CRF | LLM Parser |
|---|---|---|---|
| Training data required | No | Yes (bootstrapped) | No (prompted) |
| Generalises to unseen vocabulary | No | Partially | Yes |
| Handles negation | Yes (dependency rules) | Yes (NEG tag + window) | Yes (implicitly) |
| Semantic expansion | Manual lookup table | Learned from data | Implicit |
| Interpretability | High | High | Low |
| Implementation complexity | Low | Medium | Low |
| Inference speed | Very fast | Very fast | Slow |
| Academic precedent | Strong (rule-based NLP) | Strong (NER/slot filling) | Emerging |

---

## Recommended Evaluation Protocol

All three models (rule-based, CRF, LLM) should be evaluated on the same held-out set of queries using the following metrics computed per attribute key:

- **Precision** — Of all key-value pairs predicted, what fraction are correct?
- **Recall** — Of all ground-truth key-value pairs, what fraction were retrieved?
- **F1 Score** — Harmonic mean of precision and recall.
- **Negation Accuracy** — Of all correctly detected attribute spans, what fraction have the correct include/exclude polarity?

Evaluation should be reported both **micro-averaged** (across all key-value instances equally) and **macro-averaged** (per key, then averaged), since the attribute schema has significant class imbalance (e.g., `color` and `type` appear far more often than `insulation` or `waterproof`).