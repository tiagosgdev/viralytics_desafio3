from pathlib import Path
from typing import Dict, List, Set, Tuple

import spacy

try:
    from LNIAGIA.DB import models as db_models
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from DB import models as db_models


def _load_nlp():
    """Use en_core_web_sm when available, fallback to blank English tokenizer."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


nlp = _load_nlp()

NEGATION_WORDS = {"not", "no", "without", "avoid", "never", "nothing", "anything", "don't", "dont"}
SEPARATOR_TOKENS = {"-", "/", "&"}

TYPE_QUERY_ALIASES = {
    "short_sleeve_top": ["t-shirt", "tee"],
    "long_sleeve_top": ["shirt", "blouse", "sweater"],
    "long_sleeve_outwear": ["jacket", "coat", "outerwear", "hoodie"],
    "vest": ["tank top", "sleeveless top", "top"],
    "shorts": ["shorts"],
    "trousers": ["pants", "trousers", "jeans", "slacks", "chinos"],
    "skirt": ["skirt"],
    "short_sleeve_dress": ["short sleeve dress"],
    "long_sleeve_dress": ["long sleeve dress"],
    "vest_dress": ["sleeveless dress"],
    "sling_dress": ["slip dress", "strappy dress"],
}


def tokenise_text(text: str) -> List[str]:
    """Tokenise text with the shared spaCy tokenizer."""
    doc = nlp(text)
    return [token.text for token in doc]


def _pluralise(term: str) -> str:
    term = term.strip().lower()
    if not term:
        return term
    if term.endswith("s"):
        return term
    if term.endswith("y") and len(term) > 1 and term[-2] not in "aeiou":
        return term[:-1] + "ies"
    if term.endswith(("ch", "sh", "x", "z")):
        return term + "es"
    return term + "s"


def _value_variants(field: str, value: str) -> Set[str]:
    variants: Set[str] = set()

    if not isinstance(value, str):
        return variants

    base = value.strip().lower()
    if not base:
        return variants

    variants.add(base)
    variants.add(base.replace("_", " "))
    variants.add(base.replace("-", " "))
    variants.add(base.replace("-", " - "))

    if field == "type":
        schema_type = base

        if schema_type in db_models.TYPE:
            human = schema_type.replace("_", " ")
            variants.add(human)
            variants.add(_pluralise(human))
            for alias in TYPE_QUERY_ALIASES.get(schema_type, []):
                alias_l = alias.lower()
                variants.add(alias_l)
                variants.add(_pluralise(alias_l))
        else:
            variants.add(_pluralise(base))

    return {v for v in variants if v}


def _is_content_token(token: str) -> bool:
    return any(char.isalnum() for char in token)


def _match_tokens_with_separators(
    token_lower: List[str], start_idx: int, value_tokens: List[str]
) -> List[int] | None:
    """Match value tokens at a position, allowing separator tokens in the query."""
    q_idx = start_idx
    v_idx = 0
    matched_indices: List[int] = []

    while q_idx < len(token_lower) and v_idx < len(value_tokens):
        query_tok = token_lower[q_idx]
        value_tok = value_tokens[v_idx]

        if query_tok == value_tok:
            matched_indices.append(q_idx)
            q_idx += 1
            v_idx += 1
            continue

        # Allow punctuation separators between matched words, e.g. "t - shirt".
        if query_tok in SEPARATOR_TOKENS and v_idx > 0:
            matched_indices.append(q_idx)
            q_idx += 1
            continue

        return None

    if v_idx != len(value_tokens):
        return None

    return matched_indices


def project_bio_labels(
    tokens: List[str],
    include: Dict[str, List[str]],
    exclude: Dict[str, List[str]],
) -> List[str]:
    """Project BIO labels from include/exclude metadata onto tokenized query text."""

    labels = ["O"] * len(tokens)
    token_lower = [t.lower() for t in tokens]

    for idx, tok in enumerate(token_lower):
        if tok in NEGATION_WORDS:
            labels[idx] = "NEG"

    entries: List[Tuple[List[str], str, bool]] = []

    def _collect(section: Dict[str, List[str]], is_excluded: bool):
        for field, values in section.items():
            key = field.upper()
            for value in values:
                for variant in _value_variants(field, value):
                    variant_tokens = variant.split()
                    if variant_tokens:
                        entries.append((variant_tokens, key, is_excluded))

    _collect(include or {}, False)
    _collect(exclude or {}, True)

    # Prefer longer spans first (e.g. "smart casual" before "casual").
    entries.sort(key=lambda item: len(item[0]), reverse=True)

    i = 0
    while i < len(tokens):
        if labels[i] == "NEG":
            i += 1
            continue

        matched = False

        for value_tokens, key, is_excluded in entries:
            matched_indices = _match_tokens_with_separators(token_lower, i, value_tokens)
            if not matched_indices:
                continue

            suffix = "-X" if is_excluded else ""

            label_positions = [
                idx for idx in matched_indices
                if labels[idx] != "NEG" and _is_content_token(tokens[idx])
            ]
            if not label_positions:
                continue

            first_idx = label_positions[0]
            labels[first_idx] = f"B-{key}{suffix}"
            for idx in label_positions[1:]:
                labels[idx] = f"I-{key}{suffix}"

            i = max(matched_indices) + 1
            matched = True
            break

        if not matched:
            i += 1

    return labels


def project_query_bio(
    query: str,
    include: Dict[str, List[str]],
    exclude: Dict[str, List[str]],
) -> Tuple[List[str], List[str]]:
    """Tokenise a query and project BIO labels in one step."""
    tokens = tokenise_text(query)
    labels = project_bio_labels(tokens=tokens, include=include, exclude=exclude)
    return tokens, labels
