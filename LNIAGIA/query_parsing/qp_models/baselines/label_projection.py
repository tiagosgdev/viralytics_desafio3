from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

from .nlp_utils import NEGATION_WORDS, normalise_for_match


def _value_variants(value: str) -> Set[str]:
    base = normalise_for_match(value)
    variants = {
        base,
        base.replace(" ", "-"),
        base.replace(" ", "_"),
    }
    return {variant for variant in variants if variant}


def _tokenized_variants(value: str) -> List[List[str]]:
    tokenized: List[List[str]] = []
    for variant in _value_variants(value):
        pieces = [part for part in normalise_for_match(variant).split() if part]
        if pieces:
            tokenized.append(pieces)
    return tokenized


def project_bio_labels(tokens: Sequence[str], include: Dict[str, List[str]], exclude: Dict[str, List[str]]) -> List[str]:
    labels = ["O"] * len(tokens)
    token_norm = [normalise_for_match(token) for token in tokens]

    for idx, tok in enumerate(token_norm):
        if tok in NEGATION_WORDS:
            labels[idx] = "NEG"

    entries: List[Tuple[List[str], str, bool]] = []

    for key, values in (include or {}).items():
        for value in values:
            for variant_tokens in _tokenized_variants(value):
                entries.append((variant_tokens, key.upper(), False))

    for key, values in (exclude or {}).items():
        for value in values:
            for variant_tokens in _tokenized_variants(value):
                entries.append((variant_tokens, key.upper(), True))

    entries.sort(key=lambda item: len(item[0]), reverse=True)

    i = 0
    while i < len(tokens):
        if labels[i] == "NEG":
            i += 1
            continue

        matched = False
        for variant_tokens, key, is_excluded in entries:
            if i + len(variant_tokens) > len(tokens):
                continue

            segment = token_norm[i : i + len(variant_tokens)]
            if segment != variant_tokens:
                continue

            suffix = "-X" if is_excluded else ""
            labels[i] = f"B-{key}{suffix}"
            for j in range(1, len(variant_tokens)):
                if labels[i + j] != "NEG":
                    labels[i + j] = f"I-{key}{suffix}"

            i += len(variant_tokens)
            matched = True
            break

        if not matched:
            i += 1

    return labels
