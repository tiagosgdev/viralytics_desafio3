from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import spacy
from spacy.tokens import Doc


NEGATION_WORDS = {
    "not",
    "no",
    "without",
    "avoid",
    "never",
    "dont",
    "don't",
    "cannot",
    "can't",
    "except",
}


@dataclass
class SpanPrediction:
    start: int
    end: int
    key: str
    text: str


def load_nlp(model_name: str = "en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        return spacy.blank("en")


def annotate_tokens(nlp, tokens: Sequence[str]) -> List[dict]:
    doc = Doc(nlp.vocab, words=list(tokens))

    try:
        for _, pipe in nlp.pipeline:
            pipe(doc)
    except Exception:
        # If a component fails, still return lexical fallback features.
        pass

    features: List[dict] = []
    for token in doc:
        features.append(
            {
                "text": token.text,
                "lower": token.text.lower(),
                "lemma": (token.lemma_ or token.text).lower(),
                "pos": token.pos_ or "",
                "tag": token.tag_ or "",
                "dep": token.dep_ or "",
                "is_digit": token.is_digit,
                "is_punct": token.is_punct,
                "shape": token.shape_ or "",
            }
        )

    return features


def strip_exclusion_suffix(label: str) -> str:
    if label.endswith("-X"):
        return label[:-2]
    return label


def get_label_key(label: str) -> Optional[str]:
    normalized = strip_exclusion_suffix(label)
    if normalized.startswith("B-") or normalized.startswith("I-"):
        return normalized.split("-", maxsplit=1)[1]
    return None


def labels_to_spans(tokens: Sequence[str], labels: Sequence[str]) -> List[SpanPrediction]:
    spans: List[SpanPrediction] = []
    i = 0

    while i < len(tokens):
        label = labels[i] if i < len(labels) else "O"
        normalized = strip_exclusion_suffix(label)

        if normalized.startswith("B-"):
            key = normalized.split("-", maxsplit=1)[1]
            start = i
            end = i + 1

            j = i + 1
            while j < len(tokens):
                next_label = strip_exclusion_suffix(labels[j] if j < len(labels) else "O")
                if next_label == f"I-{key}":
                    end = j + 1
                    j += 1
                    continue
                break

            text = " ".join(tokens[start:end]).strip()
            spans.append(SpanPrediction(start=start, end=end, key=key.lower(), text=text))
            i = end
            continue

        i += 1

    return spans


def token_is_negation(token: str) -> bool:
    lowered = token.strip().lower()
    return lowered in NEGATION_WORDS


def has_negation_in_window(tokens: Sequence[str], start: int, end: int, window: int = 3) -> bool:
    left = max(0, start - window)
    right = min(len(tokens), end + window)

    for idx in range(left, right):
        if token_is_negation(tokens[idx]):
            return True
    return False


def has_predicted_negation(labels: Sequence[str], start: int, end: int, window: int = 3) -> bool:
    left = max(0, start - window)
    right = min(len(labels), end + window)

    for idx in range(left, right):
        if labels[idx] == "NEG":
            return True
    return False


def normalise_for_match(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("-", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned


def default_phrase_variants(text: str) -> List[str]:
    normalized = normalise_for_match(text)
    if not normalized:
        return []

    variants = {
        normalized,
        normalized.replace(" ", "-"),
        normalized.replace(" ", "_"),
    }
    return sorted(variants)
