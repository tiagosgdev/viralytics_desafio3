from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

from spacy.matcher import PhraseMatcher

from .data_utils import FilterDict, QueryExample
from .nlp_utils import (
    NEGATION_WORDS,
    default_phrase_variants,
    labels_to_spans,
    load_nlp,
    normalise_for_match,
    token_is_negation,
)


TYPE_ALIASES = {
    "short_sleeve_top": ["t-shirt", "tee", "tees", "tshirt", "top"],
    "long_sleeve_top": ["shirt", "blouse", "sweater", "sweatshirt"],
    "short_sleeve_outwear": ["short sleeve jacket"],
    "long_sleeve_outwear": ["jacket", "coat", "outerwear", "hoodie"],
    "vest": ["tank top", "sleeveless top", "tank"],
    "shorts": ["shorts"],
    "trousers": ["pants", "jeans", "slacks", "chinos", "trouser"],
    "skirt": ["skirt"],
    "short_sleeve_dress": ["short sleeve dress"],
    "long_sleeve_dress": ["long sleeve dress"],
    "vest_dress": ["sleeveless dress"],
    "sling_dress": ["slip dress", "strappy dress", "sling dress"],
}


SEMANTIC_HINTS = {
    "dark": {
        "color": ["black", "navy", "burgundy", "olive", "brown", "gray", "charcoal"],
    },
    "light": {
        "color": ["white", "cream", "beige", "yellow", "pastel"],
    },
    "warm": {
        "color": ["red", "orange", "coral", "burgundy"],
    },
    "cool": {
        "color": ["blue", "green", "navy", "teal"],
    },
}


@dataclass(frozen=True)
class SpanMatch:
    start: int
    end: int
    key: str
    value: str


class RuleBasedParser:
    """Deterministic parser using phrase matching + negation heuristics."""

    def __init__(self) -> None:
        self.nlp = load_nlp()
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.known_values_by_key: Dict[str, Set[str]] = defaultdict(set)
        self.surface_forms_by_pair: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        self.semantic_expansion: Dict[str, List[Tuple[str, str]]] = {}
        self.match_id_to_pair: Dict[int, Tuple[str, str]] = {}
        self._is_fitted = False

    def fit(self, train_samples: Sequence[QueryExample], val_samples: Sequence[QueryExample]) -> None:
        samples = list(train_samples) + list(val_samples)

        self.known_values_by_key.clear()
        self.match_id_to_pair.clear()
        self.surface_forms_by_pair = defaultdict(dict)
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        alias_surface_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for sample in samples:
            for block_name in ("include", "exclude"):
                block = getattr(sample, block_name)
                for key, values in block.items():
                    for value in values:
                        key_norm = key.strip().lower()
                        value_norm = value.strip()
                        if not key_norm or not value_norm:
                            continue
                        self.known_values_by_key[key_norm].add(value_norm)

            # Mine additional surface forms from BIO labels when they align to a single value.
            spans = labels_to_spans(sample.tokens, sample.bio_labels)
            key_to_values: Dict[str, List[str]] = defaultdict(list)
            for block_name in ("include", "exclude"):
                block = getattr(sample, block_name)
                for key, values in block.items():
                    key_to_values[key.lower()].extend(values)

            for span in spans:
                values = key_to_values.get(span.key, [])
                if len(values) == 1 and span.text:
                    alias_surface_map[(span.key, values[0])].add(span.text)

        # Inject well-known type aliases only when the canonical type exists in this dataset.
        for type_value, aliases in TYPE_ALIASES.items():
            if type_value not in self.known_values_by_key.get("type", set()):
                continue
            for alias in aliases:
                alias_surface_map[("type", type_value)].add(alias)

        for key, values in self.known_values_by_key.items():
            for canonical_value in sorted(values):
                all_surface_forms = set(default_phrase_variants(canonical_value))
                all_surface_forms.update(alias_surface_map.get((key, canonical_value), set()))

                cleaned_surface_forms: List[str] = []
                for surface in sorted(all_surface_forms):
                    cleaned = normalise_for_match(surface)
                    if not cleaned:
                        continue
                    cleaned_surface_forms.append(cleaned)

                if cleaned_surface_forms:
                    self.surface_forms_by_pair[key][canonical_value] = sorted(set(cleaned_surface_forms))

        if not self.surface_forms_by_pair:
            self._build_surface_forms_from_known_values()

        self._rebuild_matcher_from_surface_forms()

        self.semantic_expansion = self._build_semantic_expansion()
        self._is_fitted = True

    def parse(self, query: str) -> FilterDict:
        if not self._is_fitted:
            raise RuntimeError("RuleBasedParser must be fitted before calling parse().")

        doc = self.nlp(query)
        tokens = [token.text for token in doc]

        candidates: Set[SpanMatch] = set()

        for match_id, start, end in self.matcher(doc):
            if match_id not in self.match_id_to_pair:
                continue
            key, value = self.match_id_to_pair[match_id]
            candidates.add(SpanMatch(start=start, end=end, key=key, value=value))

        for idx, token in enumerate(doc):
            hint_key = normalise_for_match(token.text)
            for key, value in self.semantic_expansion.get(hint_key, []):
                candidates.add(SpanMatch(start=idx, end=idx + 1, key=key, value=value))

        include: Dict[str, Set[str]] = defaultdict(set)
        exclude: Dict[str, Set[str]] = defaultdict(set)

        for candidate in sorted(candidates, key=lambda item: (item.start, item.end, item.key, item.value)):
            negated = self._is_negated(doc, tokens, candidate.start, candidate.end)
            if negated:
                exclude[candidate.key].add(candidate.value)
            else:
                include[candidate.key].add(candidate.value)

        return {
            "include": {key: sorted(values) for key, values in include.items() if values},
            "exclude": {key: sorted(values) for key, values in exclude.items() if values},
        }

    @staticmethod
    def _pattern_name(key: str, value: str) -> str:
        return f"{key}::{value}".lower()

    def _build_surface_forms_from_known_values(self) -> None:
        self.surface_forms_by_pair = defaultdict(dict)

        for key, values in self.known_values_by_key.items():
            for canonical_value in sorted(values):
                variants = set(default_phrase_variants(canonical_value))

                if key == "type" and canonical_value in TYPE_ALIASES:
                    variants.update(TYPE_ALIASES[canonical_value])

                cleaned_variants = []
                for surface in sorted(variants):
                    cleaned = normalise_for_match(surface)
                    if cleaned:
                        cleaned_variants.append(cleaned)

                if cleaned_variants:
                    self.surface_forms_by_pair[key][canonical_value] = sorted(set(cleaned_variants))

    def _rebuild_matcher_from_surface_forms(self) -> None:
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.match_id_to_pair.clear()

        for key, value_map in self.surface_forms_by_pair.items():
            for canonical_value, surface_forms in value_map.items():
                docs = []
                for surface in surface_forms:
                    cleaned = normalise_for_match(surface)
                    if cleaned:
                        docs.append(self.nlp.make_doc(cleaned))

                if not docs:
                    continue

                pattern_name = self._pattern_name(key, canonical_value)
                self.matcher.add(pattern_name, docs)
                match_id = self.nlp.vocab.strings[pattern_name]
                self.match_id_to_pair[match_id] = (key, canonical_value)

    def _build_semantic_expansion(self) -> Dict[str, List[Tuple[str, str]]]:
        expansion: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for hint_token, hint_mapping in SEMANTIC_HINTS.items():
            for key, hinted_values in hint_mapping.items():
                available = self.known_values_by_key.get(key, set())
                for value in hinted_values:
                    if value in available:
                        expansion[hint_token].append((key, value))

        return dict(expansion)

    def to_export_state(self) -> dict:
        if not self._is_fitted:
            raise RuntimeError("RuleBasedParser must be fitted before export.")

        return {
            "parser_type": "rule_based",
            "spacy_model": self.nlp.meta.get("name", "en"),
            "known_values_by_key": {
                key: sorted(values)
                for key, values in self.known_values_by_key.items()
            },
            "surface_forms_by_pair": {
                key: {
                    value: list(surface_forms)
                    for value, surface_forms in value_map.items()
                }
                for key, value_map in self.surface_forms_by_pair.items()
            },
            "semantic_expansion": {
                token: [[key, value] for key, value in pairs]
                for token, pairs in self.semantic_expansion.items()
            },
            "negation_words": sorted(NEGATION_WORDS),
            "type_aliases": TYPE_ALIASES,
            "semantic_hints": SEMANTIC_HINTS,
        }

    @classmethod
    def from_export_state(cls, state: dict) -> "RuleBasedParser":
        parser = cls()

        known_values = state.get("known_values_by_key", {})
        parser.known_values_by_key = defaultdict(set)
        if isinstance(known_values, dict):
            for key, values in known_values.items():
                if isinstance(values, list):
                    parser.known_values_by_key[str(key)] = {str(value) for value in values if str(value).strip()}

        raw_surface_forms = state.get("surface_forms_by_pair", {})
        parser.surface_forms_by_pair = defaultdict(dict)
        if isinstance(raw_surface_forms, dict):
            for key, value_map in raw_surface_forms.items():
                if not isinstance(value_map, dict):
                    continue
                for canonical_value, surfaces in value_map.items():
                    if not isinstance(surfaces, list):
                        continue
                    cleaned = []
                    for surface in surfaces:
                        normalized = normalise_for_match(str(surface))
                        if normalized:
                            cleaned.append(normalized)
                    if cleaned:
                        parser.surface_forms_by_pair[str(key)][str(canonical_value)] = sorted(set(cleaned))

        if not parser.surface_forms_by_pair:
            parser._build_surface_forms_from_known_values()

        parser._rebuild_matcher_from_surface_forms()

        raw_semantic = state.get("semantic_expansion", {})
        semantic_expansion: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        if isinstance(raw_semantic, dict):
            for token, pairs in raw_semantic.items():
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    if isinstance(pair, list) and len(pair) == 2:
                        semantic_expansion[str(token)].append((str(pair[0]), str(pair[1])))

        parser.semantic_expansion = dict(semantic_expansion) if semantic_expansion else parser._build_semantic_expansion()
        parser._is_fitted = True

        return parser

    def _is_negated(self, doc, tokens: Sequence[str], start: int, end: int) -> bool:
        left = max(0, start - 4)

        for idx in range(left, start):
            if token_is_negation(tokens[idx]):
                return True

        # Dependency-based check when the parser is available.
        for token in doc[start:end]:
            if token.dep_ == "neg":
                return True
            for child in token.children:
                if child.dep_ == "neg" or child.text.lower() in NEGATION_WORDS:
                    return True
            if token.head is not None:
                for child in token.head.children:
                    if child.dep_ == "neg" or child.text.lower() in NEGATION_WORDS:
                        return True

        return False
