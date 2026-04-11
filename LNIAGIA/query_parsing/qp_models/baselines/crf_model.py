from __future__ import annotations

from collections import defaultdict
from difflib import get_close_matches
import importlib
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from .data_utils import QueryExample
from .nlp_utils import (
    NEGATION_WORDS,
    annotate_tokens,
    default_phrase_variants,
    has_negation_in_window,
    has_predicted_negation,
    labels_to_spans,
    load_nlp,
    normalise_for_match,
)
from .rule_based import TYPE_ALIASES


class CRFParser:
    """CRF sequence labeler with BIO tagging and negation post-processing."""

    def __init__(
        self,
        random_state: int = 42,
        max_iterations: int = 120,
        hyperparameter_grid: Sequence[Mapping[str, float]] | None = None,
    ) -> None:
        self.random_state = random_state
        self.max_iterations = max_iterations
        self.hyperparameter_grid = list(hyperparameter_grid) if hyperparameter_grid is not None else [
            {"c1": 0.1, "c2": 0.1},
            {"c1": 0.01, "c2": 0.01},
        ]
        self.nlp = load_nlp()
        self.model = None
        self._crf_cls = None
        self._crf_metrics_module = None

        self.known_values_by_key: Dict[str, Set[str]] = defaultdict(set)
        self.surface_to_keys: Dict[str, Set[str]] = defaultdict(set)
        self.variant_to_canonical: Dict[str, Dict[str, str]] = defaultdict(dict)

    def fit(self, train_samples: Sequence[QueryExample], val_samples: Sequence[QueryExample]) -> dict:
        self._ensure_crf_dependency()

        self._build_lookup_tables(list(train_samples) + list(val_samples))

        x_train, y_train = self._prepare_xy(train_samples)
        x_val, y_val = self._prepare_xy(val_samples)

        if not x_train:
            raise ValueError("No training examples available for CRF training.")

        train_label_candidates = sorted({label for seq in y_train for label in seq if label != "O"})
        val_label_candidates = sorted({label for seq in y_val for label in seq if label != "O"})

        candidates = list(self.hyperparameter_grid)
        if not candidates:
            candidates = [{"c1": 0.1, "c2": 0.1}]

        best_model = None
        best_score = float("-inf")
        best_params = candidates[0]

        for params in candidates:
            model = self._crf_cls(
                algorithm="lbfgs",
                c1=params["c1"],
                c2=params["c2"],
                max_iterations=self.max_iterations,
                all_possible_transitions=True,
            )
            model.fit(x_train, y_train)

            if x_val and y_val and val_label_candidates:
                y_pred = model.predict(x_val)
                score = self._crf_metrics_module.flat_f1_score(
                    y_val,
                    y_pred,
                    average="weighted",
                    labels=val_label_candidates,
                )
            elif x_val and y_val:
                y_pred = model.predict(x_val)
                score = self._crf_metrics_module.flat_accuracy_score(y_val, y_pred)
            else:
                score = 0.0

            if best_model is None or score > best_score:
                best_model = model
                best_score = score
                best_params = params

        self.model = best_model

        return {
            "best_c1": best_params["c1"],
            "best_c2": best_params["c2"],
            "val_weighted_f1": round(best_score, 6),
            "labels_seen": len(train_label_candidates),
        }

    def parse(self, query: str) -> dict:
        if self.model is None:
            raise RuntimeError("CRFParser must be fitted before calling parse().")

        doc = self.nlp(query)
        tokens = [token.text for token in doc]
        if not tokens:
            return {"include": {}, "exclude": {}}

        features = self._token_features(tokens)
        predicted_labels = self.model.predict_single(features)
        predicted_labels = self._fix_label_length(predicted_labels, len(tokens))

        spans = labels_to_spans(tokens, predicted_labels)
        include: Dict[str, Set[str]] = defaultdict(set)
        exclude: Dict[str, Set[str]] = defaultdict(set)

        for span in spans:
            value = self._resolve_value(span.key, span.text)
            if not value:
                continue

            negated = has_predicted_negation(predicted_labels, span.start, span.end, window=3)
            if not negated:
                negated = has_negation_in_window(tokens, span.start, span.end, window=3)

            if negated:
                exclude[span.key].add(value)
            else:
                include[span.key].add(value)

        return {
            "include": {key: sorted(values) for key, values in include.items() if values},
            "exclude": {key: sorted(values) for key, values in exclude.items() if values},
        }

    def _ensure_crf_dependency(self) -> None:
        if self._crf_cls is not None and self._crf_metrics_module is not None:
            return

        try:
            module = importlib.import_module("sklearn_crfsuite")
            metrics_module = importlib.import_module("sklearn_crfsuite.metrics")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'sklearn-crfsuite'. Install it with: pip install sklearn-crfsuite"
            ) from exc

        self._crf_cls = getattr(module, "CRF")
        self._crf_metrics_module = metrics_module

    def _prepare_xy(self, samples: Sequence[QueryExample]) -> Tuple[List[List[dict]], List[List[str]]]:
        x_all: List[List[dict]] = []
        y_all: List[List[str]] = []

        for sample in samples:
            tokens = sample.tokens
            if not tokens:
                continue

            x_all.append(self._token_features(tokens))
            y_all.append([self._normalize_training_label(label) for label in sample.bio_labels])

        return x_all, y_all

    def _token_features(self, tokens: Sequence[str]) -> List[dict]:
        annotations = annotate_tokens(self.nlp, tokens)

        features: List[dict] = []
        for i, token_ann in enumerate(annotations):
            token_text = token_ann["text"]
            token_lower = token_ann["lower"]

            item = {
                "bias": 1.0,
                "token.lower": token_lower,
                "token.lemma": token_ann["lemma"],
                "token.prefix3": token_lower[:3],
                "token.suffix3": token_lower[-3:],
                "token.is_digit": token_ann["is_digit"],
                "token.is_punct": token_ann["is_punct"],
                "token.pos": token_ann["pos"],
                "token.tag": token_ann["tag"],
                "token.dep": token_ann["dep"],
                "token.shape": token_ann["shape"],
                "lexicon.in_schema": token_lower in self.surface_to_keys,
                "lexicon.keys": "|".join(sorted(self.surface_to_keys.get(token_lower, set()))),
                "negation.word": token_lower in NEGATION_WORDS,
                "negation.window": has_negation_in_window(tokens, i, i + 1, window=3),
                "BOS": i == 0,
                "EOS": i == len(tokens) - 1,
            }

            if i > 0:
                prev = annotations[i - 1]
                item.update(
                    {
                        "-1:token.lower": prev["lower"],
                        "-1:token.pos": prev["pos"],
                        "-1:token.dep": prev["dep"],
                    }
                )
            if i < len(tokens) - 1:
                nxt = annotations[i + 1]
                item.update(
                    {
                        "+1:token.lower": nxt["lower"],
                        "+1:token.pos": nxt["pos"],
                        "+1:token.dep": nxt["dep"],
                    }
                )

            features.append(item)

        return features

    @staticmethod
    def _normalize_training_label(label: str) -> str:
        cleaned = label.strip()
        if cleaned.endswith("-X"):
            cleaned = cleaned[:-2]
        if cleaned in {"O", "NEG"}:
            return cleaned
        if cleaned.startswith("B-") or cleaned.startswith("I-"):
            return cleaned
        return "O"

    @staticmethod
    def _fix_label_length(labels: Sequence[str], expected_len: int) -> List[str]:
        fixed = list(labels)
        if len(fixed) < expected_len:
            fixed.extend(["O"] * (expected_len - len(fixed)))
        if len(fixed) > expected_len:
            fixed = fixed[:expected_len]
        return fixed

    def _build_lookup_tables(self, samples: Sequence[QueryExample]) -> None:
        self.known_values_by_key.clear()
        self.surface_to_keys.clear()
        self.variant_to_canonical.clear()

        alias_surface_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for sample in samples:
            key_to_values: Dict[str, List[str]] = defaultdict(list)

            for block_name in ("include", "exclude"):
                block = getattr(sample, block_name)
                for key, values in block.items():
                    key_norm = key.lower().strip()
                    for value in values:
                        value_norm = value.strip()
                        if not key_norm or not value_norm:
                            continue
                        self.known_values_by_key[key_norm].add(value_norm)
                        key_to_values[key_norm].append(value_norm)

            spans = labels_to_spans(sample.tokens, sample.bio_labels)
            for span in spans:
                values = key_to_values.get(span.key, [])
                if len(values) == 1 and span.text:
                    alias_surface_map[(span.key, values[0])].add(span.text)

        for type_value, aliases in TYPE_ALIASES.items():
            if type_value not in self.known_values_by_key.get("type", set()):
                continue
            for alias in aliases:
                alias_surface_map[("type", type_value)].add(alias)

        for key, values in self.known_values_by_key.items():
            for canonical_value in values:
                variants = set(default_phrase_variants(canonical_value))
                variants.update(alias_surface_map.get((key, canonical_value), set()))

                for surface in variants:
                    normalized_surface = normalise_for_match(surface)
                    if not normalized_surface:
                        continue
                    self.variant_to_canonical[key][normalized_surface] = canonical_value

                    # Token-level lexicon features use single-token surfaces.
                    parts = normalized_surface.split()
                    if len(parts) == 1:
                        self.surface_to_keys[parts[0]].add(key)

    def _resolve_value(self, key: str, span_text: str) -> str | None:
        key_norm = key.lower()
        candidate = normalise_for_match(span_text)

        by_variant = self.variant_to_canonical.get(key_norm, {})
        if candidate in by_variant:
            return by_variant[candidate]

        if by_variant:
            close = get_close_matches(candidate, by_variant.keys(), n=1, cutoff=0.82)
            if close:
                return by_variant[close[0]]

        # Fallback: if the exact canonical value itself exists, use it.
        for known in self.known_values_by_key.get(key_norm, set()):
            if normalise_for_match(known) == candidate:
                return known

        return None

    def to_export_state(self) -> dict:
        if self.model is None:
            raise RuntimeError("CRFParser must be fitted before export.")

        return {
            "parser_type": "crf",
            "spacy_model": self.nlp.meta.get("name", "en"),
            "random_state": self.random_state,
            "max_iterations": self.max_iterations,
            "hyperparameter_grid": list(self.hyperparameter_grid),
            "known_values_by_key": {
                key: sorted(values)
                for key, values in self.known_values_by_key.items()
            },
            "surface_to_keys": {
                token: sorted(keys)
                for token, keys in self.surface_to_keys.items()
            },
            "variant_to_canonical": {
                key: dict(value_map)
                for key, value_map in self.variant_to_canonical.items()
            },
            "negation_words": sorted(NEGATION_WORDS),
            "type_aliases": TYPE_ALIASES,
        }

    @classmethod
    def from_export_state(cls, state: dict, model=None) -> "CRFParser":
        parser = cls(
            random_state=int(state.get("random_state", 42)),
            max_iterations=int(state.get("max_iterations", 120)),
            hyperparameter_grid=state.get("hyperparameter_grid"),
        )

        known_values = state.get("known_values_by_key", {})
        parser.known_values_by_key = defaultdict(set)
        if isinstance(known_values, dict):
            for key, values in known_values.items():
                if isinstance(values, list):
                    parser.known_values_by_key[str(key)] = {str(value) for value in values if str(value).strip()}

        surface_to_keys = state.get("surface_to_keys", {})
        parser.surface_to_keys = defaultdict(set)
        if isinstance(surface_to_keys, dict):
            for token, keys in surface_to_keys.items():
                if isinstance(keys, list):
                    parser.surface_to_keys[str(token)] = {str(key) for key in keys if str(key).strip()}

        variant_to_canonical = state.get("variant_to_canonical", {})
        parser.variant_to_canonical = defaultdict(dict)
        if isinstance(variant_to_canonical, dict):
            for key, value_map in variant_to_canonical.items():
                if isinstance(value_map, dict):
                    parser.variant_to_canonical[str(key)] = {
                        str(variant): str(canonical)
                        for variant, canonical in value_map.items()
                        if str(variant).strip() and str(canonical).strip()
                    }

        parser.model = model
        return parser
