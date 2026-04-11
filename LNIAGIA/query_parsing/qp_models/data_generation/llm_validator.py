import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

try:
    from LNIAGIA.DB import models as db_models
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from DB import models as db_models


# I/O configuration
OUTPUT_FOLDER = "LNIAGIA/query_parsing_models/data_generation/single_outputs"
INPUT_NAME = "llm_augmented_data.json"
OUTPUT_NAME = "validated_data.json"
STATS_NAME = "validated_data_stats.json"

# Validation configuration
OLLAMA_MODEL = "qwen2.5:7b-instruct-q3_K_M"
SAMPLE_RATE = 0.2
SEED = 42
TEMPERATURE = 0.1
NUM_PREDICT = 500


class LLMValidator:
    """Validate and optionally correct generated include/exclude annotations with an LLM."""

    VALIDATION_PROMPT = """You are a quality control system for a clothing search dataset.

Given a query and its annotations, verify if the annotations are correct.

QUERY: "{query}"

CURRENT ANNOTATIONS:
Include: {include}
Exclude: {exclude}

VALID FIELDS:
{valid_fields}

VALID CLOSED-SET VALUES (use only these exact values):
{valid_values}

FREE-TEXT FIELDS:
{free_text_fields}

TASK:
1. Check if included attributes are mentioned or clearly implied.
2. Check if excluded attributes have explicit negation in the query.
3. Check if any obvious mentioned attributes are missing.
4. Check if any negated attributes are incorrectly in include.

OUTPUT FORMAT (JSON only, no explanation):
{{
  "is_valid": true/false,
  "issues": ["..."],
  "corrected_include": {{...}},
  "corrected_exclude": {{...}}
}}

If annotations are correct, keep corrected_include/corrected_exclude equal to current ones.
"""

    def __init__(self, model: str = OLLAMA_MODEL, seed: int = SEED):
        self.model = model
        random.seed(seed)

        self.filterable_fields = list(db_models.FILTERABLE_FIELDS)
        self.free_text_fields = list(getattr(db_models, "FREE_TEXT_FILTER_FIELDS", []))

        self.closed_set_values = self._build_closed_set_values()
        self.closed_set_lookup = {
            field: {value.lower(): value for value in values}
            for field, values in self.closed_set_values.items()
        }

    def _build_closed_set_values(self) -> Dict[str, List[str]]:
        """Build allowed closed-set values from models.py for FILTERABLE_FIELDS."""
        values: Dict[str, List[str]] = {}
        for field in self.filterable_fields:
            if field in db_models.STATIC_GLOBAL_FIELDS:
                values[field] = list(db_models.STATIC_GLOBAL_FIELDS[field])
            elif field in db_models.EXTRA_FIELD_VALUES:
                values[field] = list(db_models.EXTRA_FIELD_VALUES[field])
        return values

    def _build_prompt(self, query: str, include: Dict, exclude: Dict) -> str:
        valid_fields = json.dumps(self.filterable_fields, ensure_ascii=False)
        valid_values = json.dumps(self.closed_set_values, ensure_ascii=False, indent=2)
        free_text_fields = json.dumps(self.free_text_fields, ensure_ascii=False)

        return self.VALIDATION_PROMPT.format(
            query=query,
            include=json.dumps(include, ensure_ascii=False),
            exclude=json.dumps(exclude, ensure_ascii=False),
            valid_fields=valid_fields,
            valid_values=valid_values,
            free_text_fields=free_text_fields,
        )

    @staticmethod
    def _extract_raw_content(response) -> str:
        if isinstance(response, dict):
            raw = response.get("message", {}).get("content", "")
        else:
            raw = getattr(getattr(response, "message", None), "content", "")
        return (raw or "").strip()

    @staticmethod
    def _strip_markdown_fences(raw: str) -> str:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()
        return raw

    def _normalise_block(self, block: Optional[Dict]) -> Dict[str, List[str]]:
        """Keep only valid fields and values; normalize casing for closed-set values."""
        if not isinstance(block, dict):
            return {}

        cleaned: Dict[str, List[str]] = {}

        for field, values in block.items():
            if field not in self.filterable_fields and field not in self.free_text_fields:
                continue

            if not isinstance(values, list):
                continue

            if field in self.free_text_fields:
                filtered = []
                seen = set()
                for value in values:
                    if isinstance(value, str):
                        v = value.strip()
                        if v and v.lower() not in seen:
                            filtered.append(v)
                            seen.add(v.lower())
                if filtered:
                    cleaned[field] = filtered
                continue

            lookup = self.closed_set_lookup.get(field, {})
            filtered = []
            seen = set()
            for value in values:
                if not isinstance(value, str):
                    continue
                canonical = lookup.get(value.strip().lower())
                if canonical and canonical not in seen:
                    filtered.append(canonical)
                    seen.add(canonical)

            if filtered:
                cleaned[field] = filtered

        return cleaned

    def _normalise_annotations(
        self, include: Optional[Dict], exclude: Optional[Dict]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        include_clean = self._normalise_block(include)
        exclude_clean = self._normalise_block(exclude)

        # If same value appears in include and exclude for a field, keep it in exclude.
        for field, ex_values in exclude_clean.items():
            if field in include_clean:
                include_clean[field] = [v for v in include_clean[field] if v not in ex_values]
                if not include_clean[field]:
                    del include_clean[field]

        return include_clean, exclude_clean

    def validate_example(self, example: Dict) -> Tuple[bool, Dict]:
        """Validate one example and return (is_valid, output_example)."""
        original_include = example.get("include", {})
        original_exclude = example.get("exclude", {})

        prompt = self._build_prompt(
            query=example.get("query", ""),
            include=original_include,
            exclude=original_exclude,
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You return strict JSON only for annotation validation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
            )

            raw = self._strip_markdown_fences(self._extract_raw_content(response))
            result = json.loads(raw)

            corrected_include, corrected_exclude = self._normalise_annotations(
                include=result.get("corrected_include", original_include),
                exclude=result.get("corrected_exclude", original_exclude),
            )

            issues = result.get("issues", [])
            if not isinstance(issues, list):
                issues = []
            issues = [str(issue) for issue in issues if str(issue).strip()]

            validated = dict(example)
            validated["include"] = corrected_include
            validated["exclude"] = corrected_exclude
            validated["was_validated"] = True

            if bool(result.get("is_valid", False)):
                return True, validated

            validated["validation_issues"] = issues
            validated["was_corrected"] = True
            return False, validated

        except Exception as exc:
            fallback = dict(example)
            include_clean, exclude_clean = self._normalise_annotations(original_include, original_exclude)
            fallback["include"] = include_clean
            fallback["exclude"] = exclude_clean
            fallback["was_validated"] = False
            fallback["was_skipped"] = True
            fallback["validation_error"] = str(exc)
            return False, fallback

    def validate_dataset(
        self, examples: List[Dict], sample_rate: float = SAMPLE_RATE
    ) -> Tuple[List[Dict], Dict]:
        """Validate a dataset and return corrected data plus summary stats."""
        if sample_rate <= 0:
            sample_indices = set()
        elif sample_rate >= 1.0:
            sample_indices = set(range(len(examples)))
        else:
            n_sample = max(1, int(len(examples) * sample_rate))
            sample_indices = set(random.sample(range(len(examples)), n_sample))

        validated_examples: List[Dict] = []
        issue_counter: Counter = Counter()

        stats = {
            "total": len(examples),
            "sample_rate": sample_rate,
            "selected_for_validation": len(sample_indices),
            "attempted": 0,
            "validated": 0,
            "valid": 0,
            "corrected": 0,
            "skipped": 0,
            "validation_errors": 0,
            "issues": [],
        }

        print(f"Validating {len(sample_indices)} examples...")

        progress = tqdm(total=len(sample_indices), desc="Validating", unit="example") if tqdm is not None else None
        if progress is None and len(sample_indices) > 0:
            print("tqdm is not installed. Falling back to periodic progress logs.")

        try:
            for idx, example in enumerate(examples):
                if idx in sample_indices:
                    is_valid, result = self.validate_example(example)
                    stats["attempted"] += 1

                    if result.get("validation_error") or result.get("was_skipped"):
                        stats["validation_errors"] += 1
                        stats["skipped"] += 1
                        validated_examples.append(result)
                    else:
                        stats["validated"] += 1

                        if is_valid:
                            stats["valid"] += 1
                        else:
                            stats["corrected"] += 1
                            for issue in result.get("validation_issues", []):
                                issue_counter[issue] += 1

                        validated_examples.append(result)

                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            corrected=stats["corrected"],
                            skipped=stats["skipped"],
                            errors=stats["validation_errors"],
                        )
                    elif stats["attempted"] % 100 == 0:
                        print(f"  Validated {stats['attempted']}/{len(sample_indices)}")
                else:
                    validated_examples.append(example)
        finally:
            if progress is not None:
                progress.close()

        stats["issues"] = [
            {"issue": issue, "count": count}
            for issue, count in issue_counter.most_common()
        ]

        print("\nValidation complete:")
        print(f"  Total: {stats['total']}")
        print(f"  Selected for validation: {stats['selected_for_validation']}")
        print(f"  Attempted: {stats['attempted']}")
        print(f"  Validated: {stats['validated']}")
        print(f"  Valid: {stats['valid']} ({100 * stats['valid'] / max(1, stats['validated']):.1f}%)")
        print(
            f"  Corrected: {stats['corrected']} ({100 * stats['corrected'] / max(1, stats['validated']):.1f}%)"
        )
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Validation errors: {stats['validation_errors']}")

        return validated_examples, stats

    @staticmethod
    def save_dataset(examples: List[Dict], filepath: str):
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(examples, file, indent=2, ensure_ascii=False)
        print(f"Saved {len(examples)} examples to {output_path}")

    @staticmethod
    def save_stats(stats: Dict, filepath: str):
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2, ensure_ascii=False)
        print(f"Saved validation stats to {output_path}")


if __name__ == "__main__":
    input_path = Path(OUTPUT_FOLDER) / INPUT_NAME
    output_path = Path(OUTPUT_FOLDER) / OUTPUT_NAME
    stats_path = Path(OUTPUT_FOLDER) / STATS_NAME

    with input_path.open("r", encoding="utf-8") as file:
        examples = json.load(file)

    validator = LLMValidator(model=OLLAMA_MODEL, seed=SEED)
    validated_examples, stats = validator.validate_dataset(examples, sample_rate=SAMPLE_RATE)

    validator.save_dataset(validated_examples, str(output_path))
    validator.save_stats(stats, str(stats_path))
