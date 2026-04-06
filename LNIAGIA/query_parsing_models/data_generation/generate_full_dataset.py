import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from basic_generator import TemplateBasedGenerator
from bio_projection import project_bio_labels, tokenise_text
from llm_augmented_generator import LLMAugmentedGenerator
from llm_validator import LLMValidator


# Output configuration
OUTPUT_FOLDER = "LNIAGIA/query_parsing_models/data_generation/full_outputs/10000(5000-5000)"
TRAIN_OUTPUT_NAME = "train.json"
VAL_OUTPUT_NAME = "val.json"
TEST_OUTPUT_NAME = "test.json"
FULL_OUTPUT_NAME = "full.json"
STATS_OUTPUT_NAME = "stats.json"

# Optional intermediate outputs
SAVE_INTERMEDIATE_FILES = True
BASIC_RAW_OUTPUT_NAME = "basic_raw.json"
LLM_RAW_OUTPUT_NAME = "llm_raw.json"
LLM_VALIDATED_OUTPUT_NAME = "llm_validated.json"

# Generation sizes
BASIC_GENERATOR_QUERIES = 5000
LLM_GENERATOR_QUERIES = 5000

# Template split (must sum to 1.0)
BASIC_SIMPLE_RATIO = 0.5
BASIC_NEGATION_RATIO = 0.4
BASIC_CONTEXTUAL_RATIO = 0.1

# Validation behavior
VALIDATE_ALL_LLM_DATA = True
LLM_VALIDATION_SAMPLE_RATE = 1.0

# BIO behavior
PROJECT_BIO_FOR_LLM_DATA = True

# Training-target behavior
# Supported values for TRAINING_TARGET: "bio", "filter_parsing"
TRAINING_TARGET = "bio"
# Supported values for BIO_TRAINING_SOURCE: "basic_only", "hybrid_projected"
BIO_TRAINING_SOURCE = "hybrid_projected"

# Canonical export behavior for train/val/test
CANONICAL_EXPORT_MODE = True

# Models and reproducibility
LLM_GENERATION_MODEL = "qwen2.5:7b-instruct-q3_K_M"
LLM_VALIDATION_MODEL = "qwen2.5:7b-instruct-q3_K_M"
SEED = 42

# Dataset split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def _basic_count_split(total: int) -> Tuple[int, int, int]:
    """Split basic-generator examples by configured ratios, preserving exact total."""
    n_simple = int(total * BASIC_SIMPLE_RATIO)
    n_negation = int(total * BASIC_NEGATION_RATIO)
    n_contextual = total - n_simple - n_negation
    return n_simple, n_negation, n_contextual


def _ensure_common_schema(example: Dict, project_bio: bool = False) -> Dict:
    """Ensure every example has common keys for downstream consistency."""
    normalized = dict(example)
    query = normalized.get("query", "")

    include = normalized.get("include", {})
    if not isinstance(include, dict):
        include = {}
    normalized["include"] = include

    exclude = normalized.get("exclude", {})
    if not isinstance(exclude, dict):
        exclude = {}
    normalized["exclude"] = exclude

    tokens = normalized.get("tokens", [])

    if not isinstance(tokens, list):
        tokens = tokenise_text(query) if isinstance(query, str) else []
        normalized["tokens"] = tokens

    bio_labels = normalized.get("bio_labels")
    needs_projection = (
        project_bio
        and isinstance(query, str)
        and (not isinstance(bio_labels, list) or len(bio_labels) != len(tokens) or all(lbl == "O" for lbl in bio_labels))
    )

    if needs_projection:
        normalized["bio_labels"] = project_bio_labels(tokens=tokens, include=include, exclude=exclude)
    elif "bio_labels" not in normalized or not isinstance(normalized.get("bio_labels"), list):
        normalized["bio_labels"] = ["O"] * len(tokens)

    return normalized


def _save_json(data, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _to_canonical_example(example: Dict, index: int) -> Dict:
    """Export-only canonical schema used for train/val/test files."""
    normalized = _ensure_common_schema(example, project_bio=False)

    example_id = normalized.get("id")
    if not isinstance(example_id, str) or not example_id.strip():
        example_id = f"sample_{index:07d}"

    query = normalized.get("query")
    if not isinstance(query, str):
        query = ""

    source = normalized.get("source")
    if not isinstance(source, str) or not source:
        source = "unknown"

    return {
        "id": example_id,
        "query": query,
        "tokens": normalized.get("tokens", []),
        "bio_labels": normalized.get("bio_labels", []),
        "include": normalized.get("include", {}),
        "exclude": normalized.get("exclude", {}),
        "source": source,
    }


def _select_training_examples(
    basic_data: List[Dict],
    llm_validated_data: List[Dict],
    training_target: str,
    bio_training_source: str,
) -> Tuple[List[Dict], str]:
    """Select which data becomes the training pool, based on training objective."""
    if training_target == "bio":
        if bio_training_source == "basic_only":
            return list(basic_data), "bio_basic_only"
        if bio_training_source == "hybrid_projected":
            return list(basic_data) + list(llm_validated_data), "bio_hybrid_projected"
        raise ValueError("BIO_TRAINING_SOURCE must be 'basic_only' or 'hybrid_projected'.")

    if training_target == "filter_parsing":
        return list(basic_data) + list(llm_validated_data), "filter_parsing_hybrid"

    raise ValueError("TRAINING_TARGET must be 'bio' or 'filter_parsing'.")


def generate_full_dataset(
    output_dir: str = OUTPUT_FOLDER,
    n_basic_generator: int = BASIC_GENERATOR_QUERIES,
    n_llm_generator: int = LLM_GENERATOR_QUERIES,
    validate_all_llm_data: bool = VALIDATE_ALL_LLM_DATA,
    llm_validation_sample_rate: float = LLM_VALIDATION_SAMPLE_RATE,
    training_target: str = TRAINING_TARGET,
    bio_training_source: str = BIO_TRAINING_SOURCE,
    canonical_export_mode: bool = CANONICAL_EXPORT_MODE,
    seed: int = SEED,
) -> Tuple[Dict[str, List[Dict]], Dict]:
    """Generate full dataset with basic + LLM data and LLM-only validation."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    all_examples: List[Dict] = []

    if tqdm is None:
        print("tqdm is not installed. Falling back to stage logs without progress bars.")

    pipeline_progress = tqdm(total=7, desc="Full dataset pipeline", unit="step") if tqdm is not None else None

    def _advance_pipeline(step_name: str):
        if pipeline_progress is not None:
            pipeline_progress.update(1)
            pipeline_progress.set_postfix_str(step_name)
        else:
            print(f"[Pipeline] {step_name}")

    try:
        llm_bio_projection_enabled = PROJECT_BIO_FOR_LLM_DATA
        if training_target == "bio" and bio_training_source == "hybrid_projected":
            llm_bio_projection_enabled = True

        # Tier 1: Basic generator
        print("=" * 70)
        print("TIER 1: Basic Generator")
        print("=" * 70)

        n_simple, n_negation, n_contextual = _basic_count_split(n_basic_generator)

        basic_gen = TemplateBasedGenerator(seed=seed)
        basic_examples = basic_gen.generate_dataset(
            n_simple=n_simple,
            n_negation=n_negation,
            n_contextual=n_contextual,
        )

        basic_data: List[Dict] = []
        basic_iter = (
            tqdm(basic_examples, total=len(basic_examples), desc="Formatting basic", unit="example")
            if tqdm is not None
            else basic_examples
        )
        for ex in basic_iter:
            item = {
                "id": ex.id,
                "query": ex.query,
                "tokens": ex.tokens,
                "bio_labels": ex.bio_labels,
                "include": ex.include,
                "exclude": ex.exclude,
                "source": "basic_generator",
            }
            basic_data.append(_ensure_common_schema(item, project_bio=False))

        all_examples.extend(basic_data)
        print(f"Generated {len(basic_data)} basic examples")
        _advance_pipeline("tier1 complete")

        # Tier 2: LLM-augmented generation
        print("\n" + "=" * 70)
        print("TIER 2: LLM-Augmented Generation")
        print("=" * 70)

        llm_gen = LLMAugmentedGenerator(model=LLM_GENERATION_MODEL, seed=seed)
        llm_examples = llm_gen.generate_dataset(n_examples=n_llm_generator)

        llm_raw_data: List[Dict] = []
        llm_iter = (
            tqdm(llm_examples, total=len(llm_examples), desc="Formatting LLM", unit="example")
            if tqdm is not None
            else llm_examples
        )
        for ex in llm_iter:
            item = dict(ex)
            item["source"] = "llm_augmented"
            llm_raw_data.append(_ensure_common_schema(item, project_bio=llm_bio_projection_enabled))

        print(f"Generated {len(llm_raw_data)} LLM-augmented examples")
        _advance_pipeline("tier2 complete")

        # Tier 3: Validate and correct LLM-only data
        print("\n" + "=" * 70)
        print("TIER 3: LLM Validation (LLM-Generated Data Only)")
        print("=" * 70)

        sample_rate = 1.0 if validate_all_llm_data else llm_validation_sample_rate
        validator = LLMValidator(model=LLM_VALIDATION_MODEL, seed=seed)
        llm_validated_data, validation_stats = validator.validate_dataset(
            llm_raw_data,
            sample_rate=sample_rate,
        )

        llm_validated_iter = (
            tqdm(
                llm_validated_data,
                total=len(llm_validated_data),
                desc="Normalizing validated LLM",
                unit="example",
            )
            if tqdm is not None
            else llm_validated_data
        )
        llm_validated_data = [
            _ensure_common_schema(ex, project_bio=llm_bio_projection_enabled)
            for ex in llm_validated_iter
        ]
        _advance_pipeline("tier3 complete")

        # Merge all generated data for diagnostics/full output.
        all_examples.extend(llm_validated_data)

        # Choose the actual training pool based on target strategy.
        training_pool, selected_mode = _select_training_examples(
            basic_data=basic_data,
            llm_validated_data=llm_validated_data,
            training_target=training_target,
            bio_training_source=bio_training_source,
        )

        # Save optional intermediate files
        if SAVE_INTERMEDIATE_FILES:
            intermediate_targets = [
                (BASIC_RAW_OUTPUT_NAME, basic_data),
                (LLM_RAW_OUTPUT_NAME, llm_raw_data),
                (LLM_VALIDATED_OUTPUT_NAME, llm_validated_data),
            ]
            intermediate_iter = (
                tqdm(intermediate_targets, total=len(intermediate_targets), desc="Saving intermediate", unit="file")
                if tqdm is not None
                else intermediate_targets
            )
            for filename, data in intermediate_iter:
                _save_json(data, output_path / filename)
        _advance_pipeline("intermediate outputs")

        # Shuffle and split
        random.Random(seed).shuffle(training_pool)

        if canonical_export_mode:
            canonical_iter = (
                tqdm(training_pool, total=len(training_pool), desc="Canonicalizing train/val/test", unit="example")
                if tqdm is not None
                else training_pool
            )
            export_pool = [_to_canonical_example(ex, idx) for idx, ex in enumerate(canonical_iter, start=1)]
        else:
            export_pool = list(training_pool)

        n_total = len(export_pool)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)

        train_data = export_pool[:n_train]
        val_data = export_pool[n_train : n_train + n_val]
        test_data = export_pool[n_train + n_val :]
        _advance_pipeline("split complete")

        datasets = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "full": all_examples,
        }

        final_outputs = [
            (TRAIN_OUTPUT_NAME, train_data),
            (VAL_OUTPUT_NAME, val_data),
            (TEST_OUTPUT_NAME, test_data),
            (FULL_OUTPUT_NAME, all_examples),
        ]
        final_iter = (
            tqdm(final_outputs, total=len(final_outputs), desc="Saving datasets", unit="file")
            if tqdm is not None
            else final_outputs
        )
        for filename, data in final_iter:
            _save_json(data, output_path / filename)
        _advance_pipeline("datasets saved")

        stats = {
            "seed": seed,
            "total_examples": n_total,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "basic_examples": len(basic_data),
            "full_examples_raw": len(all_examples),
            "llm_raw_examples": len(llm_raw_data),
            "llm_validated_examples": len(llm_validated_data),
            "basic_generator_queries": n_basic_generator,
            "llm_generator_queries": n_llm_generator,
            "training_target": training_target,
            "bio_training_source": bio_training_source,
            "selected_training_mode": selected_mode,
            "canonical_export_mode": canonical_export_mode,
            "project_bio_for_llm_data": llm_bio_projection_enabled,
            "basic_split": {
                "simple": n_simple,
                "negation": n_negation,
                "contextual": n_contextual,
            },
            "llm_validation": {
                "validate_all": validate_all_llm_data,
                "sample_rate_used": sample_rate,
                "stats": validation_stats,
            },
        }

        _save_json(stats, output_path / STATS_OUTPUT_NAME)
        _advance_pipeline("stats saved")

        print("\n" + "=" * 70)
        print("DATASET GENERATION COMPLETE")
        print("=" * 70)
        print(f"Total examples: {n_total}")
        print(f"  Train: {len(train_data)}")
        print(f"  Validation: {len(val_data)}")
        print(f"  Test: {len(test_data)}")
        print(f"  Basic: {len(basic_data)}")
        print(f"  LLM validated: {len(llm_validated_data)}")

        return datasets, stats
    finally:
        if pipeline_progress is not None:
            pipeline_progress.close()


if __name__ == "__main__":
    generate_full_dataset(
        output_dir=OUTPUT_FOLDER,
        n_basic_generator=BASIC_GENERATOR_QUERIES,
        n_llm_generator=LLM_GENERATOR_QUERIES,
        validate_all_llm_data=VALIDATE_ALL_LLM_DATA,
        llm_validation_sample_rate=LLM_VALIDATION_SAMPLE_RATE,
        training_target=TRAINING_TARGET,
        bio_training_source=BIO_TRAINING_SOURCE,
        canonical_export_mode=CANONICAL_EXPORT_MODE,
        seed=SEED,
    )
