from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


FilterDict = Dict[str, List[str]]
StructuredPair = Tuple[str, str, str]


@dataclass
class QueryExample:
    example_id: str
    query: str
    tokens: List[str]
    bio_labels: List[str]
    include: FilterDict
    exclude: FilterDict
    source: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_value(value: object) -> str:
    return str(value).strip()


def normalize_filter_block(block: object) -> FilterDict:
    normalized: FilterDict = {}
    if not isinstance(block, Mapping):
        return normalized

    for key, values in block.items():
        key_str = normalize_value(key)
        if not key_str:
            continue
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue

        cleaned: List[str] = []
        seen: Set[str] = set()
        for item in values:
            value = normalize_value(item)
            value_norm = value.lower()
            if not value or value_norm in seen:
                continue
            seen.add(value_norm)
            cleaned.append(value)

        if cleaned:
            normalized[key_str] = cleaned

    return normalized


def normalize_example(raw: Mapping[str, object]) -> QueryExample:
    query = normalize_value(raw.get("query", ""))

    tokens_raw = raw.get("tokens")
    tokens: List[str]
    if isinstance(tokens_raw, Sequence) and not isinstance(tokens_raw, (str, bytes)):
        tokens = [normalize_value(token) for token in tokens_raw]
    else:
        tokens = query.split() if query else []

    labels_raw = raw.get("bio_labels")
    if isinstance(labels_raw, Sequence) and not isinstance(labels_raw, (str, bytes)):
        bio_labels = [normalize_value(label) for label in labels_raw]
    else:
        bio_labels = ["O"] * len(tokens)

    if len(bio_labels) != len(tokens):
        # Keep the same sequence length to avoid downstream alignment errors.
        bio_labels = (bio_labels + ["O"] * len(tokens))[: len(tokens)]

    include = normalize_filter_block(raw.get("include", {}))
    exclude = normalize_filter_block(raw.get("exclude", {}))

    example_id = normalize_value(raw.get("id", ""))
    if not example_id:
        example_id = f"example_{hash(query) & 0xFFFFFFFF:08x}"

    source = normalize_value(raw.get("source", "unknown")) or "unknown"

    return QueryExample(
        example_id=example_id,
        query=query,
        tokens=tokens,
        bio_labels=bio_labels,
        include=include,
        exclude=exclude,
        source=source,
    )


def load_examples(file_path: Path) -> List[QueryExample]:
    with file_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected a list in {file_path}, got {type(raw_data).__name__}.")

    return [normalize_example(item) for item in raw_data if isinstance(item, Mapping)]


def examples_to_serializable(examples: Iterable[QueryExample]) -> List[dict]:
    serializable: List[dict] = []
    for ex in examples:
        serializable.append(
            {
                "id": ex.example_id,
                "query": ex.query,
                "tokens": ex.tokens,
                "bio_labels": ex.bio_labels,
                "include": ex.include,
                "exclude": ex.exclude,
                "source": ex.source,
            }
        )
    return serializable


def write_json(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_total_size_from_folder(folder: Path) -> int:
    stats_path = folder / "stats.json"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        total_examples = payload.get("total_examples")
        if isinstance(total_examples, int):
            return total_examples

    match = re.match(r"(\d+)", folder.name)
    if not match:
        raise ValueError(f"Unable to infer dataset size from folder {folder}.")
    return int(match.group(1))


def discover_dataset_folders(full_outputs_dir: Path) -> Dict[int, Path]:
    discovered: Dict[int, Path] = {}

    if not full_outputs_dir.exists():
        raise FileNotFoundError(f"full_outputs folder does not exist: {full_outputs_dir}")

    for child in sorted(full_outputs_dir.iterdir()):
        if not child.is_dir():
            continue

        required = [child / "train.json", child / "val.json", child / "test.json", child / "stats.json"]
        if not all(path.exists() for path in required):
            continue

        total_size = extract_total_size_from_folder(child)
        discovered[total_size] = child

    return discovered


def flatten_structured_pairs(include: FilterDict, exclude: FilterDict) -> Set[StructuredPair]:
    flattened: Set[StructuredPair] = set()

    for key, values in include.items():
        key_norm = normalize_value(key).lower()
        for value in values:
            flattened.add(("include", key_norm, normalize_value(value).lower()))

    for key, values in exclude.items():
        key_norm = normalize_value(key).lower()
        for value in values:
            flattened.add(("exclude", key_norm, normalize_value(value).lower()))

    return flattened


def fold_pairs_by_key(pairs: Set[StructuredPair]) -> Dict[str, Set[Tuple[str, str]]]:
    by_key: Dict[str, Set[Tuple[str, str]]] = {}
    for polarity, key, value in pairs:
        by_key.setdefault(key, set()).add((polarity, value))
    return by_key


def plain_pairs(pairs: Set[StructuredPair]) -> Set[Tuple[str, str]]:
    return {(key, value) for _, key, value in pairs}


def build_leakage_report(train: List[QueryExample], val: List[QueryExample], test: List[QueryExample]) -> dict:
    train_ids = {sample.example_id for sample in train}
    val_ids = {sample.example_id for sample in val}
    test_ids = {sample.example_id for sample in test}

    train_queries = {sample.query.strip().lower() for sample in train}
    val_queries = {sample.query.strip().lower() for sample in val}
    test_queries = {sample.query.strip().lower() for sample in test}

    train_val_ids = train_ids | val_ids
    train_val_queries = train_queries | val_queries

    overlap_ids = sorted(train_val_ids & test_ids)
    overlap_queries = sorted(train_val_queries & test_queries)

    return {
        "overlap_id_count": len(overlap_ids),
        "overlap_query_count": len(overlap_queries),
        "overlap_ids_preview": overlap_ids[:20],
        "overlap_queries_preview": overlap_queries[:20],
    }
