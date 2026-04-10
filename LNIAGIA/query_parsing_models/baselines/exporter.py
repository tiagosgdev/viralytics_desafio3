from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from .crf_model import CRFParser
from .data_utils import ensure_dir, write_json
from .rule_based import RuleBasedParser

EXPORTS_SUBDIR = "exported_parsers"
LATEST_POINTER_FILE = "latest.json"
MANIFEST_FILE = "manifest.json"
STATE_FILE = "parser_state.json"
CRF_MODEL_FILE = "crf_model.pkl"
ARTIFACT_VERSION = 1


def _to_relative_path_str(path: Path, base: Path | None = None) -> str:
    root = base or Path.cwd()

    try:
        return str(path.relative_to(root))
    except ValueError:
        pass

    try:
        return os.path.relpath(path, root)
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected an object in {path}, got {type(payload).__name__}.")
    return payload


def _safe_numeric(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        if not text:
            return None
        if "." in text:
            return float(text)
        return int(text)
    except (TypeError, ValueError):
        return None


def _normalise_row(row: Mapping[str, Any]) -> dict:
    normalized = dict(row)

    for key in (
        "folder_size",
        "structured_micro_f1",
        "structured_macro_f1",
        "latency_p95_ms",
        "quality_score",
        "tradeoff_score",
        "fit_time_s",
    ):
        if key in normalized:
            numeric = _safe_numeric(normalized[key])
            if numeric is not None:
                normalized[key] = numeric

    normalized["model"] = str(normalized.get("model", "")).strip().lower()
    normalized["folder_name"] = str(normalized.get("folder_name", "")).strip()

    return normalized


def _allocate_artifact_dir(exports_root: Path, model_name: str, folder_size: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{model_name}_size{folder_size}"

    candidate = exports_root / base_name
    suffix = 1
    while candidate.exists():
        candidate = exports_root / f"{base_name}_{suffix}"
        suffix += 1

    ensure_dir(candidate)
    return candidate


def export_winner_parser(
    parser: RuleBasedParser | CRFParser,
    *,
    selected_row: Mapping[str, Any],
    results_dir: Path,
    full_outputs_dir: Path,
    fixed_test_path: Path,
    fixed_test_sha256: str,
    fixed_test_selector: str,
    requested_sizes: Sequence[int],
) -> Path:
    row = _normalise_row(selected_row)

    model_name = str(row.get("model", "")).lower()
    folder_size = int(row.get("folder_size", 0))

    if model_name not in {"rule_based", "crf"}:
        raise ValueError(f"Unsupported model for export: {model_name}")

    exports_root = results_dir / EXPORTS_SUBDIR
    ensure_dir(exports_root)

    artifact_dir = _allocate_artifact_dir(exports_root, model_name, folder_size)

    state_payload = parser.to_export_state()
    state_path = artifact_dir / STATE_FILE
    write_json(state_path, state_payload)

    files_block = {
        "state": STATE_FILE,
    }

    if model_name == "crf":
        model_obj = getattr(parser, "model", None)
        if model_obj is None:
            raise RuntimeError("CRF parser has no trained model to export.")

        model_path = artifact_dir / CRF_MODEL_FILE
        with model_path.open("wb") as handle:
            pickle.dump(model_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        files_block["model"] = CRF_MODEL_FILE

    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": timestamp_utc,
        "model": model_name,
        "folder_size": folder_size,
        "folder_name": row.get("folder_name", ""),
        "selection_metric": "tradeoff_score",
        "selection_row": row,
        "results_dir": _to_relative_path_str(results_dir),
        "artifact_dir": artifact_dir.name,
        "benchmark_context": {
            "full_outputs_dir": _to_relative_path_str(full_outputs_dir),
            "requested_sizes": [int(size) for size in requested_sizes],
            "fixed_test_path": _to_relative_path_str(fixed_test_path),
            "fixed_test_sha256": fixed_test_sha256,
            "fixed_test_selector": fixed_test_selector,
        },
        "files": files_block,
    }

    manifest_path = artifact_dir / MANIFEST_FILE
    write_json(manifest_path, manifest)

    latest_payload = {
        "updated_at_utc": timestamp_utc,
        "manifest_path": f"{artifact_dir.name}/{MANIFEST_FILE}",
        "model": model_name,
        "folder_size": folder_size,
        "tradeoff_score": row.get("tradeoff_score", 0.0),
    }
    write_json(exports_root / LATEST_POINTER_FILE, latest_payload)

    return manifest_path


def load_parser_from_manifest(manifest_path: Path) -> Tuple[RuleBasedParser | CRFParser, dict]:
    manifest = _read_json(manifest_path)
    model_name = str(manifest.get("model", "")).strip().lower()

    files = manifest.get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError(f"Invalid files block in {manifest_path}.")

    state_file = files.get("state")
    if not isinstance(state_file, str) or not state_file.strip():
        raise ValueError(f"Missing parser state file in {manifest_path}.")

    artifact_dir = manifest_path.parent
    state_payload = _read_json(artifact_dir / state_file)

    if model_name == "rule_based":
        parser = RuleBasedParser.from_export_state(state_payload)
        return parser, manifest

    if model_name == "crf":
        model_file = files.get("model")
        if not isinstance(model_file, str) or not model_file.strip():
            raise ValueError(f"Missing CRF model file entry in {manifest_path}.")

        model_path = artifact_dir / model_file
        with model_path.open("rb") as handle:
            model_obj = pickle.load(handle)

        parser = CRFParser.from_export_state(state_payload, model=model_obj)
        return parser, manifest

    raise ValueError(f"Unsupported exported parser type: {model_name}")


def load_latest_exported_parser(results_dir: Path) -> Tuple[RuleBasedParser | CRFParser, dict]:
    exports_root = results_dir / EXPORTS_SUBDIR
    latest_path = exports_root / LATEST_POINTER_FILE

    if not latest_path.exists():
        raise FileNotFoundError(f"No latest export pointer found at {latest_path}.")

    latest = _read_json(latest_path)
    manifest_rel = latest.get("manifest_path")
    if not isinstance(manifest_rel, str) or not manifest_rel.strip():
        raise ValueError(f"Invalid manifest_path in {latest_path}.")

    manifest_path = exports_root / manifest_rel
    if not manifest_path.exists():
        raise FileNotFoundError(f"Export manifest not found: {manifest_path}")

    return load_parser_from_manifest(manifest_path)
