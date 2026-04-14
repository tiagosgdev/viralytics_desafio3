import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_IMAGE_URL = (
    "https://myisepipp-my.sharepoint.com/:i:/r/personal/"
    "1220823_isep_ipp_pt/Documents/Clothing_Items/black/"
    "short_sleeve_top.png?csf=1&web=1&e=biqol6"
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "clothing.db"
DEFAULT_DATA_SOURCES_PATH = BASE_DIR / "DataSources"


def _normalize_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _humanize_item_type(item_type: Any) -> str:
    raw_type = _normalize_text(item_type, "clothing item")
    return raw_type.replace("_", " ")


def build_short_description(item: Dict[str, Any]) -> str:
    item_type = _humanize_item_type(item.get("type"))
    color = _normalize_text(item.get("color"), "versatile")
    style = _normalize_text(item.get("style"), "casual")
    material = _normalize_text(item.get("material"), "mixed-fabric")
    fit = _normalize_text(item.get("fit"), "regular")
    pattern = _normalize_text(item.get("pattern"), "plain")
    occasion = _normalize_text(item.get("occasion"), "everyday wear")

    return (
        f"{color.title()} {item_type} in {style} style, made from {material}, "
        f"with a {fit} fit and {pattern} pattern, ideal for {occasion}."
    )


def list_json_files(folder: Path) -> List[Path]:
    return sorted(path for path in folder.glob("*.json") if path.is_file())


def update_json_file(json_path: Path, image_url: str) -> int:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in JSON file: {json_path}")

    updated_rows = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        item["short_description"] = build_short_description(item)
        item["image_url"] = image_url
        updated_rows += 1

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return updated_rows


def _ensure_db_columns(
    cursor: sqlite3.Cursor,
    table_name: str,
    required_columns: Sequence[Tuple[str, str]],
) -> List[str]:
    cursor.execute(f'PRAGMA table_info("{table_name}")')
    existing_columns = {row[1] for row in cursor.fetchall()}

    added_columns: List[str] = []
    for column_name, column_type in required_columns:
        if column_name in existing_columns:
            continue
        cursor.execute(
            f'ALTER TABLE "{table_name}" ADD COLUMN "{column_name}" {column_type}'
        )
        added_columns.append(column_name)

    return added_columns


def update_database(db_path: Path, image_url: str) -> Tuple[int, List[str]]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
    )
    if cursor.fetchone() is None:
        connection.close()
        raise RuntimeError("Table 'items' was not found in the SQLite database.")

    added_columns = _ensure_db_columns(
        cursor,
        "items",
        (
            ("short_description", "TEXT"),
            ("image_url", "TEXT"),
        ),
    )

    cursor.execute(
        """
        SELECT id, type, color, style, pattern, material, fit, occasion
        FROM items
        """
    )
    rows = cursor.fetchall()

    payload: List[Tuple[str, str, int]] = []
    for row in rows:
        row_dict = dict(row)
        payload.append(
            (
                build_short_description(row_dict),
                image_url,
                int(row_dict["id"]),
            )
        )

    cursor.executemany(
        """
        UPDATE items
        SET short_description = ?, image_url = ?
        WHERE id = ?
        """,
        payload,
    )

    connection.commit()
    connection.close()
    return len(payload), added_columns


def _resolve_json_targets(args: argparse.Namespace) -> Iterable[Path]:
    if args.json_file:
        files: List[Path] = []
        for raw_path in args.json_file:
            path = Path(raw_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"JSON file not found: {path}")
            files.append(path)
        return files

    data_sources_path = Path(args.data_sources_path).expanduser()
    if not data_sources_path.is_dir():
        raise FileNotFoundError(f"DataSources folder not found: {data_sources_path}")
    return list_json_files(data_sources_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add or refresh short_description and image_url fields "
            "in JSON files and in the SQLite items table."
        )
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to clothing.db (default: local clothing.db).",
    )
    parser.add_argument(
        "--data-sources-path",
        default=str(DEFAULT_DATA_SOURCES_PATH),
        help="Folder with JSON files (default: local DataSources folder).",
    )
    parser.add_argument(
        "--json-file",
        action="append",
        help=(
            "Specific JSON file to update. "
            "Can be provided multiple times. If omitted, all JSON files are updated."
        ),
    )
    parser.add_argument(
        "--image-url",
        default=DEFAULT_IMAGE_URL,
        help="Image URL to assign to every item.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path).expanduser()
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    json_files = list(_resolve_json_targets(args))

    total_json_rows = 0
    json_files_updated = 0
    for json_file in json_files:
        updated_rows = update_json_file(json_file, args.image_url)
        total_json_rows += updated_rows
        json_files_updated += 1
        print(f"Updated {updated_rows} items in JSON file: {json_file.name}")

    updated_db_rows, added_columns = update_database(db_path, args.image_url)

    if not json_files:
        print("No JSON files were found to update.")
    print(
        f"JSON update complete: {json_files_updated} file(s), "
        f"{total_json_rows} item(s) updated."
    )
    print(
        f"Database update complete: {updated_db_rows} row(s) updated "
        f"in table 'items'."
    )
    if added_columns:
        print(f"Added database column(s): {', '.join(added_columns)}")


if __name__ == "__main__":
    main()