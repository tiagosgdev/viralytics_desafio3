from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "clothing.db"
DEFAULT_CSV_PATH = BASE_DIR / "Images" / "clothing_full.csv"
DEFAULT_TABLE_NAME = "items"
DEFAULT_URL_COLUMN = "image_url"
DEFAULT_ID_COLUMN = "id"
DEFAULT_COLOR_COLUMN = "color"
DEFAULT_TYPE_COLUMN = "type"

# Color fallback preference map used when the CSV has no image for a target color.
#
# Edit this map to control substitutions. The script will always try the exact
# DB color first, then fallback colors in the order listed here.
COLOR_FALLBACK_MAP: dict[str, tuple[str, ...]] = {
    "black": ("black", "navy", "brown"),
    "white": ("white", "cream", "beige"),
    "gray": ("black", "white", "navy", "brown"),
    "navy": ("blue", "black", "teal"),
    "blue": ("blue", "navy", "teal"),
    "red": ("red", "burgundy", "coral", "pink"),
    "green": ("green", "olive", "teal", "blue"),
    "yellow": ("yellow", "cream", "beige", "orange"),
    "orange": ("yellow", "red", "coral"),
    "pink": ("red", "coral", "purple"),
    "purple": ("burgundy", "blue", "pink"),
    "brown": ("black", "beige", "olive"),
    "beige": ("white", "cream", "brown", "yellow"),
    "cream": ("white", "beige", "yellow"),
    "burgundy": ("red", "purple", "brown"),
    "olive": ("green", "brown", "black"),
    "teal": ("blue", "green", "navy"),
    "coral": ("red", "orange", "pink"),
    "multicolor": ("red", "white", "blue", "black"),
}

# Last-resort colors checked for every target color when close colors have no
# image for a specific type in the CSV.
GLOBAL_FALLBACK_COLORS: tuple[str, ...] = (
    "black",
    "white",
    "blue",
    "red",
    "green",
    "yellow",
)

_DRIVE_FILE_PATTERN = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str, *, label: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid {label}: {identifier}")
    return f'"{identifier}"'


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def extract_file_id(drive_url: str) -> str | None:
    """Extract the file ID from a Google Drive URL."""
    clean_url = str(drive_url or "").strip()

    match = _DRIVE_FILE_PATTERN.search(clean_url)
    if match:
        return match.group(1)

    parsed = urlparse(clean_url)
    query_id = parse_qs(parsed.query).get("id")
    if query_id and query_id[0].strip():
        return query_id[0].strip()

    return None


def transform_drive_url(drive_url: str) -> str:
    """Transform a Google Drive viewer URL to a direct image URL."""
    file_id = extract_file_id(drive_url)
    if file_id:
        return f"https://drive.google.com/uc?export=view&id={file_id}"
    return str(drive_url or "").strip()


def _find_csv_key(fieldnames: list[str], expected_name: str) -> str | None:
    expected = _normalize(expected_name)
    for name in fieldnames:
        if _normalize(name) == expected:
            return name
    return None


def load_csv_images(csv_path: Path) -> dict[tuple[str, str], list[str]]:
    """Load CSV rows into mapping: (color, type) -> list[direct_image_urls]."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    mapping: dict[tuple[str, str], list[str]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        color_key = _find_csv_key(reader.fieldnames, "cor")
        type_key = _find_csv_key(reader.fieldnames, "tipo_de_peca")
        link_key = _find_csv_key(reader.fieldnames, "link")

        if not color_key or not type_key or not link_key:
            raise ValueError(
                "CSV must contain columns: cor, tipo_de_peca, link"
            )

        for row in reader:
            color = _normalize(row.get(color_key))
            item_type = _normalize(row.get(type_key))
            raw_url = str(row.get(link_key) or "").strip()
            if not color or not item_type or not raw_url:
                continue

            direct_url = transform_drive_url(raw_url)
            if not direct_url:
                continue

            key = (color, item_type)
            if direct_url not in mapping[key]:
                mapping[key].append(direct_url)

    return dict(mapping)


def _color_candidates(target_color: str) -> list[str]:
    normalized = _normalize(target_color)
    configured = list(COLOR_FALLBACK_MAP.get(normalized, ()))
    return _unique_preserve_order([normalized, *configured, *GLOBAL_FALLBACK_COLORS])


def _pick_image_url(
    *,
    csv_images: dict[tuple[str, str], list[str]],
    item_color: str,
    item_type: str,
    round_robin_index: dict[tuple[str, str], int],
) -> tuple[str | None, str | None]:
    normalized_type = _normalize(item_type)

    for candidate_color in _color_candidates(item_color):
        key = (candidate_color, normalized_type)
        links = csv_images.get(key)
        if not links:
            continue

        index = round_robin_index.get(key, 0)
        selected = links[index % len(links)]
        round_robin_index[key] = index + 1
        return selected, candidate_color

    return None, None


def update_database(
    db_path: Path = DEFAULT_DB_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    url_column: str = DEFAULT_URL_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    color_column: str = DEFAULT_COLOR_COLUMN,
    type_column: str = DEFAULT_TYPE_COLUMN,
    dry_run: bool = False,
) -> int:
    """Update image_url in SQLite using CSV mappings by (color, type)."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    csv_images = load_csv_images(csv_path)
    if not csv_images:
        raise ValueError(f"CSV produced no usable mappings: {csv_path}")

    table_sql = _quote_identifier(table_name, label="table name")
    url_sql = _quote_identifier(url_column, label="URL column")
    id_sql = _quote_identifier(id_column, label="ID column")
    color_sql = _quote_identifier(color_column, label="color column")
    type_sql = _quote_identifier(type_column, label="type column")

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    try:
        print(f"Connected to SQLite database: {db_path}")
        print(f"Loaded {len(csv_images)} (color, type) mapping key(s) from: {csv_path}")

        select_query = (
            f"SELECT {id_sql}, {color_sql}, {type_sql}, {url_sql} "
            f"FROM {table_sql} "
            f"ORDER BY {id_sql}"
        )
        cursor.execute(select_query)
        rows = cursor.fetchall()
        print(f"Scanned {len(rows)} row(s)")

        update_query = (
            f"UPDATE {table_sql} "
            f"SET {url_sql} = ? "
            f"WHERE {id_sql} = ?"
        )

        round_robin_index: dict[tuple[str, str], int] = {}

        updated_count = 0
        unchanged_count = 0
        unresolved_count = 0
        fallback_usage: dict[tuple[str, str], int] = defaultdict(int)
        unresolved_pairs: dict[tuple[str, str], int] = defaultdict(int)

        for row_id, color, item_type, current_url in rows:
            if not isinstance(color, str) or not isinstance(item_type, str):
                unresolved_count += 1
                continue

            new_url, source_color = _pick_image_url(
                csv_images=csv_images,
                item_color=color,
                item_type=item_type,
                round_robin_index=round_robin_index,
            )

            if not new_url:
                unresolved_count += 1
                unresolved_pairs[(_normalize(color), _normalize(item_type))] += 1
                continue

            fallback_usage[(_normalize(color), source_color)] += 1

            if new_url == str(current_url or "").strip():
                unchanged_count += 1
                continue

            updated_count += 1
            if not dry_run:
                cursor.execute(update_query, (new_url, row_id))

        if dry_run:
            connection.rollback()
            print(f"Dry run complete. {updated_count} row(s) would be updated.")
        else:
            connection.commit()
            print(f"Updated {updated_count} row(s).")

        print(f"Unchanged rows: {unchanged_count}")
        print(f"Rows without image match: {unresolved_count}")

        fallback_only = [
            (target, source, count)
            for (target, source), count in fallback_usage.items()
            if target != source
        ]
        if fallback_only:
            fallback_only.sort(key=lambda item: item[2], reverse=True)
            print("Top fallback color usages (target_color -> source_color):")
            for target, source, count in fallback_only[:10]:
                print(f"  - {target} -> {source}: {count} row(s)")

        if unresolved_pairs:
            missing_sorted = sorted(unresolved_pairs.items(), key=lambda item: item[1], reverse=True)
            print("Top unresolved (color, type) combinations:")
            for (color, item_type), count in missing_sorted[:10]:
                print(f"  - ({color}, {item_type}): {count} row(s)")

        return updated_count
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update image_url in SQLite using clothing_full.csv by matching "
            "(color, type) with fallback color mapping."
        )
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database (default: local clothing.db)",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help="Path to clothing_full CSV (default: local Images/clothing_full.csv)",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_NAME,
        help="Table name containing image URLs",
    )
    parser.add_argument(
        "--url-column",
        default=DEFAULT_URL_COLUMN,
        help="Column that stores the image URL",
    )
    parser.add_argument(
        "--id-column",
        default=DEFAULT_ID_COLUMN,
        help="Primary key column",
    )
    parser.add_argument(
        "--color-column",
        default=DEFAULT_COLOR_COLUMN,
        help="Column that stores color values",
    )
    parser.add_argument(
        "--type-column",
        default=DEFAULT_TYPE_COLUMN,
        help="Column that stores type values",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would change without writing to the database",
    )

    args = parser.parse_args()

    update_database(
        db_path=Path(args.db_path),
        csv_path=Path(args.csv_path),
        table_name=args.table,
        url_column=args.url_column,
        id_column=args.id_column,
        color_column=args.color_column,
        type_column=args.type_column,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
