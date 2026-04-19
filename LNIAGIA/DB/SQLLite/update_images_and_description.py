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
DEFAULT_CSV_PATH = BASE_DIR / "DataSources" / "clothing_full (1).csv"
FALLBACK_CSV_PATHS: tuple[Path, ...] = (
    BASE_DIR / "Images" / "clothing_full.csv",
    BASE_DIR / "DataSources" / "clothing_full.csv",
)
DEFAULT_TABLE_NAME = "items"
DEFAULT_URL_COLUMN = "image_url"
DEFAULT_DESCRIPTION_COLUMN = "short_description"
LEGACY_DESCRIPTION_COLUMN = "short_decription"
DEFAULT_ID_COLUMN = "id"
DEFAULT_COLOR_COLUMN = "color"
DEFAULT_TYPE_COLUMN = "type"
DEFAULT_MISSING_IMAGE_URL = "https://drive.google.com/file/d/1F36ApP4XY9Ctn8TPNePc9b94HYc5vG_b/view?usp=drivesdk"

# Requested color remapping from DB colors to the CSV-covered colors.
COLOR_TO_CSV_COLOR_MAP: dict[str, str] = {
    "black": "black",
    "white": "white",
    "gray": "black",
    "navy": "blue",
    "blue": "blue",
    "red": "red",
    "green": "green",
    "yellow": "yellow",
    "orange": "yellow",
    "pink": "red",
    "purple": "blue",
    "brown": "black",
    "beige": "white",
    "cream": "white",
    "burgundy": "red",
    "olive": "green",
    "teal": "blue",
    "coral": "red",
    "multicolor": "green",
}

EMPTY_LIKE_VALUES = {"", "none", "null", "nan", "n/a", "na"}

_DRIVE_FILE_PATTERN = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str, *, label: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid {label}: {identifier}")
    return f'"{identifier}"'


def _normalize(value: object) -> str:
    return str(value or "").strip().lower()


def _clean_value(value: object) -> str:
    text = str(value or "").strip()
    if _normalize(text) in EMPTY_LIKE_VALUES:
        return ""
    return text


def _humanize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ")).strip()


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


def _resolve_csv_path(csv_path: Path) -> Path:
    if csv_path.is_file():
        return csv_path

    for fallback in FALLBACK_CSV_PATHS:
        if fallback.is_file():
            print(f"CSV file not found at {csv_path}; using fallback: {fallback}")
            return fallback

    raise FileNotFoundError(f"CSV file not found: {csv_path}")


def _map_color_to_csv(target_color: str) -> str:
    normalized = _normalize(target_color)
    return COLOR_TO_CSV_COLOR_MAP.get(normalized, normalized)


def _pick_image_url(
    *,
    csv_images: dict[tuple[str, str], list[str]],
    item_color: str,
    item_type: str,
    round_robin_index: dict[tuple[str, str], int],
) -> tuple[str | None, str]:
    normalized_type = _normalize(item_type)
    source_color = _map_color_to_csv(item_color)
    key = (source_color, normalized_type)
    links = csv_images.get(key)
    if not links:
        return None, source_color

    index = round_robin_index.get(key, 0)
    selected = links[index % len(links)]
    round_robin_index[key] = index + 1
    return selected, source_color


def _table_columns(cursor: sqlite3.Cursor, table_name: str) -> dict[str, str]:
    table_sql = _quote_identifier(table_name, label="table name")
    cursor.execute(f"PRAGMA table_info({table_sql})")
    columns = cursor.fetchall()
    return {_normalize(row[1]): str(row[1]) for row in columns}


def _ensure_items_columns(
    cursor: sqlite3.Cursor,
    *,
    table_name: str,
    url_column: str,
    description_column: str,
) -> tuple[bool, bool, bool]:
    table_sql = _quote_identifier(table_name, label="table name")
    url_sql = _quote_identifier(url_column, label="URL column")
    description_sql = _quote_identifier(description_column, label="description column")

    existing_columns = _table_columns(cursor, table_name)

    added_url = False
    added_description = False
    migrated_legacy_description = False

    if _normalize(url_column) not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_sql} ADD COLUMN {url_sql} TEXT")
        added_url = True
        existing_columns[_normalize(url_column)] = url_column

    if _normalize(description_column) not in existing_columns:
        cursor.execute(f"ALTER TABLE {table_sql} ADD COLUMN {description_sql} TEXT")
        added_description = True
        existing_columns[_normalize(description_column)] = description_column

    legacy_key = _normalize(LEGACY_DESCRIPTION_COLUMN)
    description_key = _normalize(description_column)
    if legacy_key in existing_columns and legacy_key != description_key:
        legacy_sql = _quote_identifier(existing_columns[legacy_key], label="legacy description column")
        cursor.execute(
            f"UPDATE {table_sql} "
            f"SET {description_sql} = COALESCE(NULLIF(TRIM({description_sql}), ''), {legacy_sql}) "
            f"WHERE {legacy_sql} IS NOT NULL AND TRIM({legacy_sql}) <> ''"
        )
        migrated_legacy_description = cursor.rowcount > 0

    return added_url, added_description, migrated_legacy_description


def _build_short_description(
    row: dict[str, object],
    *,
    id_column: str,
    color_column: str,
    type_column: str,
    url_column: str,
    description_column: str,
) -> str:
    item_type = _humanize_text(_clean_value(row.get(type_column)))
    color = _humanize_text(_clean_value(row.get(color_column)))

    title_bits = [bit for bit in (color, item_type) if bit]
    title = " ".join(title_bits).strip()
    if title:
        title = title[0].upper() + title[1:]
    else:
        title = "Clothing item"

    style = _humanize_text(_clean_value(row.get("style")))
    material = _humanize_text(_clean_value(row.get("material")))
    fit = _humanize_text(_clean_value(row.get("fit")))
    gender = _humanize_text(_clean_value(row.get("gender")))
    age_group = _humanize_text(_clean_value(row.get("age_group")))
    season = _humanize_text(_clean_value(row.get("season")))
    occasion = _humanize_text(_clean_value(row.get("occasion")))
    brand = _clean_value(row.get("brand"))
    price = _clean_value(row.get("price"))

    details: list[str] = []
    if material:
        details.append(f"in {material}")
    if fit:
        details.append(f"with a {fit} fit")
    if style:
        details.append(f"{style} style")
    if gender:
        details.append(f"for {gender}")
    if age_group:
        details.append(f"for {age_group}")
    if season:
        details.append(f"ideal for {season}")
    if occasion:
        details.append(f"suitable for {occasion}")
    if brand:
        details.append(f"by {brand}")
    if price:
        details.append(f"priced at {price}")

    excluded_fields = {
        _normalize(id_column),
        _normalize(color_column),
        _normalize(type_column),
        _normalize(url_column),
        _normalize(description_column),
        _normalize(LEGACY_DESCRIPTION_COLUMN),
        "style",
        "material",
        "fit",
        "gender",
        "age_group",
        "season",
        "occasion",
        "brand",
        "price",
    }

    extra_details: list[str] = []
    for field_name, raw_value in row.items():
        field_key = _normalize(field_name)
        if field_key in excluded_fields:
            continue

        value = _clean_value(raw_value)
        if not value:
            continue

        pretty_field = _humanize_text(field_name)
        pretty_value = _humanize_text(value)
        extra_details.append(f"{pretty_field}: {pretty_value}")

    if extra_details:
        details.append(f"details include {', '.join(extra_details[:4])}")

    description = title
    if details:
        description = f"{description} " + ", ".join(details)

    return re.sub(r"\s+", " ", description).strip().rstrip(".") + "."


def update_database(
    db_path: Path = DEFAULT_DB_PATH,
    csv_path: Path = DEFAULT_CSV_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    url_column: str = DEFAULT_URL_COLUMN,
    description_column: str = DEFAULT_DESCRIPTION_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    color_column: str = DEFAULT_COLOR_COLUMN,
    type_column: str = DEFAULT_TYPE_COLUMN,
    dry_run: bool = False,
) -> int:
    """Update image_url and short_description in SQLite."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    resolved_csv_path = _resolve_csv_path(csv_path)
    csv_images = load_csv_images(resolved_csv_path)
    if not csv_images:
        raise ValueError(f"CSV produced no usable mappings: {resolved_csv_path}")

    table_sql = _quote_identifier(table_name, label="table name")
    url_sql = _quote_identifier(url_column, label="URL column")
    description_sql = _quote_identifier(description_column, label="description column")
    id_sql = _quote_identifier(id_column, label="ID column")
    color_sql = _quote_identifier(color_column, label="color column")
    type_sql = _quote_identifier(type_column, label="type column")

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    try:
        print(f"Connected to SQLite database: {db_path}")
        print(f"Loaded {len(csv_images)} (color, type) mapping key(s) from: {resolved_csv_path}")

        existing_columns = _table_columns(cursor, table_name)
        missing_url = _normalize(url_column) not in existing_columns
        missing_description = _normalize(description_column) not in existing_columns
        migrated_legacy_description = False

        if dry_run:
            if missing_url:
                print(f"Dry run: would add missing column: {url_column}")
            if missing_description:
                print(f"Dry run: would add missing column: {description_column}")
        else:
            added_url, added_description, migrated_legacy_description = _ensure_items_columns(
                cursor,
                table_name=table_name,
                url_column=url_column,
                description_column=description_column,
            )

            if added_url:
                print(f"Added missing column: {url_column}")
            if added_description:
                print(f"Added missing column: {description_column}")
        if migrated_legacy_description:
            print(
                "Migrated values from legacy column "
                f"{LEGACY_DESCRIPTION_COLUMN} to {description_column}"
            )

        select_query = (
            f"SELECT * "
            f"FROM {table_sql} "
            f"ORDER BY {id_sql}"
        )
        cursor.execute(select_query)
        rows = cursor.fetchall()
        column_names = [column[0] for column in cursor.description]
        print(f"Scanned {len(rows)} row(s)")

        update_query = (
            f"UPDATE {table_sql} "
            f"SET {url_sql} = ?, {description_sql} = ? "
            f"WHERE {id_sql} = ?"
        )

        round_robin_index: dict[tuple[str, str], int] = {}

        updated_count = 0
        url_updated_count = 0
        description_updated_count = 0
        unchanged_count = 0
        unresolved_count = 0
        default_url_used_count = 0
        default_url_updated_count = 0
        fallback_usage: dict[tuple[str, str], int] = defaultdict(int)
        unresolved_pairs: dict[tuple[str, str], int] = defaultdict(int)

        for row in rows:
            row_data = dict(zip(column_names, row))

            row_id = row_data.get(id_column)
            color = row_data.get(color_column)
            item_type = row_data.get(type_column)
            current_url = _clean_value(row_data.get(url_column))
            current_description = _clean_value(row_data.get(description_column))

            if row_id is None:
                unresolved_count += 1
                continue

            new_description = _build_short_description(
                row_data,
                id_column=id_column,
                color_column=color_column,
                type_column=type_column,
                url_column=url_column,
                description_column=description_column,
            )
            description_changed = new_description != current_description

            new_url = current_url
            url_changed = False

            if isinstance(color, str) and isinstance(item_type, str):
                picked_url, source_color = _pick_image_url(
                    csv_images=csv_images,
                    item_color=color,
                    item_type=item_type,
                    round_robin_index=round_robin_index,
                )

                fallback_usage[(_normalize(color), source_color)] += 1

                if picked_url:
                    new_url = picked_url
                    url_changed = new_url != current_url
                else:
                    new_url = DEFAULT_MISSING_IMAGE_URL
                    url_changed = new_url != current_url
                    unresolved_count += 1
                    default_url_used_count += 1
                    if url_changed:
                        default_url_updated_count += 1
                    unresolved_pairs[(_normalize(color), _normalize(item_type))] += 1
            else:
                new_url = DEFAULT_MISSING_IMAGE_URL
                url_changed = new_url != current_url
                unresolved_count += 1
                default_url_used_count += 1
                if url_changed:
                    default_url_updated_count += 1
                unresolved_pairs[(_normalize(color), _normalize(item_type))] += 1

            if not url_changed and not description_changed:
                unchanged_count += 1
                continue

            updated_count += 1
            if url_changed:
                url_updated_count += 1
            if description_changed:
                description_updated_count += 1

            if not dry_run:
                cursor.execute(update_query, (new_url, new_description, row_id))

        if dry_run:
            connection.rollback()
            print(f"Dry run complete. {updated_count} row(s) would be updated.")
        else:
            connection.commit()
            print(f"Updated {updated_count} row(s).")

        print(f"Rows with image_url changes: {url_updated_count}")
        print(f"Rows with short_description changes: {description_updated_count}")
        print(f"Unchanged rows: {unchanged_count}")
        print(f"Rows without image match: {unresolved_count}")
        print(f"Rows using default image URL: {default_url_used_count}")
        print(f"Rows updated to default image URL: {default_url_updated_count}")

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
            "Update image_url and short_description in SQLite by matching "
            "(color, type) with a DB-color to CSV-color mapping."
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
        help="Path to CSV with columns cor, tipo_de_peca and link",
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
        "--description-column",
        default=DEFAULT_DESCRIPTION_COLUMN,
        help="Column that stores short descriptions",
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
        description_column=args.description_column,
        id_column=args.id_column,
        color_column=args.color_column,
        type_column=args.type_column,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
