import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    from LNIAGIA.DB.models import COLOR, TYPE
except ModuleNotFoundError:
    db_dir = Path(__file__).resolve().parent.parent
    if str(db_dir) not in sys.path:
        sys.path.insert(0, str(db_dir))
    from models import COLOR, TYPE


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "clothing.db"
DATA_SOURCES_PATH = BASE_DIR / "DataSources"


def _print_options(title: str, options: Sequence[str]) -> None:
    print(f"\n{title}:")
    for idx, option in enumerate(options, 1):
        print(f"  [{idx}] {option}")


def _parse_indices(raw_input: str, max_index: int, allow_multiple: bool) -> List[int]:
    normalized = raw_input.replace(",", " ")
    tokens = normalized.split()

    if not tokens:
        raise ValueError("selection cannot be empty")

    if not allow_multiple and len(tokens) != 1:
        raise ValueError("select exactly one number")

    picked: List[int] = []
    seen = set()
    for token in tokens:
        if not token.isdigit():
            raise ValueError("use only numbers")

        index = int(token)
        if index < 1 or index > max_index:
            raise ValueError(f"numbers must be between 1 and {max_index}")

        if index not in seen:
            picked.append(index)
            seen.add(index)

    return picked


def choose_colors() -> List[str]:
    _print_options("Available colors", COLOR)

    while True:
        raw = input("\nSelect one or more colors (numbers separated by comma or space): ").strip()
        try:
            indices = _parse_indices(raw, len(COLOR), allow_multiple=True)
            return [COLOR[i - 1] for i in indices]
        except ValueError as exc:
            print(f"  Invalid input: {exc}.")


def choose_type() -> str:
    _print_options("Available types", TYPE)

    while True:
        raw = input("\nSelect exactly one type (one number): ").strip()
        try:
            index = _parse_indices(raw, len(TYPE), allow_multiple=False)[0]
            return TYPE[index - 1]
        except ValueError as exc:
            print(f"  Invalid input: {exc}.")


def ask_url() -> str:
    while True:
        url = input("\nType the image URL: ").strip()
        if url:
            return url
        print("  URL cannot be empty.")


def _load_json_items(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("root JSON value is not a list")

    return data


def update_json_files(selected_colors: Sequence[str], selected_type: str, image_url: str) -> Tuple[int, int]:
    json_files = sorted(path for path in DATA_SOURCES_PATH.glob("*.json") if path.is_file())

    if not json_files:
        print("  No JSON files found in DataSources.")
        return 0, 0

    total_rows_updated = 0
    files_updated = 0

    for json_file in json_files:
        try:
            items = _load_json_items(json_file)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"  Skipping {json_file.name}: {exc}")
            continue

        file_rows_updated = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("color") in selected_colors and item.get("type") == selected_type:
                item["image_url"] = image_url
                file_rows_updated += 1

        if file_rows_updated > 0:
            try:
                with json_file.open("w", encoding="utf-8") as file:
                    json.dump(items, file, indent=2, ensure_ascii=False)
                files_updated += 1
                total_rows_updated += file_rows_updated
            except OSError as exc:
                print(f"  Could not write {json_file.name}: {exc}")
                continue

        print(f"  {json_file.name}: {file_rows_updated} row(s) updated")

    return total_rows_updated, files_updated


def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _ensure_image_url_column(cursor: sqlite3.Cursor) -> bool:
    cursor.execute('PRAGMA table_info("items")')
    existing_columns = {row[1] for row in cursor.fetchall()}

    if "image_url" in existing_columns:
        return False

    cursor.execute('ALTER TABLE "items" ADD COLUMN "image_url" TEXT')
    return True


def update_database(selected_colors: Sequence[str], selected_type: str, image_url: str) -> Tuple[int, bool]:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"database file not found: {DB_PATH}")

    placeholders = ",".join("?" for _ in selected_colors)

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    try:
        if not _table_exists(cursor, "items"):
            raise RuntimeError("table 'items' was not found in the database")

        added_column = _ensure_image_url_column(cursor)

        count_query = (
            f'SELECT COUNT(*) FROM "items" '
            f'WHERE "type" = ? AND "color" IN ({placeholders})'
        )
        count_params = [selected_type, *selected_colors]
        cursor.execute(count_query, count_params)
        rows_to_update = int(cursor.fetchone()[0])

        update_query = (
            f'UPDATE "items" SET "image_url" = ? '
            f'WHERE "type" = ? AND "color" IN ({placeholders})'
        )
        update_params = [image_url, selected_type, *selected_colors]
        cursor.execute(update_query, update_params)

        connection.commit()
        return rows_to_update, added_column
    finally:
        connection.close()


def main() -> None:
    print("=" * 60)
    print("               Update Image URL By Color And Type")
    print("=" * 60)

    selected_colors = choose_colors()
    selected_type = choose_type()
    image_url = ask_url()

    print("\nSelected filters:")
    print(f"  Colors: {', '.join(selected_colors)}")
    print(f"  Type:   {selected_type}")
    print(f"  URL:    {image_url}")

    print("\nUpdating JSON files...")
    json_rows_updated, json_files_updated = update_json_files(
        selected_colors,
        selected_type,
        image_url,
    )

    print("\nUpdating SQLite database...")
    try:
        db_rows_updated, added_column = update_database(
            selected_colors,
            selected_type,
            image_url,
        )
        if added_column:
            print("  Added missing column: image_url")
        print(f"  Database rows updated: {db_rows_updated}")
    except (FileNotFoundError, RuntimeError, sqlite3.Error) as exc:
        db_rows_updated = 0
        print(f"  Database update failed: {exc}")

    print("\nSummary:")
    print(f"  JSON files updated: {json_files_updated}")
    print(f"  JSON rows updated:  {json_rows_updated}")
    print(f"  DB rows updated:    {db_rows_updated}")


if __name__ == "__main__":
    main()