from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "clothing.db"
DEFAULT_TABLE_NAME = "items"
DEFAULT_URL_COLUMN = "image_url"
DEFAULT_ID_COLUMN = "id"

_DRIVE_FILE_PATTERN = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str, *, label: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(identifier):
        raise ValueError(f"Invalid {label}: {identifier}")
    return f'"{identifier}"'


def extract_file_id(drive_url: str) -> str | None:
    """Extract the file ID from a Google Drive URL."""
    match = _DRIVE_FILE_PATTERN.search(drive_url or "")
    if match:
        return match.group(1)
    return None


def transform_drive_url(drive_url: str) -> str:
    """Transform a Google Drive viewer URL to a direct image URL."""
    file_id = extract_file_id(drive_url)
    if file_id:
        return f"https://drive.google.com/uc?export=view&id={file_id}"
    return drive_url


def update_database(
    db_path: Path = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    url_column: str = DEFAULT_URL_COLUMN,
    id_column: str = DEFAULT_ID_COLUMN,
    dry_run: bool = False,
) -> int:
    """Update Drive viewer URLs in SQLite and return the number of changed rows."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    table_sql = _quote_identifier(table_name, label="table name")
    url_sql = _quote_identifier(url_column, label="URL column")
    id_sql = _quote_identifier(id_column, label="ID column")

    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    try:
        print(f"Connected to SQLite database: {db_path}")

        select_query = (
            f"SELECT {id_sql}, {url_sql} "
            f"FROM {table_sql} "
            f"WHERE {url_sql} LIKE '%drive.google.com/file/d/%'"
        )
        cursor.execute(select_query)
        rows = cursor.fetchall()
        print(f"Found {len(rows)} rows with Google Drive viewer URLs")

        update_query = (
            f"UPDATE {table_sql} "
            f"SET {url_sql} = ? "
            f"WHERE {id_sql} = ?"
        )

        updated_count = 0
        for row_id, url in rows:
            if not isinstance(url, str):
                continue

            new_url = transform_drive_url(url)
            if new_url == url:
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

        return updated_count
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Google Drive viewer URLs in SQLite to direct image URLs."
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to SQLite database (default: local clothing.db)",
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
        "--dry-run",
        action="store_true",
        help="Show how many rows would change without writing to the database",
    )

    args = parser.parse_args()

    update_database(
        db_path=Path(args.db_path),
        table_name=args.table,
        url_column=args.url_column,
        id_column=args.id_column,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
