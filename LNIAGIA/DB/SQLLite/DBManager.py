import sqlite3
import json
import os
import glob
import sys
from pathlib import Path

try:
    from LNIAGIA.DB.models import GLOBAL_FIELDS, TYPE_FIELDS, EXTRA_FIELD_VALUES
except ModuleNotFoundError:
    db_dir = Path(__file__).resolve().parent.parent
    if str(db_dir) not in sys.path:
        sys.path.insert(0, str(db_dir))

    from models import GLOBAL_FIELDS, TYPE_FIELDS, EXTRA_FIELD_VALUES

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clothing.db')
DATA_SOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataSources')


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def db_exists():
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        return len(tables) > 0
    except sqlite3.Error:
        return False


def get_items_by_ids(item_ids):
    """Fetch items from SQLite by id while preserving the input order."""
    if not item_ids:
        return []

    normalized_ids = []
    for item_id in item_ids:
        try:
            normalized_ids.append(int(item_id))
        except (TypeError, ValueError):
            continue

    if not normalized_ids or not db_exists():
        return []

    placeholders = ",".join("?" for _ in normalized_ids)
    query = f"SELECT * FROM items WHERE id IN ({placeholders})"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(query, normalized_ids)
        rows = [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error:
        conn.close()
        return []

    conn.close()

    rows_by_id = {}
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            continue
        rows_by_id[int(row_id)] = row

    ordered_rows = []
    for item_id in normalized_ids:
        row = rows_by_id.get(item_id)
        if row is not None:
            ordered_rows.append(row)

    return ordered_rows


def list_json_files():
    return glob.glob(os.path.join(DATA_SOURCES_PATH, '*.json'))


def pick_json_file():
    files = list_json_files()
    if not files:
        print("\n  No JSON files found in DataSources folder.")
        return None

    print("\n  Available JSON files:")
    for i, f in enumerate(files, 1):
        print(f"    [{i}] {os.path.basename(f)}")

    while True:
        choice = input("\n  Select a file (number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        print("  Invalid choice. Please try again.")


def _all_columns():
    """Returns the full ordered list of columns: global fields + every possible type-specific field."""
    seen = set()
    columns = []
    for col in GLOBAL_FIELDS:
        if col not in seen:
            columns.append(col)
            seen.add(col)
    for fields in TYPE_FIELDS.values():
        for col in fields:
            if col not in seen:
                columns.append(col)
                seen.add(col)
    return columns


def populate_db(json_path, recreate=False):
    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    if not items:
        print("\n  The selected JSON file is empty.")
        return

    columns = _all_columns()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if recreate:
        cursor.execute("DROP TABLE IF EXISTS items")

    cols_def = ', '.join(f'"{col}" TEXT' for col in columns)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols_def}
        )
    ''')

    insert_columns = ["id"] + columns
    col_names = ', '.join(f'"{col}"' for col in insert_columns)
    placeholders = ', '.join('?' for _ in insert_columns)

    print(f"\n  Inserting {len(items)} items into the database...")

    for item in items:
        values = tuple(
            [item.get("id")] +
            [str(item[col]) if col in item else None for col in columns]
        )
        cursor.execute(
            f'INSERT INTO items ({col_names}) VALUES ({placeholders})',
            values
        )

    conn.commit()
    conn.close()

    print(f"\n  Success! {len(items)} items loaded from '{os.path.basename(json_path)}'.")


def manual_query():
    print("\n" + "="*50)
    print("  Manual Query")
    print("="*50)
    print("  Type your SQL query below, or type 'back' to return.\n")

    query = input("  >> ").strip()

    if query.lower() == 'back':
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)

        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            if cursor.description:
                headers = [desc[0] for desc in cursor.description]
                col_width = 18
                header_line = ' | '.join(h.ljust(col_width) for h in headers)
                print('\n  ' + header_line)
                print('  ' + '-' * len(header_line))
                for row in rows:
                    print('  ' + ' | '.join(str(v).ljust(col_width) for v in row))
                print(f"\n  {len(rows)} row(s) returned.")
            else:
                print("  Query executed with no output.")
        else:
            conn.commit()
            print(f"\n  Query executed successfully. {cursor.rowcount} row(s) affected.")

        conn.close()
    except sqlite3.Error as e:
        print(f"\n  Error: {e}")


def show_menu(exists):
    clear()
    print("="*50)
    print("         DB Manager — Viralytics")
    print("="*50)
    if not exists:
        print("  Database: NOT created yet\n")
        print("  [1] Create Database")
        print("  [0] Exit")
    else:
        print("  Database: Ready\n")
        print("  [1] Recreate / Regenerate Database")
        print("  [2] Manual Query")
        print("  [0] Exit")
    print("="*50)


def main():
    while True:
        exists = db_exists()
        show_menu(exists)
        choice = input("  Choose an option: ").strip()

        if not exists:
            if choice == '1':
                json_file = pick_json_file()
                if json_file:
                    populate_db(json_file, recreate=False)
                input("\n  Press Enter to return to the menu...")
            elif choice == '0':
                print("\n  Goodbye!\n")
                break
            else:
                print("\n  Invalid option. Please try again.")
                input("  Press Enter to continue...")
        else:
            if choice == '1':
                json_file = pick_json_file()
                if json_file:
                    populate_db(json_file, recreate=True)
                input("\n  Press Enter to return to the menu...")
            elif choice == '2':
                manual_query()
                input("\n  Press Enter to return to the menu...")
            elif choice == '0':
                print("\n  Goodbye!\n")
                break
            else:
                print("\n  Invalid option. Please try again.")
                input("  Press Enter to continue...")


if __name__ == '__main__':
    main()