import sqlite3
import json
import os
import glob
import sys
import random
import hashlib
from pathlib import Path

try:
    from LNIAGIA.DB.models import GLOBAL_FIELDS, TYPE_FIELDS, EXTRA_FIELD_VALUES
except ModuleNotFoundError:
    db_dir = Path(__file__).resolve().parent.parent
    if str(db_dir) not in sys.path:
        sys.path.insert(0, str(db_dir))

    from models import AGE_GROUP, AGE_GROUP_WEIGHTS, GENDER, GENDER_WEIGHTS, GLOBAL_FIELDS, TYPE_FIELDS, EXTRA_FIELD_VALUES

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clothing.db')
DATA_SOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DataSources')


# ------------------ UTIL ------------------

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


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ------------------ USERS ------------------

def create_users_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE,
            password TEXT NOT NULL,

            age_group TEXT,
            gender TEXT,

            favorite_colors TEXT,
            favorite_styles TEXT,
            favorite_materials TEXT,
            preferred_seasons TEXT,
            preferred_occasions TEXT
        )
    """)


def generate_random_profile(items):
    def pick(field, k=3):
        values = list(set(item[field] for item in items if item.get(field)))
        return random.sample(values, min(k, len(values))) if values else []

    return {
        "age_group": weighted_choice(AGE_GROUP, AGE_GROUP_WEIGHTS),
        "gender": weighted_choice(GENDER, GENDER_WEIGHTS),
        "favorite_colors": ", ".join(pick("color")),
        "favorite_styles": ", ".join(pick("style")),
        "favorite_materials": ", ".join(pick("material")),
        "preferred_seasons": ", ".join(pick("season")),
        "preferred_occasions": ", ".join(pick("occasion")),
    }


FIRST_NAMES = (
    "Liam", "Noah", "Oliver", "James", "Elijah", "Mateo", "Ethan", "Lucas",
    "Sophia", "Emma", "Ava", "Isabella", "Mia", "Amelia", "Harper", "Evelyn",
    "Daniel", "Michael", "Sebastian", "Benjamin", "William", "Alexander",
    "Charlotte", "Grace", "Chloe", "Lily", "Hannah", "Zoe", "Sofia"
)

LAST_NAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez"
)

EMAIL_PROVIDERS = ("gmail.com", "outlook.com", "yahoo.com", "hotmail.com")

def create_random_users(cursor, items, n=10):
    for _ in range(n):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)

        # different username styles (adds realism + distribution)
        style = random.randint(1, 4)

        if style == 1:
            username = f"{first.lower()}{last.lower()}"
        elif style == 2:
            username = f"{first.lower()}.{last.lower()}"
        elif style == 3:
            username = f"{first.lower()}_{last.lower()}{random.randint(1, 99)}"
        else:
            username = f"{first.lower()}{random.randint(10, 999)}"

        email = f"{username}@{random.choice(EMAIL_PROVIDERS)}"
        password = hash_password("password123")

        profile = generate_random_profile(items)

        try:
            cursor.execute("""
                INSERT INTO users (
                    username, email, password,
                    age_group, gender,
                    favorite_colors, favorite_styles,
                    favorite_materials, preferred_seasons,
                    preferred_occasions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                username,
                email,
                password,
                profile["age_group"],
                profile["gender"],
                profile["favorite_colors"],
                profile["favorite_styles"],
                profile["favorite_materials"],
                profile["preferred_seasons"],
                profile["preferred_occasions"]
            ))
        except sqlite3.IntegrityError:
            continue

def weighted_choice(options, weights_dict):
    weights = [weights_dict.get(opt, 0.01) for opt in options]
    return random.choices(options, weights=weights, k=1)[0]

# ------------------ ITEMS ------------------

def get_items_by_ids(item_ids):
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

    rows_by_id = {int(row["id"]): row for row in rows if row.get("id")}

    return [rows_by_id[i] for i in normalized_ids if i in rows_by_id]


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
        cursor.execute("DROP TABLE IF EXISTS users")

    cols_def = ', '.join(f'"{col}" TEXT' for col in columns)
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {cols_def}
        )
    ''')

    # ✅ NEW: create users table
    create_users_table(cursor)

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

    # ✅ NEW: create random users
    create_random_users(cursor, items, n=10)

    conn.commit()
    conn.close()

    print(f"\n  Success! {len(items)} items + users loaded.")


def manual_query():
    print("\n" + "="*50)
    print("  Manual Query")
    print("="*50)

    query = input("  >> ").strip()

    if query.lower() == 'back':
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)

        if query.strip().upper().startswith('SELECT'):
            rows = cursor.fetchall()
            headers = [desc[0] for desc in cursor.description]

            print("\n  " + " | ".join(headers))
            print("  " + "-" * 50)

            for row in rows:
                print("  " + " | ".join(str(v) for v in row))

            print(f"\n  {len(rows)} row(s)")
        else:
            conn.commit()
            print("Query executed.")

        conn.close()

    except sqlite3.Error as e:
        print(f"Error: {e}")


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
                break
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
                break


if __name__ == '__main__':
    main()