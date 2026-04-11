# How to Run

This guide explains exactly what to install, what to do first, and how to run the clothing search app.

## 1. Prerequisites

- Python 3.10+ (recommended: 3.11)
- pip (comes with Python)
- Ollama installed: https://ollama.com/download


## 2. Install Python dependencies

```CMD
pip install -r LNIAGIA\requirements.txt
```

## 3. Install the Ollama model used by the parser

The parser is configured to use `qwen2.5:7b-instruct-q3_K_M`.

```CMD
ollama pull qwen2.5:7b-instruct-q3_K_M
ollama list
```

Make sure Ollama is running before you start the app.

## 4. First-time data setup (run once)

The search app needs a generated catalog, natural-language descriptions, and a built vector database.

### 4.1 Generate catalog data

```powershell
python -m LNIAGIA.DB.SQLLite.DataGenerator
```

This creates a JSON file in `LNIAGIA/DB/SQLLite/DataSources`.

### 4.2 Generate natural-language descriptions

```powershell
python -m LNIAGIA.DB.vector.description_generator
```

Choose the JSON file generated in the previous step when prompted.

### 4.3 Build the vector database

```powershell
python -m LNIAGIA.DB.vector.VectorDBManager
```

Inside the menu:

1. Choose option `1` (Create/Recreate Vector DB)
2. Select the generated description JSON file
3. Wait until indexing is complete

## 5. Run the interactive search app

```powershell
python -m LNIAGIA.search_app
```

In the app:

- Type a natural-language query (example: `I want a black casual t-shirt, not floral`)
- Choose strict mode (`y`) or soft mode (`n`)
- Use commands:
  - `new: <query>` start a new query context
  - `show` show current query and filters
  - `reset` clear context
  - `exit` quit

## 6. Optional: run evaluation (to improve)

```powershell
python -m LNIAGIA.tests.run_evaluation
```

Results are written in `LNIAGIA/tests/output`.

## 7. Quick troubleshooting

- `Vector DB not found`:
  - Run `python -m LNIAGIA.DB.vector.VectorDBManager` and create/recreate the DB.
- `model ... not found` from Ollama:
  - Run `ollama pull qwen2.5:7b-instruct-q3_K_M`.
- `No module named ...`:
  - Confirm virtual environment is active and reinstall requirements.
