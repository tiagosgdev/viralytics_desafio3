# ════════════════════════════════════════════════════════════
# llm_query_parser.py
# Uses a local Ollama LLM to parse a natural language clothing
# query into structured include/exclude filters + price range.
# ════════════════════════════════════════════════════════════

import json
import sys
from pathlib import Path

import ollama

# Allow importing nl_mappings from the same vector/ folder
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from DB.vector.nl_mappings import ALL_MAPPINGS
from DB.models import FILTERABLE_FIELDS, FREE_TEXT_FILTER_FIELDS

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

OLLAMA_MODEL = "qwen2.5:3b-instruct"

# ══════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    """Build the system prompt with all valid field values."""

    field_lines = []
    for field in FILTERABLE_FIELDS:
        keys = sorted(ALL_MAPPINGS[field].keys())
        field_lines.append(f'  "{field}": {json.dumps(keys)}')

    fields_block = ",\n".join(field_lines)

    free_text_block = "\n".join(
        f'  "{f}": <copy the exact name from the query, e.g. "Nike", "Gucci">'
        for f in FREE_TEXT_FILTER_FIELDS
    )

    return f"""\
You are a structured-data extractor for a clothing search engine.

Given a user's natural-language query, extract structured filters
that will be used to search a clothing database.

VALID FIELD VALUES (use ONLY these exact strings):
{{
{fields_block}
}}

FREE-TEXT FIELDS (copy the exact value the user mentions, any capitalisation):
{{
{free_text_block}
}}

Return ONLY a JSON object with this schema — no explanation, no markdown:
{{
  "include": {{
    "<field>": ["<value>", ...]
  }},
  "exclude": {{
    "<field>": ["<value>", ...]
  }},
  "price_range": {{
    "min": <number or null>,
    "max": <number or null>
  }}
}}

RULES:
- "include" lists the values the user WANTS.
- "exclude" lists the values the user does NOT want (negation).
- Only include fields that the user explicitly or implicitly mentions.
- If the user says "t-shirt", map it to "short_sleeve_top".
- If the user says "dress" generically, include ALL dress types:
  ["short_sleeve_dress","long_sleeve_dress","vest_dress","sling_dress"].
- If the user says "jacket" or "coat", include ALL outwear types:
  ["short_sleeve_outwear","long_sleeve_outwear"].
- PRICE rules:
  * Set price_range ONLY when the user explicitly mentions price, cost, or money with words
    like "cheap", "budget", "affordable", "expensive", "premium", "luxury".
  * If the user mentions NO price, omit price_range entirely (min: null, max: null).
  * Style words like "casual", "formal", "elegant", "fancy" refer to STYLE ONLY — they
    NEVER imply any price range. Do NOT set price_range based on style words.
  * If the user says "cheap" or "budget", set price_range (min: 0, max: 50).
  * If the user says "premium" or "luxury", set price_range (min: 150, max: null).
- COLOR rules:
  * If the user says "dark colors", expand to: ["black","navy","burgundy","olive","brown"].
  * If the user says "bright colors", "vivid colors", or "bold colors", expand to:
    ["white","yellow","orange","pink","red","purple","coral","teal"].
  * If the user says "neutral colors", expand to: ["white","gray","beige","cream","brown"].
- If the user mentions NO exclusions, set "exclude" to {{}}.
- Omit any section that is empty.
"""


def _build_user_prompt(query: str) -> str:
    return f'User query: "{query}"'


# ══════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════

def parse_query(query: str, model: str | None = None, verbose: bool = False) -> dict:
    """
    Send a natural-language query to Ollama and return structured filters.

    Args:
        query:   The user's free-text search query.
        model:   Ollama model name (defaults to OLLAMA_MODEL).
        verbose: Print the raw LLM response for debugging.

    Returns:
        Dict with keys "include", "exclude", "price_range" (any may be absent).
    """
    model = model or OLLAMA_MODEL

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": _build_user_prompt(query)},
        ],
        options={"temperature": 0},
    )

    raw = response.message.content.strip()

    if verbose:
        print(f"\n  [LLM raw response]\n  {raw}\n")

    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # drop first ```json line
        raw = raw.rsplit("```", 1)[0]  # drop trailing ```
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: LLM returned invalid JSON. Using empty filters.")
        parsed = {}

    # Validate values against known keys
    parsed = _validate(parsed)
    return parsed


def _validate(parsed: dict) -> dict:
    """Remove any field values that are not in the valid set."""
    for section in ("include", "exclude"):
        block = parsed.get(section, {})
        cleaned = {}
        for field, values in block.items():
            # Free-text fields (e.g. brand) — just ensure non-empty strings,
            # no closed-set validation needed.
            if field in FREE_TEXT_FILTER_FIELDS:
                filtered = [v for v in values if isinstance(v, str) and v.strip()]
                if filtered:
                    cleaned[field] = filtered
                continue
            if field not in ALL_MAPPINGS:
                continue
            valid = set(ALL_MAPPINGS[field].keys())
            filtered = [v for v in values if v in valid]
            if filtered:
                cleaned[field] = filtered
        if cleaned:
            parsed[section] = cleaned
        elif section in parsed:
            del parsed[section]

    # Validate price_range
    pr = parsed.get("price_range", {})
    if pr:
        pr_min = pr.get("min")
        pr_max = pr.get("max")
        if pr_min is not None and not isinstance(pr_min, (int, float)):
            pr_min = None
        if pr_max is not None and not isinstance(pr_max, (int, float)):
            pr_max = None
        if pr_min is not None or pr_max is not None:
            parsed["price_range"] = {"min": pr_min, "max": pr_max}
        else:
            parsed.pop("price_range", None)

    return parsed


# ══════════════════════════════════════════════════════════════
# CLI — quick manual test
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LLM Query Parser  (Ollama)")
    print("=" * 60)
    print(f"  Model: {OLLAMA_MODEL}")
    print("  Type a clothing query, or 'exit' to quit.\n")

    print(" System prompt:")
    print(_build_system_prompt())
    while True:
        q = input("  Query > ").strip()
        if not q or q.lower() == "exit":
            break

        result = parse_query(q, verbose=True)
        print("  Parsed filters:")
        print(f"  {json.dumps(result, indent=2)}\n")
