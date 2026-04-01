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

OLLAMA_MODEL = "qwen2.5:7b-instruct-q3_K_M"
#OLLAMA_MODEL = "qwen2.5:3b-instruct"

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
  }}
}}

RULES:

1. INCLUDE vs EXCLUDE:
   - "include" lists the values the user WANTS (explicit or reasonably inferred).
   - "exclude" lists the values the user does NOT want (negations).

2. NEGATION HANDLING:
   When the user expresses negation using words like "not", "no", "don't", 
   "avoid", "without", "hate", or "don't want":
   
   - Put the negated value in "exclude" for that field.
   - DO NOT add alternative or opposite values to "include" for that same field.
   - Leave the field out of "include" entirely unless the user also mentions 
     a positive preference for that field.
   
   Examples:
   - "not fitted" -> exclude: {{"fit": ["fitted"]}}
     Do NOT add "relaxed", "loose", or other fits to include.
   
   - "no floral pattern" -> exclude: {{"pattern": ["floral"]}}
     Do NOT add "plain" or other patterns to include.
   
   - "not too short" -> exclude: {{"length": ["short", "mini"]}}
     Do NOT add "long", "midi", or other lengths to include.
   
   - "I want a casual black t-shirt, not fitted" ->
     include: {{"style": ["casual"], "color": ["black"], "type": ["short_sleeve_top"]}}
     exclude: {{"fit": ["fitted"]}}

3. CONTEXTUAL INFERENCE (allowed for "include" only):
   You may infer reasonable values for "include" based on context when the user
   describes a situation or occasion without being explicit:
   
   - "meeting with a CEO" -> infer formal style, professional occasion
   - "beach vacation" -> infer summer season, casual or sporty style
   - "wedding guest" -> infer elegant or formal style, wedding occasion
   
   However, do NOT infer values for "exclude" unless the user explicitly
   expresses negation.

4. TYPE MAPPINGS:
   4. TYPE MAPPINGS:
   When the user mentions a generic clothing term, map it to the appropriate 
   specific type(s) from the valid values. If the term is ambiguous or generic,
   include all relevant subtypes.
   
   Common mappings:
   - "t-shirt", "tee" -> "short_sleeve_top"
   - "shirt", "blouse" -> "long_sleeve_top" (or "short_sleeve_top" if context suggests short sleeves)
   - "sweater", "jumper", "pullover", "sweatshirt" -> "long_sleeve_top"
   - "hoodie" -> "long_sleeve_outwear"
   - "jacket", "coat", "blazer", "cardigan", "parka", "windbreaker" -> "long_sleeve_outwear"
   - "tank top", "camisole", "sleeveless top" -> "vest"
   - "pants", "jeans", "slacks", "chinos", "leggings" -> "trousers"
   - "dress" (generic) -> include ALL dress types: ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"]
   - "top" (generic) -> include ALL top types: ["short_sleeve_top", "long_sleeve_top", "vest"]
   - "slip dress", "strappy dress", "spaghetti strap dress" -> "sling_dress"
   - "sleeveless dress" -> "vest_dress"
   - "gown", "maxi dress", "evening dress" -> include ALL dress types
   - "outerwear", "outer layer" (generic) -> "long_sleeve_outwear"
   - "bottoms" (generic) -> include ALL bottom types: ["shorts", "trousers", "skirt"]

5. COLOR EXPANSIONS:
   - "dark colors" -> ["black", "navy", "burgundy", "olive", "brown"]
   - "vivid colors", "bold colors" -> ["white", "yellow", "orange", "pink", "red", "purple", "coral", "teal"]
   - "neutral colors" -> ["white", "gray", "beige", "cream", "brown"]
   - "light colors" -> ["white", "beige", "cream", "pink", "yellow"]
   
   If the user says "not dark colors" or "avoid dark colors", put the dark colors 
   in "exclude" and do NOT add light colors to "include".

6. PATTERN NEGATION:
   - If the user says they do not want patterns (e.g., "no patterns", "without patterns"),
     set "pattern" to ["plain"] in "include".
   - If the user says they do not want a specific pattern (e.g., "no floral"),
     put only that pattern in "exclude" and do NOT add anything to "include" for pattern.

7. AGE GROUP RANGES:
   - "baby": (0, 2)
   - "child": (3, 12)
   - "teenager": (13, 17)
   - "young adult": (18, 29)
   - "adult": (30, 59)
   - "senior": (60, 120)

8. EMPTY SECTIONS:
   - If there are no inclusions, set "include" to {{}}.
   - If there are no exclusions, set "exclude" to {{}}.
   - Omit any section that is empty.

EXAMPLES:

Example 1:
  User: "I want a black T-shirt that is not too fitted, I want it to be casual"
  Output:
  {{
    "include": {{
      "color": ["black"],
      "type": ["short_sleeve_top"],
      "style": ["casual"]
    }},
    "exclude": {{
      "fit": ["fitted"]
    }}
  }}

Example 2:
  User: "I am going to have a very important meeting with a CEO of a company, what do you recommend"
  Output:
  {{
    "include": {{
      "style": ["formal", "smart casual"],
      "occasion": ["work"],
      "color": ["black", "navy", "gray"]
    }},
    "exclude": {{}}
  }}

Example 3:
  User: "Show me summer dresses, but no floral"
  Output:
  {{
    "include": {{
      "type": ["short_sleeve_dress", "long_sleeve_dress", "vest_dress", "sling_dress"],
      "season": ["summer"]
    }},
    "exclude": {{
      "pattern": ["floral"]
    }}
  }}

Example 4:
  User: "casual pants that are not skinny and not too long"
  Output:
  {{
    "include": {{
      "type": ["trousers"],
      "style": ["casual"]
    }},
    "exclude": {{
      "fit": ["slim fit"],
      "length": ["full length", "ankle"]
    }}
  }}

Example 5:
  User: "I need something for a beach vacation, avoid dark colors"
  Output:
  {{
    "include": {{
      "season": ["summer"],
      "occasion": ["beach"],
      "style": ["casual"]
    }},
    "exclude": {{
      "color": ["black", "navy", "burgundy", "olive", "brown"]
    }}
  }}

After creating the JSON, verify:
- Are negated values placed in "exclude" and not "include"?
- Did you avoid adding opposite values to "include" when the user only expressed a negation?
- Is the JSON valid and properly formatted?

If any check fails, fix the JSON before outputting.
"""

def _build_user_prompt(query: str) -> str:
    return f'User query: "{query}"'


def _build_refinement_system_prompt() -> str:
    """Build the system prompt for iterative query refinement."""

    field_lines = []
    for field in FILTERABLE_FIELDS:
        keys = sorted(ALL_MAPPINGS[field].keys())
        field_lines.append(f'  "{field}": {json.dumps(keys)}')

    fields_block = ",\n".join(field_lines)

    free_text_block = "\n".join(
        f'  "{f}": <free text, copy exact value>'
        for f in FREE_TEXT_FILTER_FIELDS
    )

    return f"""\
You update a clothing-search state from a follow-up message.

You receive:
1) previous semantic query text
2) previous filters JSON
3) user follow-up refinement

VALID FIELD VALUES (use ONLY these exact strings for closed-set fields):
{{
{fields_block}
}}

FREE-TEXT FIELDS:
{{
{free_text_block}
}}

Return ONLY JSON with this exact schema:
{{
  "query": "<updated semantic query text>",
  "filters": {{
    "include": {{"<field>": ["<value>"]}},
    "exclude": {{"<field>": ["<value>"]}}
  }}
}}

Rules:
- Keep previous intent unless the refinement changes it.
- If user says "also/add/include", add constraints.
- If user says "instead/change/swap/replace", replace old value with new value in the same field.
- If user negates something ("not", "don't", "without"), place that value in "exclude".
- Ensure include/exclude never contain the same value for the same field.
- Keep query text consistent with the final filters and concise.
- If a section is empty, you may omit it.
- If follow-up is ambiguous, prefer minimal safe changes.
"""


def _build_refinement_user_prompt(previous_query: str, previous_filters: dict, refinement: str) -> str:
    return (
        f"Previous semantic query: {json.dumps(previous_query)}\n"
        f"Previous filters: {json.dumps(previous_filters, ensure_ascii=False)}\n"
        f"User refinement: {json.dumps(refinement)}"
    )


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
        Dict with keys "include", "exclude" (either may be absent).
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


def refine_query(
    previous_query: str,
    previous_filters: dict,
    refinement: str,
    model: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Update semantic query text + filters using a follow-up refinement.

    Args:
        previous_query: Existing semantic query string.
        previous_filters: Existing parsed filters.
        refinement: User follow-up text (e.g., "also plain", "white instead").
        model: Ollama model name (defaults to OLLAMA_MODEL).
        verbose: Print raw LLM response.

    Returns:
        Dict with keys:
            "query": updated semantic query string
            "filters": updated validated filter dict
    """
    model = model or OLLAMA_MODEL

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _build_refinement_system_prompt()},
            {
                "role": "user",
                "content": _build_refinement_user_prompt(
                    previous_query=previous_query,
                    previous_filters=previous_filters,
                    refinement=refinement,
                ),
            },
        ],
        options={"temperature": 0},
    )

    raw = response.message.content.strip()

    if verbose:
        print(f"\n  [LLM raw refinement response]\n  {raw}\n")

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print("  WARNING: LLM returned invalid refinement JSON. Keeping previous state.")
        return {
            "query": f"{previous_query} {refinement}".strip(),
            "filters": _validate(dict(previous_filters)),
        }

    updated_query = parsed.get("query")
    if not isinstance(updated_query, str) or not updated_query.strip():
        updated_query = f"{previous_query} {refinement}".strip()

    updated_filters = parsed.get("filters", previous_filters)
    if not isinstance(updated_filters, dict):
        updated_filters = previous_filters

    return {
        "query": updated_query.strip(),
        "filters": _validate(updated_filters),
    }


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
