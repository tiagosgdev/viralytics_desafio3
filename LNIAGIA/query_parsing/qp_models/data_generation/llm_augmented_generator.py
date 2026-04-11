import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ollama
import spacy

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

try:
    from LNIAGIA.DB import models as db_models
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from DB import models as db_models


# Output configuration
OUTPUT_FOLDER = "LNIAGIA/query_parsing_models/data_generation/output"
OUTPUT_NAME = "llm_augmented_data.json"

# Dataset size configuration
N_EXAMPLES = 2000

# LLM configuration
OLLAMA_MODEL = "qwen2.5:7b-instruct-q3_K_M"
TEMPERATURE = 0.55
NUM_PREDICT = 180

# Generation behavior
SEED = 42
GLOBAL_ONLY_RATIO = 0.8
MAX_RETRIES_PER_EXAMPLE = 3
MAX_FAILURE_RATIO = 0.5
MAX_CORRECTION_ATTEMPTS = 1
MULTI_VALUE_INCLUDE_PROB = 0.2


def _load_nlp():
    """Use the small English model when available, fallback to tokenizer-only pipeline."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


nlp = _load_nlp()


@dataclass
class GenerationSpec:
    """Specification for LLM generation."""

    attributes_to_include: Dict[str, List[str]]
    attributes_to_exclude: Dict[str, List[str]]
    style_hints: List[str]


class LLMAugmentedGenerator:
    """Generate natural-language queries with an LLM and validate outputs."""

    GENERATION_PROMPT = """Generate a natural-sounding clothing search query based on these requirements.

REQUIREMENTS:
- Include these attributes: {include_json}
- Exclude these attributes (use negation words like not, no, avoid, without): {exclude_json}
- Style: {style}

RULES:
1. Write one or two natural sentences a real person might say.
2. Mention all values from every include field.
3. Mention every excluded value with explicit negation near that value.
4. Do not list attributes mechanically.
5. Vary wording and sentence structure.
6. Keep it concise and clear.q

GOOD EXAMPLES:
- Include: {{"color": ["black"], "type": ["t-shirt"], "style": ["casual"]}}
    Exclude: {{"fit": ["fitted"]}}
    Query: "I'm looking for a casual black t-shirt, but not fitted."

- Include: {{"occasion": ["wedding"], "color": ["navy"], "type": ["dress"]}}
    Exclude: {{"pattern": ["floral"]}}
    Query: "Can you show me a navy dress for a wedding, with no floral pattern?"

- Include: {{"material": ["linen"], "season": ["summer"], "type": ["pants"]}}
    Exclude: {{"color": ["black"]}}
    Query: "Need summer linen pants, but avoid black."

CHECKLIST BEFORE OUTPUT:
- Did I include all include fields?
- Did I negate each excluded value explicitly?
- Is the text natural and concise?

OUTPUT: Return ONLY the query text.
"""

    CORRECTION_PROMPT = """Rewrite the query so it satisfies all requirements.

INCLUDE REQUIREMENTS:
{include_json}

EXCLUDE REQUIREMENTS:
{exclude_json}

CURRENT QUERY:
"{query}"

ISSUES TO FIX:
{issues}

RULES:
1. Keep it natural and concise (one or two sentences).
2. Mention all values from every include field.
3. Mention each excluded value with explicit negation near the value.
4. Return only the rewritten query.
"""

    STYLES = [
        "casual conversational",
        "polite request",
        "direct command",
        "question form",
        "describing a situation or occasion",
        "informal with contractions",
        "detailed and specific",
        "brief and concise",
    ]

    TYPE_QUERY_ALIASES = {
        "short_sleeve_top": ["t-shirt", "tee"],
        "long_sleeve_top": ["shirt", "blouse", "sweater"],
        "long_sleeve_outwear": ["jacket", "coat", "outerwear"],
        "vest": ["tank top", "sleeveless top"],
        "shorts": ["shorts"],
        "trousers": ["pants", "trousers", "jeans"],
        "skirt": ["skirt"],
        "short_sleeve_dress": ["short sleeve dress"],
        "long_sleeve_dress": ["long sleeve dress"],
        "vest_dress": ["sleeveless dress"],
        "sling_dress": ["slip dress", "strappy dress"],
    }

    NEGATION_WORDS = ["not", "no", "don't", "avoid", "without", "never", "nothing", "hate"]

    def __init__(self, model: str = OLLAMA_MODEL, seed: int = SEED):
        self.model = model
        random.seed(seed)
        self.counter = 0

        self.global_fields = [
            field for field in db_models.FILTERABLE_FIELDS if field in db_models.STATIC_GLOBAL_FIELDS
        ]
        self.all_fields = [
            field
            for field in db_models.FILTERABLE_FIELDS
            if field in db_models.STATIC_GLOBAL_FIELDS or field in db_models.EXTRA_FIELD_VALUES
        ]
        self.field_values = self._build_field_values()
        self.type_alias_to_schema = self._build_type_alias_to_schema()
        self.type_alias_values = sorted(self.type_alias_to_schema.keys())

    def _build_field_values(self) -> Dict[str, List[str]]:
        """Build all generation values directly from models.py."""
        values = {}
        for field in self.all_fields:
            if field in db_models.STATIC_GLOBAL_FIELDS:
                values[field] = list(db_models.STATIC_GLOBAL_FIELDS[field])
            elif field in db_models.EXTRA_FIELD_VALUES:
                values[field] = list(db_models.EXTRA_FIELD_VALUES[field])
        return values

    def _build_type_alias_to_schema(self) -> Dict[str, List[str]]:
        alias_map: Dict[str, List[str]] = {}
        for schema_type, aliases in self.TYPE_QUERY_ALIASES.items():
            for alias in aliases:
                alias_key = alias.lower()
                alias_map.setdefault(alias_key, [])
                if schema_type not in alias_map[alias_key]:
                    alias_map[alias_key].append(schema_type)
        return alias_map

    def generate_id(self) -> str:
        self.counter += 1
        return f"llm_aug_{self.counter:05d}"

    def _sample_values(self, field: str, max_values: int = 2) -> List[str]:
        """Sample one or more values for a field."""
        if field == "type":
            max_possible = min(max_values, len(self.type_alias_values))
            if max_possible <= 1:
                n_values = 1
            else:
                n_values = 2 if random.random() < MULTI_VALUE_INCLUDE_PROB else 1
            return random.sample(self.type_alias_values, n_values)

        candidates = self.field_values.get(field, [])
        if not candidates:
            return []
        max_possible = min(max_values, len(candidates))
        if max_possible <= 1:
            n_values = 1
        else:
            n_values = 2 if random.random() < MULTI_VALUE_INCLUDE_PROB else 1
        return random.sample(candidates, n_values)

    def _pick_field_pool(self) -> List[str]:
        """Use mostly global fields, with a smaller share of all filterable fields."""
        if random.random() < GLOBAL_ONLY_RATIO:
            return self.global_fields
        return self.all_fields

    def create_random_spec(self) -> GenerationSpec:
        """Create a random generation specification from models.py values."""
        field_pool = self._pick_field_pool()

        n_include = random.randint(1, min(4, len(field_pool)))
        include_keys = random.sample(field_pool, n_include)
        include_attrs: Dict[str, List[str]] = {}

        for key in include_keys:
            sampled = self._sample_values(key, max_values=2)
            if sampled:
                include_attrs[key] = sampled

        n_exclude = random.choices([0, 1, 2], weights=[0.35, 0.45, 0.2], k=1)[0]
        exclude_attrs: Dict[str, List[str]] = {}

        if n_exclude > 0:
            preferred_negation_fields = {"color", "pattern", "fit", "material", "length", "style", "type"}
            candidate_fields = [
                f for f in field_pool if (f in preferred_negation_fields or f in include_attrs)
            ]
            if candidate_fields:
                exclude_keys = random.sample(candidate_fields, min(n_exclude, len(candidate_fields)))
                for key in exclude_keys:
                    if key == "type":
                        all_values = self.type_alias_values
                    else:
                        all_values = self.field_values.get(key, [])

                    included_values = include_attrs.get(key, [])
                    available_values = [v for v in all_values if v not in included_values]
                    if available_values:
                        exclude_attrs[key] = [random.choice(available_values)]

        return GenerationSpec(
            attributes_to_include=include_attrs,
            attributes_to_exclude=exclude_attrs,
            style_hints=[random.choice(self.STYLES)],
        )

    def _build_generation_prompt(self, spec: GenerationSpec) -> str:
        return self.GENERATION_PROMPT.format(
            include_json=json.dumps(spec.attributes_to_include, ensure_ascii=False),
            exclude_json=json.dumps(spec.attributes_to_exclude, ensure_ascii=False),
            style=spec.style_hints[0],
        )

    def _build_correction_prompt(self, query: str, spec: GenerationSpec, issues: List[str]) -> str:
        return self.CORRECTION_PROMPT.format(
            include_json=json.dumps(spec.attributes_to_include, ensure_ascii=False),
            exclude_json=json.dumps(spec.attributes_to_exclude, ensure_ascii=False),
            query=query,
            issues="; ".join(issues) if issues else "Unknown issue",
        )

    def generate_query(self, spec: GenerationSpec) -> Optional[str]:
        """Generate one query with Ollama."""
        prompt = self._build_generation_prompt(spec)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You generate realistic user search queries."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
            )

            if isinstance(response, dict):
                query = response.get("message", {}).get("content", "")
            else:
                query = getattr(getattr(response, "message", None), "content", "")

            query = query.strip().strip('"\'')
            query = re.sub(r"^(query:|output:)\s*", "", query, flags=re.IGNORECASE)
            query = re.sub(r"\s+", " ", query).strip()

            return query or None
        except Exception as exc:
            print(f"LLM generation error: {exc}")
            return None

    @staticmethod
    def _pluralise_phrase(text: str) -> str:
        text = text.strip().lower()
        if not text:
            return text
        if text.endswith("s"):
            return text
        if text.endswith("y") and len(text) > 1 and text[-2] not in "aeiou":
            return text[:-1] + "ies"
        if text.endswith(("ch", "sh", "x", "z")):
            return text + "es"
        return text + "s"

    def _value_variants(self, field: str, value: str) -> List[str]:
        base = value.lower().strip()
        variants = {base, base.replace("-", " "), base.replace("_", " ")}

        if field == "type":
            variants.add(self._pluralise_phrase(base))
            for schema_type in self.type_alias_to_schema.get(base, []):
                for alias in self.TYPE_QUERY_ALIASES.get(schema_type, []):
                    alias_l = alias.lower()
                    variants.add(alias_l)
                    variants.add(self._pluralise_phrase(alias_l))

        return [v for v in variants if v]

    def _has_negation_near(self, text: str, phrase: str, window_chars: int = 40) -> bool:
        phrase_l = phrase.lower()
        text_l = text.lower()
        for match in re.finditer(re.escape(phrase_l), text_l):
            start, end = match.start(), match.end()
            snippet = text_l[max(0, start - window_chars) : min(len(text_l), end + window_chars)]
            if any(neg in snippet for neg in self.NEGATION_WORDS):
                return True
        return False

    def validate_query(self, query: str, spec: GenerationSpec) -> bool:
        """Validate generated query against basic requirements from the specification."""
        return len(self._validation_issues(query, spec)) == 0

    def _validation_issues(self, query: str, spec: GenerationSpec) -> List[str]:
        """Return validation issues to guide automatic rewrite passes."""
        issues: List[str] = []

        if len(query) < 10 or len(query) > 500:
            issues.append("Query length must be between 10 and 500 characters.")
            return issues

        query_lower = query.lower()

        for field, values in spec.attributes_to_include.items():
            field_hit = False
            for value in values:
                variants = self._value_variants(field, value)
                if any(variant in query_lower for variant in variants):
                    field_hit = True
                    break
            if not field_hit:
                issues.append(f"Missing include field '{field}' with one of values {values}.")

        if spec.attributes_to_exclude and not any(word in query_lower for word in self.NEGATION_WORDS):
            issues.append("Missing explicit negation words for excluded values.")

        for field, values in spec.attributes_to_exclude.items():
            for value in values:
                variants = self._value_variants(field, value)
                found_variant = None
                for variant in variants:
                    if variant in query_lower:
                        found_variant = variant
                        break

                if found_variant is None:
                    issues.append(f"Missing excluded value '{value}' in query text.")
                    continue

                if not self._has_negation_near(query, found_variant):
                    issues.append(f"Excluded value '{value}' is not explicitly negated.")

        return issues

    def correct_query(self, query: str, spec: GenerationSpec, issues: List[str]) -> Optional[str]:
        """Ask the LLM to rewrite an invalid query so it satisfies the specification."""
        prompt = self._build_correction_prompt(query=query, spec=spec, issues=issues)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You rewrite queries to satisfy strict constraints."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.2, "num_predict": NUM_PREDICT},
            )

            if isinstance(response, dict):
                corrected = response.get("message", {}).get("content", "")
            else:
                corrected = getattr(getattr(response, "message", None), "content", "")

            corrected = corrected.strip().strip('"\'')
            corrected = re.sub(r"^(query:|output:)\s*", "", corrected, flags=re.IGNORECASE)
            corrected = re.sub(r"\s+", " ", corrected).strip()
            return corrected or None
        except Exception as exc:
            print(f"LLM correction error: {exc}")
            return None

    def normalise_spec(self, spec: GenerationSpec) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Convert type aliases to schema values while preserving other fields."""
        include: Dict[str, List[str]] = {}
        exclude: Dict[str, List[str]] = {}

        for key, values in spec.attributes_to_include.items():
            if key == "type":
                mapped_types: List[str] = []
                for value in values:
                    mapped_types.extend(self.type_alias_to_schema.get(value.lower(), [value]))
                include["type"] = sorted(set(mapped_types))
            else:
                include[key] = values

        for key, values in spec.attributes_to_exclude.items():
            if key == "type":
                mapped_types = []
                for value in values:
                    mapped_types.extend(self.type_alias_to_schema.get(value.lower(), [value]))
                exclude["type"] = sorted(set(mapped_types))
            else:
                exclude[key] = values

        return include, exclude

    @staticmethod
    def tokenise(text: str) -> List[str]:
        doc = nlp(text)
        return [token.text for token in doc]

    def generate_example(self, max_retries: int = MAX_RETRIES_PER_EXAMPLE) -> Optional[Dict]:
        """Generate one validated example, retrying if needed."""
        for _ in range(max_retries):
            spec = self.create_random_spec()
            query = self.generate_query(spec)

            if not query:
                continue

            issues = self._validation_issues(query, spec)
            was_rewritten = False

            correction_attempts = 0
            while issues and correction_attempts < MAX_CORRECTION_ATTEMPTS:
                corrected = self.correct_query(query=query, spec=spec, issues=issues)
                correction_attempts += 1
                if not corrected:
                    break

                query = corrected
                was_rewritten = True
                issues = self._validation_issues(query, spec)

            if not issues:
                include, exclude = self.normalise_spec(spec)
                return {
                    "id": self.generate_id(),
                    "query": query,
                    "tokens": self.tokenise(query),
                    "include": include,
                    "exclude": exclude,
                    "generation_spec": {
                        "include": spec.attributes_to_include,
                        "exclude": spec.attributes_to_exclude,
                        "style": spec.style_hints,
                        "rewritten": was_rewritten,
                    },
                }

        return None

    def generate_dataset(self, n_examples: int = N_EXAMPLES) -> List[Dict]:
        """Generate a validated dataset of LLM-authored user queries."""
        examples: List[Dict] = []
        failures = 0

        print(f"Generating {n_examples} LLM-augmented examples...")

        progress = tqdm(total=n_examples, desc="Generating", unit="example") if tqdm is not None else None

        try:
            while len(examples) < n_examples:
                example = self.generate_example()
                if example:
                    examples.append(example)
                    if progress is not None:
                        progress.update(1)
                        if failures:
                            progress.set_postfix(failures=failures)
                    elif len(examples) % 100 == 0:
                        print(f"  Generated {len(examples)}/{n_examples}")
                else:
                    failures += 1
                    if progress is not None:
                        progress.set_postfix(failures=failures)
                    if failures > int(n_examples * MAX_FAILURE_RATIO):
                        print("Too many failures, stopping early")
                        break
        finally:
            if progress is not None:
                progress.close()

        print(f"Generated {len(examples)} examples ({failures} failures)")
        return examples

    @staticmethod
    def save_dataset(examples: List[Dict], filepath: str):
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(examples, file, indent=2, ensure_ascii=False)
        print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    generator = LLMAugmentedGenerator(model=OLLAMA_MODEL, seed=SEED)
    examples = generator.generate_dataset(n_examples=N_EXAMPLES)

    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME

    generator.save_dataset(examples, str(output_path))
