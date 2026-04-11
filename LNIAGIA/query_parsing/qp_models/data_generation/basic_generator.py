import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import spacy

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from LNIAGIA.query_parsing.qp_models.data_generation.bio_projection import project_bio_labels

try:
    from LNIAGIA.DB import models as db_models
except ModuleNotFoundError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from DB import models as db_models


def _load_nlp():
    """Use en_core_web_sm when available, fallback to blank English tokenizer."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


nlp = _load_nlp()

# Output configuration
OUTPUT_FOLDER = "LNIAGIA/query_parsing_models/data_generation/single_outputs"
OUTPUT_NAME = "basic_generated_data.json"

# Dataset size configuration
N_SIMPLE_EXAMPLES = 1500
N_NEGATION_EXAMPLES = 1500
N_CONTEXTUAL_EXAMPLES = 500

# Generation mix configuration
GLOBAL_ONLY_RATIO = 0.8


@dataclass
class GeneratedExample:
    """Single training example"""
    id: str
    query: str
    tokens: List[str]
    bio_labels: List[str]
    include: Dict[str, List[str]]
    exclude: Dict[str, List[str]]


class TemplateBasedGenerator:
    """
    Generates training data using templates with slot filling.
    Guarantees correctness through deterministic construction.
    """
    
    # Template patterns with slots
    # {ATTR} = attribute value, {TYPE} = clothing type
    # [NEG] = optional negation, [ADJ] = optional adjective
    
    SIMPLE_TEMPLATES = [
        # Single attribute
        "I want a {COLOR} {TYPE}",
        "Show me {COLOR} {TYPE_PLURAL}",
        "Looking for a {STYLE} {TYPE}",
        "I need a {PATTERN} {TYPE}",
        "Find me {FIT} {TYPE_PLURAL}",
        "I am looking for {TYPE_PLURAL} for {OCCASION}",
        "{TYPE_PLURAL} for {SEASON}",
        "Do you have any {COLOR} {TYPE_PLURAL}?",
        "Can you show me {STYLE} {TYPE_PLURAL}?",
        "I would like a {MATERIAL} {TYPE}",
    ]
    
    MULTI_ATTRIBUTE_TEMPLATES = [
        # Two attributes
        "I want a {COLOR} {STYLE} {TYPE}",
        "Show me {FIT} {COLOR} {TYPE_PLURAL}",
        "Looking for {STYLE} {TYPE_PLURAL} in {COLOR}",
        "{COLOR} {TYPE} for {OCCASION}",
        "{STYLE} {PATTERN} {TYPE}",
        "I need a {MATERIAL} {TYPE} in {COLOR}",
        "{FIT} {STYLE} {TYPE_PLURAL} for {SEASON}",
        
        # Three+ attributes
        "I want a {COLOR} {STYLE} {TYPE} for {OCCASION}",
        "{FIT} {COLOR} {PATTERN} {TYPE}",
        "Show me {STYLE} {TYPE_PLURAL} in {COLOR} for {SEASON}",
        "Looking for a {MATERIAL} {COLOR} {TYPE} that is {STYLE}",
    ]

    TYPE_SPECIFIC_TEMPLATES = [
        "I want a {TYPE} with {NECKLINE}",
        "Show me {TYPE_PLURAL} with {CLOSURE}",
        "Looking for {TYPE_PLURAL} with {WAIST_STYLE} and {LENGTH}",
        "I need a {TYPE} with {SLEEVE_STYLE} and {HEM_STYLE}",
        "Can you show me {TYPE_PLURAL} with {DRESS_STYLE}",
        "I want {TYPE_PLURAL} with {OUTWEAR_POCKETS}",
    ]
    
    NEGATION_TEMPLATES = [
        # Single negation
        "I want a {TYPE} but not {NEG_COLOR}",
        "Show me {TYPE_PLURAL}, not {NEG_PATTERN}",
        "{TYPE} without {NEG_PATTERN} pattern",
        "I need a {COLOR} {TYPE}, not too {NEG_FIT}",
        "Looking for {TYPE_PLURAL} that are not {NEG_STYLE}",
        "I want {STYLE} {TYPE_PLURAL}, avoid {NEG_COLOR}",
        "{TYPE} for {OCCASION}, nothing {NEG_STYLE}",
        "Show me {TYPE_PLURAL} but no {NEG_PATTERN}",
        "I do not want {NEG_COLOR} {TYPE_PLURAL}",
        "Anything but {NEG_PATTERN} {TYPE_PLURAL}",
        
        # Multiple negations
        "I want a {COLOR} {TYPE}, not {NEG_FIT} and not {NEG_PATTERN}",
        "{TYPE} without {NEG_PATTERN} and not too {NEG_FIT}",
        "Show me {TYPE_PLURAL}, no {NEG_COLOR} and no {NEG_PATTERN}",
        
        # Mixed positive and negative
        "I want a {COLOR} {STYLE} {TYPE}, but not {NEG_FIT}",
        "{FIT} {TYPE} in {COLOR}, avoid {NEG_PATTERN}",
        "Looking for {STYLE} {TYPE_PLURAL} for {OCCASION}, nothing {NEG_COLOR}",
    ]

    TYPE_SPECIFIC_NEGATION_TEMPLATES = [
        "I want a {TYPE} with {NECKLINE}, but not {NEG_COLLAR}",
        "Show me {TYPE_PLURAL} with {CLOSURE}, not {NEG_HOOD}",
        "Looking for {TYPE_PLURAL} with {WAIST_STYLE}, not {NEG_LEG_STYLE}",
        "I need a {TYPE} with {DRESS_STYLE}, avoid {NEG_LENGTH}",
    ]
    
    CONTEXTUAL_TEMPLATES = [
        # Occasion-based (requires inference)
        "I have a job interview tomorrow",
        "Going to a beach party this weekend",
        "I need something for a wedding",
        "What should I wear to a business meeting?",
        "Outfit for a first date",
        "Something comfortable for working from home",
        "I am going hiking this weekend",
        "Attending a formal dinner",
        "Casual Friday at the office",
        "Going to the gym",
    ]
    
    COMPLEX_TEMPLATES = [
        # Long, natural queries
        "I am looking for a {COLOR} {TYPE} that is {STYLE} and {FIT}, preferably for {OCCASION}",
        "Can you help me find {TYPE_PLURAL} in {COLOR}? I want something {STYLE} but not {NEG_FIT}",
        "I need a new {TYPE} for {SEASON}. I prefer {COLOR} and {STYLE} styles, but please no {NEG_PATTERN}",
        "Show me your best {STYLE} {TYPE_PLURAL}. I like {COLOR} and {COLOR2}, but I hate {NEG_PATTERN}",
        "Looking for {FIT} {TYPE_PLURAL} that would work for {OCCASION}. I do not want anything {NEG_STYLE}",
    ]
    
    GLOBAL_FILTERABLE_FIELDS = [
        field for field in db_models.FILTERABLE_FIELDS
        if field in db_models.STATIC_GLOBAL_FIELDS
    ]

    TYPE_SPECIFIC_FILTERABLE_FIELDS = [
        field for field in db_models.FILTERABLE_FIELDS
        if field in db_models.EXTRA_FIELD_VALUES
    ]
    
    # Contextual inference mappings
    CONTEXT_INFERENCES = {
        "job interview": {"style": ["formal", "smart casual"], "occasion": ["work"]},
        "beach party": {"season": ["summer"], "occasion": ["beach", "party"], "style": ["casual"]},
        "wedding": {"style": ["elegant", "formal"], "occasion": ["wedding"]},
        "business meeting": {"style": ["formal", "smart casual"], "occasion": ["work"]},
        "first date": {"style": ["smart casual", "elegant"], "occasion": ["date night"]},
        "working from home": {"style": ["casual"], "occasion": ["lounge"]},
        "hiking": {"style": ["sporty"], "occasion": ["sport"]},
        "formal dinner": {"style": ["formal", "elegant"], "occasion": ["formal event"]},
        "casual Friday": {"style": ["smart casual", "casual"], "occasion": ["work"]},
        "gym": {"style": ["sporty"], "occasion": ["sport"]},
    }
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.counter = 0
        self.attributes = self._build_attributes_from_models()
        self.type_to_query = {item_type: item_type.replace("_", " ") for item_type in db_models.TYPE}
        self.type_plural = {
            item_type: self._pluralise_type(self.type_to_query[item_type])
            for item_type in db_models.TYPE
        }

    def _build_attributes_from_models(self) -> Dict[str, List[str]]:
        """Build attribute dictionary from FILTERABLE_FIELDS in models.py."""
        attributes = {}

        for field in db_models.FILTERABLE_FIELDS:
            values = None

            if field in db_models.STATIC_GLOBAL_FIELDS:
                values = db_models.STATIC_GLOBAL_FIELDS[field]
            elif field in db_models.EXTRA_FIELD_VALUES:
                values = db_models.EXTRA_FIELD_VALUES[field]

            if values:
                attributes[field.upper()] = list(values)

        return attributes

    @staticmethod
    def _pluralise_type(type_text: str) -> str:
        """Simple pluralisation for human-readable type names."""
        if type_text.endswith("s"):
            return type_text
        if type_text.endswith("y") and len(type_text) > 1 and type_text[-2] not in "aeiou":
            return type_text[:-1] + "ies"
        if type_text.endswith(("ch", "sh", "x", "z")):
            return type_text + "es"
        return type_text + "s"

    def _uses_only_global_fields(self, template: str) -> bool:
        """Check whether a template uses only global filterable fields."""
        placeholders = re.findall(r"\{(?:NEG_)?([A-Z_]+)\}", template)
        for placeholder in placeholders:
            if placeholder in {"TYPE", "TYPE_PLURAL"}:
                continue
            if placeholder.lower() not in self.GLOBAL_FILTERABLE_FIELDS:
                return False
        return True

    def _template_pool(self, templates: List[str], ratio_global_only: float) -> str:
        """Select templates with an 80/20 global-only vs broader-field split."""
        global_only = [t for t in templates if self._uses_only_global_fields(t)]
        broader = [t for t in templates if not self._uses_only_global_fields(t)]

        if random.random() < ratio_global_only and global_only:
            return random.choice(global_only)
        if broader:
            return random.choice(broader)
        return random.choice(templates)
    
    def generate_id(self) -> str:
        self.counter += 1
        return f"template_{self.counter:05d}"
    
    def tokenise(self, text: str) -> List[str]:
        """Tokenise using spaCy"""
        doc = nlp(text)
        return [token.text for token in doc]
    
    def generate_bio_labels(
        self,
        tokens: List[str],
        include: Dict[str, List[str]],
        exclude: Dict[str, List[str]]
    ) -> List[str]:
        """Generate BIO labels for tokens"""
        return project_bio_labels(tokens=tokens, include=include, exclude=exclude)
    
    def fill_template(
        self,
        template: str,
        template_type: str
    ) -> Tuple[str, Dict[str, List[str]], Dict[str, List[str]]]:
        """Fill a template with random values and return query + labels"""
        
        include = {}
        exclude = {}
        
        filled = template
        
        # Fill positive attributes from FILTERABLE_FIELDS-backed values
        for attr, values in self.attributes.items():
            if attr == "TYPE":
                continue

            if f"{{{attr}}}" in filled:
                value = random.choice(values)
                filled = filled.replace(f"{{{attr}}}", value, 1)
                key = attr.lower()
                if key not in include:
                    include[key] = []
                include[key].append(value)
            
            # Handle second occurrence (e.g., {COLOR2})
            if f"{{{attr}2}}" in filled:
                available = [v for v in values if v not in include.get(attr.lower(), [])]
                if available:
                    value = random.choice(available)
                    filled = filled.replace(f"{{{attr}2}}", value, 1)
                    key = attr.lower()
                    if key not in include:
                        include[key] = []
                    include[key].append(value)
        
        # Fill TYPE
        if "{TYPE}" in filled:
            type_schema = random.choice(self.attributes["TYPE"])
            type_nl = self.type_to_query[type_schema]
            filled = filled.replace("{TYPE}", type_nl)
            include["type"] = [type_schema]
        
        if "{TYPE_PLURAL}" in filled:
            type_schema = random.choice(self.attributes["TYPE"])
            type_plural = self.type_plural[type_schema]
            filled = filled.replace("{TYPE_PLURAL}", type_plural)
            include["type"] = [type_schema]
        
        # Fill negative attributes
        for attr, values in self.attributes.items():
            if attr == "TYPE":
                continue

            neg_key = f"{{NEG_{attr}}}"
            if neg_key in filled:
                # Choose a value different from included ones
                included_values = include.get(attr.lower(), [])
                available = [v for v in values if v not in included_values]
                if available:
                    value = random.choice(available)
                    filled = filled.replace(neg_key, value, 1)
                    key = attr.lower()
                    if key not in exclude:
                        exclude[key] = []
                    exclude[key].append(value)
        
        return filled, include, exclude
    
    def generate_simple_example(self) -> GeneratedExample:
        """Generate a simple single/multi attribute example"""
        template = self._template_pool(
            self.SIMPLE_TEMPLATES + self.MULTI_ATTRIBUTE_TEMPLATES + self.TYPE_SPECIFIC_TEMPLATES,
            GLOBAL_ONLY_RATIO,
        )
        query, include, exclude = self.fill_template(template, "simple")
        tokens = self.tokenise(query)
        bio_labels = self.generate_bio_labels(tokens, include, exclude)
        
        return GeneratedExample(
            id=self.generate_id(),
            query=query,
            tokens=tokens,
            bio_labels=bio_labels,
            include=include,
            exclude=exclude
        )
    
    def generate_negation_example(self) -> GeneratedExample:
        """Generate an example with negation"""
        template = self._template_pool(
            self.NEGATION_TEMPLATES + self.TYPE_SPECIFIC_NEGATION_TEMPLATES,
            GLOBAL_ONLY_RATIO,
        )
        query, include, exclude = self.fill_template(template, "negation")
        tokens = self.tokenise(query)
        bio_labels = self.generate_bio_labels(tokens, include, exclude)
        
        return GeneratedExample(
            id=self.generate_id(),
            query=query,
            tokens=tokens,
            bio_labels=bio_labels,
            include=include,
            exclude=exclude
        )
    
    def generate_contextual_example(self) -> GeneratedExample:
        """Generate a contextual inference example"""
        template = random.choice(self.CONTEXTUAL_TEMPLATES)
        
        # Find matching context
        include = {}
        exclude = {}
        
        for context_key, inferences in self.CONTEXT_INFERENCES.items():
            if context_key in template.lower():
                include = {k: list(v) for k, v in inferences.items()}
                break
        
        tokens = self.tokenise(template)
        bio_labels = ["O"] * len(tokens)  # Contextual examples have no explicit spans
        
        return GeneratedExample(
            id=self.generate_id(),
            query=template,
            tokens=tokens,
            bio_labels=bio_labels,
            include=include,
            exclude=exclude
        )
    
    def generate_dataset(
        self,
        n_simple: int = 1500,
        n_negation: int = 1500,
        n_contextual: int = 500
    ) -> List[GeneratedExample]:
        """Generate full dataset"""
        
        examples = []

        if tqdm is None:
            print("tqdm is not installed. Falling back to periodic progress logs.")
        
        print(f"Generating {n_simple} simple examples...")
        simple_iter = tqdm(range(n_simple), desc="Basic simple", unit="example") if tqdm is not None else range(n_simple)
        for idx, _ in enumerate(simple_iter, start=1):
            examples.append(self.generate_simple_example())
            if tqdm is None and idx % 100 == 0:
                print(f"  Simple {idx}/{n_simple}")
        
        print(f"Generating {n_negation} negation examples...")
        negation_iter = tqdm(range(n_negation), desc="Basic negation", unit="example") if tqdm is not None else range(n_negation)
        for idx, _ in enumerate(negation_iter, start=1):
            examples.append(self.generate_negation_example())
            if tqdm is None and idx % 100 == 0:
                print(f"  Negation {idx}/{n_negation}")
        
        print(f"Generating {n_contextual} contextual examples...")
        contextual_iter = tqdm(range(n_contextual), desc="Basic contextual", unit="example") if tqdm is not None else range(n_contextual)
        for idx, _ in enumerate(contextual_iter, start=1):
            examples.append(self.generate_contextual_example())
            if tqdm is None and idx % 100 == 0:
                print(f"  Contextual {idx}/{n_contextual}")
        
        random.shuffle(examples)
        
        return examples
    
    def save_dataset(self, examples: List[GeneratedExample], filepath: str):
        """Save dataset to JSON"""
        data = [
            {
                "id": ex.id,
                "query": ex.query,
                "tokens": ex.tokens,
                "bio_labels": ex.bio_labels,
                "include": ex.include,
                "exclude": ex.exclude
            }
            for ex in examples
        ]
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(examples)} examples to {filepath}")


# Generate template-based data
if __name__ == "__main__":
    generator = TemplateBasedGenerator(seed=42)
    
    examples = generator.generate_dataset(
        n_simple=N_SIMPLE_EXAMPLES,
        n_negation=N_NEGATION_EXAMPLES,
        n_contextual=N_CONTEXTUAL_EXAMPLES
    )

    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    
    generator.save_dataset(examples, str(output_path))
    
    # Print samples
    print("\nSample examples:")
    for ex in examples[:5]:
        print(f"\nQuery: {ex.query}")
        print(f"Include: {ex.include}")
        print(f"Exclude: {ex.exclude}")
        print(f"BIO: {list(zip(ex.tokens, ex.bio_labels))}")