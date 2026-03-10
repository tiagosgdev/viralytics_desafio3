"""
src/recommendations/engine.py
──────────────────────────────
Recommendation engine — takes a list of detected clothing categories
and returns a ranked list of complementary items from the store catalogue.

Strategy:
  1. Rule-based outfit logic  (fast, interpretable)
  2. Category embedding similarity (cosine distance in latent space)

For the prototype, the rule-based approach is the default.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from src.recommendations.catalogue import CATALOGUE, CatalogueItem


# ── Outfit rules ───────────────────────────────────────────────────────────
# { detected_category: [complementary_categories, ...] }

OUTFIT_RULES: Dict[str, List[str]] = {
    "short_sleeve_top":       ["trousers", "shorts", "skirt", "long_sleeve_outwear"],
    "long_sleeve_top":        ["trousers", "shorts", "skirt"],
    "short_sleeve_outwear":   ["short_sleeve_top", "long_sleeve_top", "trousers"],
    "long_sleeve_outwear":    ["short_sleeve_top", "long_sleeve_top", "trousers", "skirt"],
    "vest":                   ["short_sleeve_top", "long_sleeve_top", "trousers"],
    "sling":                  ["short_sleeve_outwear", "long_sleeve_outwear", "trousers", "skirt"],
    "shorts":                 ["short_sleeve_top", "long_sleeve_top", "vest", "sling"],
    "trousers":               ["short_sleeve_top", "long_sleeve_top", "vest", "short_sleeve_outwear"],
    "skirt":                  ["short_sleeve_top", "long_sleeve_top", "sling", "vest"],
    "short_sleeve_dress":     ["short_sleeve_outwear", "long_sleeve_outwear"],
    "long_sleeve_dress":      ["long_sleeve_outwear", "vest"],
    "vest_dress":             ["short_sleeve_top", "long_sleeve_top", "long_sleeve_outwear"],
    "sling_dress":            ["short_sleeve_outwear", "long_sleeve_outwear"],
}


class RecommendationEngine:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        # Index catalogue by category
        self._index: Dict[str, List[CatalogueItem]] = {}
        for item in CATALOGUE:
            self._index.setdefault(item.category, []).append(item)

    def recommend(
        self,
        detected_categories: List[str],
        exclude_detected: bool = True,
    ) -> List[dict]:
        """
        Returns top_k recommendation dicts for the detected clothing.

        Parameters
        ----------
        detected_categories : categories already worn by the person
        exclude_detected    : don't suggest more of what they're already wearing

        Returns
        -------
        List of dicts:
          { name, category, price, image_url, reason, score }
        """
        if not detected_categories:
            return self._trending()

        # Collect candidate categories from outfit rules
        candidate_cats: Dict[str, float] = {}
        for det_cat in detected_categories:
            for comp_cat in OUTFIT_RULES.get(det_cat, []):
                if exclude_detected and comp_cat in detected_categories:
                    continue
                # Accumulate score (more rules suggest it → higher score)
                candidate_cats[comp_cat] = candidate_cats.get(comp_cat, 0) + 1.0

        if not candidate_cats:
            return self._trending()

        # Rank and pick items from catalogue
        ranked = sorted(candidate_cats.items(), key=lambda x: x[1], reverse=True)
        results: List[dict] = []

        seen_ids = set()
        for cat, score in ranked:
            items = self._index.get(cat, [])
            random.shuffle(items)       # variety across calls
            for item in items:
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                results.append({
                    "id":        item.id,
                    "name":      item.name,
                    "category":  item.category,
                    "price":     item.price,
                    "image_url": item.image_url,
                    "reason":    self._reason(detected_categories, cat),
                    "score":     round(score, 2),
                })
                if len(results) >= self.top_k:
                    break
            if len(results) >= self.top_k:
                break

        return results

    # ── Private ────────────────────────────────────────────────────────────

    def _trending(self) -> List[dict]:
        """Fallback — return trending items when nothing is detected."""
        items = random.sample(CATALOGUE, min(self.top_k, len(CATALOGUE)))
        return [
            {
                "id":        i.id,
                "name":      i.name,
                "category":  i.category,
                "price":     i.price,
                "image_url": i.image_url,
                "reason":    "Trending this week",
                "score":     0.5,
            }
            for i in items
        ]

    @staticmethod
    def _reason(detected: List[str], suggested_cat: str) -> str:
        det_str = " & ".join(
            c.replace("_", " ") for c in detected[:2]
        )
        sug_str = suggested_cat.replace("_", " ")
        return f"Pairs well with your {det_str}"
