"""
Feedback learning system.
When a user gives thumbs up, we save the code as a new few-shot example.
Over time the system learns what good animations look like.
"""

import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback_examples.json"


def save_good_example(query: str, code: str, category: str, plan: dict = None):
    """
    Called when user gives thumbs up.
    Saves this query+code pair as a future few-shot example.
    """
    os.makedirs("data", exist_ok=True)

    # Load existing feedback
    examples = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                examples = json.load(f)
        except Exception:
            examples = []

    # Add new example
    new_example = {
        "query": query,
        "code": code,
        "category": category,
        "plan": plan,
        "timestamp": datetime.now().isoformat(),
        "upvotes": 1
    }

    # Check if similar query exists — increment upvotes instead
    for ex in examples:
        if ex["query"].lower() == query.lower():
            ex["upvotes"] += 1
            print(f"[Feedback] Updated existing example (upvotes: {ex['upvotes']})")
            with open(FEEDBACK_FILE, "w") as f:
                json.dump(examples, f, indent=2)
            return

    examples.append(new_example)

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"[Feedback] Saved good example for: {query}")
    print(f"[Feedback] Total examples: {len(examples)}")


def get_learned_examples(category: str, k: int = 2) -> list:
    """
    Returns the highest-rated learned examples for a given category.
    These are injected into the prompt alongside the built-in examples.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return []

    try:
        with open(FEEDBACK_FILE, "r") as f:
            examples = json.load(f)
    except Exception:
        return []

    # Filter by category and sort by upvotes
    category_examples = [e for e in examples if e.get("category") == category]
    category_examples.sort(key=lambda x: x.get("upvotes", 0), reverse=True)

    return category_examples[:k]


def get_feedback_stats() -> dict:
    """Returns stats about learned examples."""
    if not os.path.exists(FEEDBACK_FILE):
        return {"total": 0, "by_category": {}}

    try:
        with open(FEEDBACK_FILE, "r") as f:
            examples = json.load(f)
    except Exception:
        return {"total": 0, "by_category": {}}

    by_category = {}
    for ex in examples:
        cat = ex.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    return {"total": len(examples), "by_category": by_category}