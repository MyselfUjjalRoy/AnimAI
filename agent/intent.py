"""
agent/intent.py
───────────────
Classifies an incoming user query as one of three tiers:
  - "bare_topic"          → user typed only a topic name (e.g. "RNN", "Gradient Descent")
  - "simple_explanation"  → user typed a SHORT educational request (e.g. "explain gradient descent")
  - "detailed"            → user wrote a specific, multi-part instruction or complex question

Used by the orchestrator to select the correct Teacher and Planner prompt template.

Both "bare_topic" and "simple_explanation" route through the 5-beat pedagogical arc.
"detailed" uses the comprehensive planner.

Direct problem-solving queries (e.g. "solve x²+3x+2=0", "find the area",
"integrate sin(x) dx") always route to "detailed" — they are NOT topics to teach.
"""

import re

# Words that signal the user is giving an instruction or asking a question,
# not just naming a topic they want explained from scratch.
_INSTRUCTION_VERBS = {
    "explain", "show", "demonstrate", "teach", "describe", "how", "why",
    "what", "when", "where", "create", "make", "generate", "build", "compare",
    "visualise", "visualize", "animate", "illustrate", "walk", "derive",
    "prove", "calculate", "compute", "plot", "draw", "step", "difference",
    "relationship", "examples", "example", "use", "used", "uses",
}

# Simple explanation verbs — these indicate the user wants a topic explained
# but the request is still essentially "teach me topic X".
_SIMPLE_EXPLAIN_VERBS = {
    "explain", "show", "teach", "describe", "what", "how", "demonstrate",
    "visualize", "visualise", "animate", "illustrate",
}

# Filler phrases that wrap a topic name in a simple request
_FILLER_PHRASES = [
    r"^explain\s+(me\s+)?",
    r"^what\s+is\s+(an?\s+)?",
    r"^what\s+are\s+",
    r"^how\s+does\s+(an?\s+)?",
    r"^how\s+do\s+",
    r"^teach\s+(me\s+)?(about\s+)?",
    r"^describe\s+(me\s+)?",
    r"^show\s+(me\s+)?(about\s+)?",
    r"^tell\s+(me\s+)?(about\s+)?",
    r"^demonstrate\s+",
    r"^visualize\s+",
    r"^visualise\s+",
    r"^animate\s+",
    r"^illustrate\s+",
]

# Punctuation marks that strongly indicate a sentence / question, not a topic name
_SENTENCE_PUNCTUATION = re.compile(r"[!,;:]")

# Math / equation symbols that indicate a direct problem, not a topic name
_MATH_SYMBOLS = re.compile(r"[=^∫∑∏∂√÷×]|\d+[+\-*/]\d+")

# Problem-solving verbs — when the user is giving a DIRECT TASK, not asking to
# learn about a topic. These ALWAYS route to "detailed".
_PROBLEM_SOLVING_VERBS = {
    "solve", "find", "evaluate", "simplify", "factor", "factorise", "factorize",
    "integrate", "differentiate", "convert", "verify", "check", "compute",
    "calculate", "derive", "prove", "plot", "graph", "sketch", "draw",
    "determine", "expand", "reduce", "balance",
}

# Complex qualifiers that push a query to "detailed" even if short
_COMPLEX_QUALIFIERS = {
    "compare", "versus", "vs", "difference", "between", "step",
    "detailed", "depth", "advanced", "comprehensive", "minute",
}


def classify_query_intent(query: str) -> str:
    """Return ``"bare_topic"``, ``"simple_explanation"``, or ``"detailed"``
    for a user query.

    Classification logic (evaluated in order):

    1. If the query is ≤ 8 words, contains no instruction verbs, and no
       sentence-level punctuation → ``"bare_topic"``.

    2. If the query is ≤ 15 words, starts with a simple explain-verb
       (e.g. "explain", "what is", "teach me"), contains no complex
       qualifiers, and has no sentence punctuation → ``"simple_explanation"``.

    3. Everything else → ``"detailed"``.

    Examples
    --------
    >>> classify_query_intent("RNN")
    'bare_topic'
    >>> classify_query_intent("Gradient Descent")
    'bare_topic'
    >>> classify_query_intent("explain gradient descent")
    'simple_explanation'
    >>> classify_query_intent("what is backpropagation")
    'simple_explanation'
    >>> classify_query_intent("how does CNN work")
    'simple_explanation'
    >>> classify_query_intent("teach me about transformers")
    'simple_explanation'
    >>> classify_query_intent("solve x^2 + 3x + 2 = 0")
    'detailed'
    >>> classify_query_intent("find the area of a circle with radius 5")
    'detailed'
    >>> classify_query_intent("integrate sin(x) dx")
    'detailed'
    >>> classify_query_intent("create a detailed comparison of RNN vs LSTM")
    'detailed'
    >>> classify_query_intent("explain step by step with examples how backpropagation works")
    'detailed'
    """
    if not query or not query.strip():
        return "detailed"

    q = query.strip()
    tokens = q.lower().split()
    word_count = len(tokens)

    # ── Tier 0: Direct problem / task ──
    # If the query starts with a problem-solving verb OR contains math symbols,
    # this is a direct task (not a topic to teach) → always "detailed".
    first_word = tokens[0].rstrip(".,;:!?") if tokens else ""
    if first_word in _PROBLEM_SOLVING_VERBS:
        return "detailed"
    if _MATH_SYMBOLS.search(q):
        return "detailed"

    # ── Tier 1: Bare topic ──
    # No sentence punctuation, no verbs, ≤ 8 words
    has_sentence_punct = bool(_SENTENCE_PUNCTUATION.search(q))
    # Allow trailing ? for simple questions — it doesn't disqualify simple_explanation
    has_question_mark = "?" in q

    if not has_sentence_punct and not has_question_mark and word_count <= 8:
        has_verb = any(
            tok.rstrip(".,;:!?") in _INSTRUCTION_VERBS for tok in tokens
        )
        if not has_verb:
            return "bare_topic"

    # ── Tier 2: Simple explanation ──
    # Short query with a simple explain-verb, no complex qualifiers
    if word_count <= 15:
        first_word = tokens[0].rstrip(".,;:!?")
        has_complex = any(
            tok.rstrip(".,;:!?") in _COMPLEX_QUALIFIERS for tok in tokens
        )
        # Check if first word (or "what is" pattern) starts a simple request
        is_simple_start = first_word in _SIMPLE_EXPLAIN_VERBS

        if is_simple_start and not has_complex and not has_sentence_punct:
            return "simple_explanation"

    # ── Tier 3: Detailed ──
    return "detailed"


def extract_topic_from_query(query: str) -> str:
    """Extract the core topic name from a simple explanation query.

    Strips common instruction prefixes like "explain", "what is", "teach me about"
    and returns the cleaned topic name with title casing.

    Examples
    --------
    >>> extract_topic_from_query("explain gradient descent")
    'Gradient Descent'
    >>> extract_topic_from_query("what is backpropagation?")
    'Backpropagation'
    >>> extract_topic_from_query("teach me about transformers")
    'Transformers'
    >>> extract_topic_from_query("how does CNN work")
    'CNN Work'
    """
    if not query or not query.strip():
        return query or ""

    cleaned = query.strip()

    # Remove trailing question marks
    cleaned = cleaned.rstrip("?").strip()

    # Apply filler phrase removal (case-insensitive)
    for pattern in _FILLER_PHRASES:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    # Remove trailing filler words
    trailing_fillers = [
        r"\s+works?$", r"\s+working$", r"\s+in\s+detail$",
        r"\s+briefly$", r"\s+simply$", r"\s+please$",
    ]
    for pattern in trailing_fillers:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    if not cleaned:
        return query.strip()

    # Title-case the result, preserving acronyms
    words = cleaned.split()
    titled = []
    for w in words:
        if w.isupper() and len(w) >= 2:
            titled.append(w)  # Keep acronyms like "RNN", "CNN", "LSTM"
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def is_bare_topic(query: str) -> bool:
    """Convenience wrapper — returns True if the query is a bare topic name."""
    return classify_query_intent(query) == "bare_topic"


def is_structured_intent(query: str) -> bool:
    """Returns True if the query should use the structured 7-beat arc.

    Both ``bare_topic`` and ``simple_explanation`` use the structured arc.
    """
    return classify_query_intent(query) in ("bare_topic", "simple_explanation")
