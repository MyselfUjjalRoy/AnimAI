import json
import os
import logging
from agent.llm import call_llm

log = logging.getLogger("animai.teacher")

TEACHER_MODEL = os.getenv("GEMINI_TEACHER_MODEL", "gemini-2.5-flash")

# ─────────────────────────────────────────────────────────────────────────────
# 7-BEAT STRUCTURED TEACHING ARC
# Used when the user types a topic name OR a simple educational request
# (e.g. "RNN", "explain gradient descent", "what is backpropagation").
# Produces a strict 7-beat arc with ZERO content overlap between beats.
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURED_TEACHER_PROMPT = """You are a world-class educator. A student wants to learn about: {query}

Your task is to design a 5-beat pedagogical arc for an educational animation.
Each beat is a distinct teaching moment. There must be ZERO content overlap between beats.

━━━━━━━━━━ BEAT 1 — TOPIC NAME (the title slide) ━━━━━━━━━━
Goal: Show ONLY the topic name — large, prominent, visually striking.
Rules:
- on_screen_text: The exact topic name/title (2-6 words max).
- voiceover: ONE sentence introducing the topic by name. "Today we learn about X." or "Let's explore X."
- NO definition. NO theory. NO use case. JUST the name.
- This is a 5-8 second title card.

━━━━━━━━━━ BEAT 2 — WHAT IT IS / THEORY (explain the concept) ━━━━━━━━━━
Goal: Define the topic AND explain the core theory so the student understands it conceptually.
Rules:
- definition: ONE clear sentence defining what the topic IS (a 12-year-old would understand).
- theory_points: 3-5 bullet points covering the core theory, key principles, and important details.
  Each point should teach something CONCRETE — not vague or abstract.
  Include key terms, core mechanisms, and the "clever bit" that makes this topic special.
- on_screen_text: Definition + theory points shown on screen as text.
- voiceover: MUST READ ALOUD everything shown on screen. First the definition, then each theory point.
  ⚠️ CRITICAL: Whatever text appears on screen MUST be spoken by the voiceover. No silent text.
- This segment is TEXT-ONLY. No diagrams, no animations, no flowcharts. Just spoken theory.
- DO NOT include use cases or "why it's needed" here — that's Beat 4.
- DO NOT describe how it works step-by-step — that's Beat 3.

━━━━━━━━━━ BEAT 3 — HOW IT WORKS (mechanism / the visual heart) ━━━━━━━━━━
Goal: Show the internal process visually — this is the HEART of the video.
Rules:
- suggested_visual: What type of visual best explains this topic? Pick ONE:
  "diagram", "flowchart", "equation_walkthrough", "step_animation", "graph_plot",
  "neural_network_diagram", "matrix_grid", "timeline", "comparison_chart"
  Be specific about WHY this visual type is best for this topic.
- visual_description: Describe concretely what the animation/diagram should look like.
- steps: 3-5 concrete steps describing the mechanism with specific values/examples.
  Each step should describe a VISIBLE change in the animation.
- step_voiceovers: One narration line per step (12-22 words each).
- aha_step_index: Which step is the "aha moment" (usually step 2 or 3, 0-indexed).
- This is about PROCESS and FLOW — the visual demonstration of how it works.
- Use CONCRETE numbers, values, and examples — never be abstract.
- DO NOT repeat the definition or theory points from Beat 2.
- DO NOT mention use cases — that's Beat 4.

━━━━━━━━━━ BEAT 4 — NEED / USE CASE (why does this exist?) ━━━━━━━━━━
Goal: Explain WHY this topic is needed and WHERE it is used in the real world.
Rules:
- need: ONE clear sentence explaining the problem this topic solves or why it was created.
- use_cases: 2-4 concrete, real-world use cases or applications.
  Each must be a SPECIFIC example (e.g., "Google Translate uses this for language translation")
  not a vague category (e.g., "used in industry").
- on_screen_text: Need statement + use cases shown on screen as text.
- voiceover: MUST READ ALOUD everything shown on screen. The need, then each use case.
  ⚠️ CRITICAL: Whatever text appears on screen MUST be spoken. No silent text.
- DO NOT re-define the topic (that was Beat 2).
- DO NOT explain how it works (that was Beat 3).
- This is purely about WHY it matters and WHERE it's applied.

━━━━━━━━━━ BEAT 5 — SUMMARY (the takeaway) ━━━━━━━━━━
Goal: Leave the student with ONE memorable, punchy statement.
Rules:
- takeaway: 5-15 words. Memorable. Punchy.
- NOT a repetition of the definition (Beat 2) or the need (Beat 4).
- Should capture the ESSENCE of what they just learned.
- on_screen_text: The takeaway text shown in a green banner.
- voiceover: Read the takeaway with conviction.
  ⚠️ CRITICAL: The voiceover text must MATCH the on-screen takeaway text.

Return ONLY a JSON object in this exact format:
{{
    "topic": "Clear name of the topic",
    "difficulty_level": "beginner or intermediate or advanced",
    "recommended_duration": 65,
    "visual_metaphor": "The concrete image the core animation (Beat 3) is built around",
    "beats": [
        {{
             "beat": 1,
             "label": "Topic Name",
             "on_screen_text": "The exact topic name (2-6 words)",
             "voiceover": "Today we learn about [topic]. (5-12 words)"
        }},
        {{
             "beat": 2,
             "label": "What It Is",
             "definition": "One-sentence definition a 12-year-old would understand",
             "theory_points": [
                 "Core principle 1 — concrete and specific",
                 "Core principle 2 — concrete and specific",
                 "Core principle 3 — what makes this special",
                 "Key detail 4 (optional)"
             ],
             "on_screen_text": "Definition + each theory point shown as text (≤60 words total)",
             "voiceover": "Read the definition aloud, then read each theory point aloud (40-70 words)"
        }},
        {{
            "beat": 3,
            "label": "How It Works",
            "suggested_visual": "diagram | flowchart | equation_walkthrough | step_animation | graph_plot | neural_network_diagram | matrix_grid | timeline | comparison_chart",
            "visual_description": "Concrete description of what the animation should look like",
            "steps": [
                "Step 1: concrete description with specific values or examples",
                "Step 2: concrete description with visible animation change",
                "Step 3: the AHA moment — concrete description"
            ],
            "step_voiceovers": [
                "Step 1 narration (12-22 words)",
                "Step 2 narration (12-22 words)",
                "Step 3 AHA narration (12-22 words)"
            ],
            "aha_step_index": 2,
            "voiceover": "Overall narration for this segment (used if step_voiceovers is unavailable)"
        }},
        {{
            "beat": 4,
            "label": "Need / Use Case",
            "need": "Why this topic exists — what problem it solves (1 sentence)",
            "use_cases": [
                "Concrete use case 1 with specific example",
                "Concrete use case 2 with specific example",
                "Concrete use case 3 with specific example"
            ],
            "on_screen_text": "Need statement + use cases shown on screen (≤50 words)",
            "voiceover": "Read aloud: why it's needed, then each use case (30-50 words)"
        }},
        {{
            "beat": 5,
            "label": "Summary",
            "takeaway": "5-15 word memorable statement (NOT a repeat of the definition or need)",
            "on_screen_text": "Exact text for the summary banner (≤15 words)",
            "voiceover": "Read the takeaway with conviction (10-20 words)"
        }}
    ]
}}

QUALITY CHECKLIST (self-verify before returning):
- Beat 1 is ONLY the topic name — no definition, no theory, no use case.
- Beat 2 (theory) has NO use cases and NO step-by-step mechanism — those are Beats 4 and 3.
- Beat 2 voiceover READS ALOUD all on-screen text. No silent text allowed.
- Beat 3 (working) has NO re-definition and NO use cases — purely visual mechanism.
- Beat 3 suggested_visual is specific and justified for this topic.
- Beat 4 (need/use case) does NOT re-define the topic — only says WHY it matters and WHERE it's used.
- Beat 4 voiceover READS ALOUD all on-screen text. No silent text allowed.
- Beat 5 takeaway is NOT the same sentence as Beat 2 definition or Beat 4 need.
- Beat 5 voiceover MATCHES the on-screen takeaway text.
- recommended_duration: 45-55 for beginner, 55-70 for intermediate, 65-85 for advanced.

Return ONLY the JSON. No explanation before or after."""


STRUCTURED_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "topic": {"type": "STRING"},
        "difficulty_level": {"type": "STRING"},
        "recommended_duration": {"type": "INTEGER"},
        "visual_metaphor": {"type": "STRING"},
        "beats": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "beat": {"type": "INTEGER"},
                    "label": {"type": "STRING"},
                    "definition": {"type": "STRING"},
                    "theory_points": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "suggested_visual": {"type": "STRING"},
                    "visual_description": {"type": "STRING"},
                    "steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "step_voiceovers": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "aha_step_index": {"type": "INTEGER"},
                    "need": {"type": "STRING"},
                    "use_cases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "takeaway": {"type": "STRING"},
                    "on_screen_text": {"type": "STRING"},
                    "voiceover": {"type": "STRING"},
                },
                "required": ["beat", "label", "voiceover"],
            },
        },
    },
    "required": ["topic", "difficulty_level", "recommended_duration", "visual_metaphor", "beats"],
}

TEACHER_PROMPT = """You are a world-class teacher AND curriculum designer who explains concepts to students in a simple, clear, and engaging way.

A student has asked about the following topic:
{query}

Your job is to break down this concept so that an animation designer can turn it into a visual educational video that creates REAL understanding — not just a sequence of facts.

Think step-by-step through these dimensions:

── UNDERSTANDING THE LEARNER ──
1. What must the student ALREADY know before this topic makes sense? (prerequisites)
2. How difficult is this concept? (beginner / intermediate / advanced)
3. What is the #1 thing students get WRONG about this topic? (misconception)

── CORE TEACHING ──
4. What is the core idea in ONE sentence that a 14-year-old would understand?
5. What are the key terms the student must learn? (with simple definitions)
6. What is the step-by-step process? (Use CONCRETE values, not abstract descriptions)

── MAKING IT CLICK ──
7. What is a real-world ANALOGY that makes this concept intuitive?
8. What is the single VISUAL METAPHOR the animation should be built around?
   - This should be a concrete image: "a ball rolling downhill", "a tree growing branches", "a spotlight scanning a row of boxes"
   - The entire animation should revolve around this metaphor
9. What is the AHA MOMENT — the single visual scene that makes everything click?
   - Describe what the learner SEES and what they suddenly UNDERSTAND
10. What is the BUILD-UP SEQUENCE — the order to introduce ideas from simplest to most complex?
    - Each item should be ONE concept. The animation will introduce them in this exact order.

── EMPHASIS CONTROL ──
11. What MUST the animation make unmistakably clear? (what_to_emphasize)
12. What should the animation NOT focus on? (what_to_deemphasize — things that are true but distract beginners)
13. What should the student walk away remembering? (takeaway)

── DURATION & COMPLEXITY ──
14. How long should this animation be? Simple concepts: 30-45s. Medium: 45-60s. Complex: 60-90s.
15. How many objects can be on screen at once without overwhelming? (max_simultaneous_objects)

Return ONLY a JSON object in this exact format:
{{
    "topic": "Clear name of the topic",
    "core_idea": "One-sentence explanation that a 14-year-old would get",
    "prerequisites": ["Prerequisite 1", "Prerequisite 2"],
    "difficulty_level": "beginner or intermediate or advanced",
    "key_terms": [
        {{"term": "Term 1", "meaning": "Simple definition"}},
        {{"term": "Term 2", "meaning": "Simple definition"}}
    ],
    "step_by_step": [
        "Step 1: concrete description with specific values",
        "Step 2: concrete description with specific values",
        "Step 3: concrete description with specific values"
    ],
    "analogy": "A relatable real-world analogy",
    "visual_metaphor": "The single concrete visual image the animation should revolve around",
    "aha_moment": "What the learner SEES and suddenly UNDERSTANDS in the key scene",
    "build_up_sequence": ["Simplest concept first", "Next concept", "Most complex concept last"],
    "what_to_emphasize": "The ONE thing the animation must make crystal clear",
    "what_to_deemphasize": "What to skip or minimize so beginners aren't overwhelmed",
    "misconception": "The most common mistake students make",
    "takeaway": "The single most important thing to remember",
    "recommended_duration": 60,
    "visual_complexity": "low or medium or high",
    "max_simultaneous_objects": 8
}}

QUALITY RULES:
- Use specific numbers, values, and examples — never be abstract.
- Speak as if explaining to a curious 14-year-old.
- Keep each step_by_step entry to 1-2 sentences max.
- The analogy should be something from everyday life.
- Generate 3 to 6 step_by_step entries.
- Generate 3 to 6 build_up_sequence entries — ordered from SIMPLEST to MOST COMPLEX.
- The visual_metaphor must be a CONCRETE IMAGE, not an abstract description.
- The aha_moment must describe what the learner SEES on screen and what they UNDERSTAND.
- recommended_duration: 30-45 for simple, 45-60 for medium, 60-90 for advanced.
- max_simultaneous_objects: realistic count — never more than 10 for complex topics.

Return ONLY the JSON. No explanation before or after."""

TEACHER_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "topic": {"type": "STRING"},
        "core_idea": {"type": "STRING"},
        "prerequisites": {"type": "ARRAY", "items": {"type": "STRING"}},
        "difficulty_level": {"type": "STRING"},
        "key_terms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "term": {"type": "STRING"},
                    "meaning": {"type": "STRING"},
                },
                "required": ["term", "meaning"],
            },
        },
        "step_by_step": {"type": "ARRAY", "items": {"type": "STRING"}},
        "analogy": {"type": "STRING"},
        "visual_metaphor": {"type": "STRING"},
        "aha_moment": {"type": "STRING"},
        "build_up_sequence": {"type": "ARRAY", "items": {"type": "STRING"}},
        "what_to_emphasize": {"type": "STRING"},
        "what_to_deemphasize": {"type": "STRING"},
        "misconception": {"type": "STRING"},
        "takeaway": {"type": "STRING"},
        "recommended_duration": {"type": "INTEGER"},
        "visual_complexity": {"type": "STRING"},
        "max_simultaneous_objects": {"type": "INTEGER"},
    },
    "required": [
        "topic",
        "core_idea",
        "prerequisites",
        "difficulty_level",
        "key_terms",
        "step_by_step",
        "analogy",
        "visual_metaphor",
        "aha_moment",
        "build_up_sequence",
        "what_to_emphasize",
        "what_to_deemphasize",
        "misconception",
        "takeaway",
        "recommended_duration",
        "visual_complexity",
        "max_simultaneous_objects",
    ],
}


# ── Duration mapping by difficulty ──────────────────────────────────
_DURATION_BY_DIFFICULTY = {
    "beginner": 40,
    "intermediate": 55,
    "advanced": 75,
}

# ── Duration mapping for structured 5-beat arc ──────────────────────
_STRUCTURED_DURATION_BY_DIFFICULTY = {
    "beginner": 50,
    "intermediate": 65,
    "advanced": 80,
}


def _normalize_explanation(explanation: dict, query: str) -> dict:
    """Ensure the teacher output is structurally safe."""
    safe = dict(explanation or {})

    safe["topic"] = safe.get("topic") or query[:60]
    safe["core_idea"] = safe.get("core_idea") or f"Understanding the fundamentals of {query}"
    safe["prerequisites"] = safe.get("prerequisites") or []
    safe["difficulty_level"] = safe.get("difficulty_level") or "intermediate"
    safe["key_terms"] = safe.get("key_terms") or []
    safe["step_by_step"] = safe.get("step_by_step") or [
        f"Start by understanding what {query} means at its simplest level.",
        "Walk through the main process with a concrete example.",
        "Observe the key moment where the result becomes clear.",
    ]
    safe["analogy"] = safe.get("analogy") or ""
    safe["visual_metaphor"] = safe.get("visual_metaphor") or f"A clear visual showing {query} in action"
    safe["aha_moment"] = safe.get("aha_moment") or f"The moment when {query} clicks visually"
    safe["build_up_sequence"] = safe.get("build_up_sequence") or [
        s.split(": ", 1)[-1] if ": " in s else s
        for s in safe["step_by_step"]
    ]
    safe["what_to_emphasize"] = safe.get("what_to_emphasize") or f"The core mechanism of {query}"
    safe["what_to_deemphasize"] = safe.get("what_to_deemphasize") or "Mathematical derivations and proofs"
    safe["misconception"] = safe.get("misconception") or ""
    safe["takeaway"] = safe.get("takeaway") or f"The key idea behind {query}"
    safe["visual_complexity"] = safe.get("visual_complexity") or "medium"
    safe["max_simultaneous_objects"] = int(safe.get("max_simultaneous_objects") or 8)

    # Clamp recommended_duration to [30, 90] and fallback by difficulty.
    raw_duration = safe.get("recommended_duration")
    if raw_duration and isinstance(raw_duration, (int, float)):
        safe["recommended_duration"] = max(30, min(90, int(raw_duration)))
    else:
        difficulty = safe["difficulty_level"].lower()
        safe["recommended_duration"] = _DURATION_BY_DIFFICULTY.get(difficulty, 55)

    return safe


def _normalize_structured_beats(beats_data: dict, query: str) -> dict:
    """Ensure the structured 5-beat teacher output is safe and complete."""
    safe = dict(beats_data or {})
    safe["topic"] = safe.get("topic") or query[:60]
    safe["difficulty_level"] = safe.get("difficulty_level") or "intermediate"
    safe["visual_metaphor"] = safe.get("visual_metaphor") or f"A visual diagram of {query}"

    raw_duration = safe.get("recommended_duration")
    if raw_duration and isinstance(raw_duration, (int, float)):
        safe["recommended_duration"] = max(40, min(90, int(raw_duration)))
    else:
        level = safe.get("difficulty_level", "intermediate")
        safe["recommended_duration"] = _STRUCTURED_DURATION_BY_DIFFICULTY.get(level, 65)

    beats = safe.get("beats", [])
    if len(beats) < 5:
        # Pad with safe defaults if LLM returned fewer beats
        default_labels = [
            "Topic Name", "What It Is", "How It Works",
            "Need / Use Case", "Summary"
        ]
        while len(beats) < 5:
            idx = len(beats)
            beats.append({
                "beat": idx + 1,
                "label": default_labels[idx],
                "voiceover": f"Learn about {query}.",
            })
    safe["beats"] = beats[:5]

    return safe


def teach_concept(query: str, intent: str = "detailed") -> dict:
    """
    Takes a user's topic and returns a deep, structured educational
    explanation that can guide the animation planner.

    Parameters
    ----------
    query  : The user's input (topic name or detailed instruction).
    intent : ``"bare_topic"`` or ``"simple_explanation"`` selects the
             strict 7-beat arc.
             ``"detailed"`` (default) uses the original comprehensive prompt.
    """
    log.info("Explaining concept: %s (intent=%s)", query, intent)

    if intent in ("bare_topic", "simple_explanation"):
        # ── Structured path: strict 5-beat arc, zero overlap between beats ──
        response = call_llm(
            STRUCTURED_TEACHER_PROMPT.format(query=query),
            max_tokens=4096,
            response_mime_type="application/json",
            response_schema=STRUCTURED_RESPONSE_SCHEMA,
            preferred_model=TEACHER_MODEL,
        )
        try:
            beats_data = json.loads(response.strip())
            beats_data = _normalize_structured_beats(beats_data, query)
            log.info("Structured 5-beat arc: %d beats", len(beats_data.get('beats', [])))
            log.info("Difficulty: %s → %ds", beats_data.get('difficulty_level'), beats_data.get('recommended_duration'))
            log.info("Visual metaphor: %s", beats_data.get('visual_metaphor', '')[:60])
            # Tag the response so planner knows which arc was used
            beats_data["_intent"] = "structured"
            return beats_data
        except Exception as e:
            log.warning("Structured JSON parse failed: %s — falling back to detailed prompt", e)
            # Fall through to detailed prompt below

    # ── Detailed path: original comprehensive prompt ──
    response = call_llm(
        TEACHER_PROMPT.format(query=query),
        max_tokens=4096,
        response_mime_type="application/json",
        response_schema=TEACHER_RESPONSE_SCHEMA,
        preferred_model=TEACHER_MODEL,
    )

    try:
        explanation = json.loads(response.strip())
        explanation = _normalize_explanation(explanation, query)
        explanation["_intent"] = "detailed"
        log.info("Core idea: %s", explanation.get('core_idea', '')[:80])
        log.info("Steps: %d", len(explanation.get('step_by_step', [])))
        log.info("Visual metaphor: %s", explanation.get('visual_metaphor', '')[:60])
        log.info("Difficulty: %s → %ds", explanation.get('difficulty_level', ''), explanation.get('recommended_duration', 45))
        return explanation
    except Exception as e:
        log.warning("JSON parse failed: %s", e)
        result = _normalize_explanation({}, query)
        result["_intent"] = "detailed"
        return result
