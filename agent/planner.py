import json
import os
from agent.llm import call_llm
from agent.topic_hints import get_topic_hints, format_hints_for_prompt

PLANNER_MODEL = os.getenv("GEMINI_PLANNER_MODEL", "gemini-2.5-flash")

# ─────────────────────────────────────────────────────────────────────
# WORKING SEGMENT ADVISOR — LLM pre-call to decide what the visual
# heart of the video (Segment 3) should include.
# Called BEFORE the planner so the planner gets rich visual guidance.
# ─────────────────────────────────────────────────────────────────────

_WORKING_ADVISOR_PROMPT = """You are an expert educational animation advisor.

A student wants to learn about: "{query}"

The teacher has already prepared a working/mechanism beat with these steps:
--- TEACHER'S WORKING BEAT ---
Suggested visual: {suggested_visual}
Visual description: {visual_description}
Steps:
{steps_text}
--- END ---

Your job is to advise the animation planner on EXACTLY what the working segment
(the visual heart of the video) should include. This is the most important segment.

Think carefully about:
1. What TYPE of visual best explains this mechanism? (diagram, flowchart, equation walkthrough, etc.)
2. What specific Manim objects and animations should be used?
3. How should the animation be structured step-by-step?
4. What makes the AHA moment visually powerful?
5. What common visualization mistakes should be avoided?

Return ONLY a JSON object:
{{
    "visual_type": "The best visual type for this topic (diagram/flowchart/equation_walkthrough/step_animation/graph_plot/neural_network_diagram/matrix_grid/timeline)",
    "why_this_visual": "1-2 sentences explaining why this visual type is the best choice",
    "manim_objects": ["List of specific Manim classes to use: e.g. Axes, Arrow, Circle, RoundedRectangle, MathTex, Text, etc."],
    "animation_steps": [
        "Step 1: Exact description of what to create and animate",
        "Step 2: Exact description of next visual change",
        "Step 3: Exact description of AHA moment visual"
    ],
    "layout": "How to arrange elements on screen (e.g. split-screen, centered, flow left-to-right)",
    "color_scheme": "What colors to use and what they represent",
    "aha_visual": "Exactly what the viewer SEES during the aha moment that makes it click",
    "avoid": ["What NOT to do when animating this topic"],
    "key_formulas": ["Any formulas to show using MathTex (LaTeX format), empty if none"],
    "max_objects_on_screen": 8
}}

RULES:
- Be SPECIFIC to this exact topic — not generic advice.
- manim_objects should reference real Manim classes.
- animation_steps must describe VISIBLE changes, not abstract concepts.
- Return ONLY the JSON. No explanation."""

_WORKING_ADVISOR_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "visual_type": {"type": "STRING"},
        "why_this_visual": {"type": "STRING"},
        "manim_objects": {"type": "ARRAY", "items": {"type": "STRING"}},
        "animation_steps": {"type": "ARRAY", "items": {"type": "STRING"}},
        "layout": {"type": "STRING"},
        "color_scheme": {"type": "STRING"},
        "aha_visual": {"type": "STRING"},
        "avoid": {"type": "ARRAY", "items": {"type": "STRING"}},
        "key_formulas": {"type": "ARRAY", "items": {"type": "STRING"}},
        "max_objects_on_screen": {"type": "INTEGER"},
    },
    "required": [
        "visual_type", "why_this_visual", "manim_objects",
        "animation_steps", "layout", "color_scheme",
        "aha_visual", "avoid", "max_objects_on_screen",
    ],
}


def generate_working_advice(query: str, teacher_working_beat: dict) -> dict:
    """Ask the LLM what the working segment should include before planning.

    This is the dedicated pre-call that makes Segment 3 (the visual heart)
    as rich and well-planned as possible.

    Returns a dict of visual recommendations, or a safe fallback on failure.
    """
    suggested_visual = teacher_working_beat.get("suggested_visual", "diagram")
    visual_description = teacher_working_beat.get("visual_description", "")
    steps = teacher_working_beat.get("steps", [])
    steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)) if steps else "  (no steps provided)"

    try:
        print(f"[WorkingAdvisor] Asking LLM what to include for working segment: {query[:50]}")
        response = call_llm(
            _WORKING_ADVISOR_PROMPT.format(
                query=query,
                suggested_visual=suggested_visual,
                visual_description=visual_description,
                steps_text=steps_text,
            ),
            max_tokens=2048,
            response_mime_type="application/json",
            response_schema=_WORKING_ADVISOR_SCHEMA,
            preferred_model=PLANNER_MODEL,
        )
        advice = json.loads(response.strip())
        print(f"[WorkingAdvisor] Visual type: {advice.get('visual_type')}, "
              f"steps: {len(advice.get('animation_steps', []))}, "
              f"manim objects: {len(advice.get('manim_objects', []))}")
        return advice
    except Exception as e:
        print(f"[WorkingAdvisor] Failed: {e} — using teacher's beat directly")
        return {
            "visual_type": suggested_visual,
            "why_this_visual": "Fallback — using teacher's recommendation",
            "manim_objects": [],
            "animation_steps": steps,
            "layout": "Split screen: visual LEFT 60%, key text RIGHT 40%",
            "color_scheme": "Standard educational colors",
            "aha_visual": "",
            "avoid": [],
            "key_formulas": [],
            "max_objects_on_screen": 8,
        }


def _format_working_advice(advice: dict) -> str:
    """Format working advisor output into readable text for the planner prompt."""
    if not advice:
        return ""

    parts = []
    parts.append(f"Visual Type: {advice.get('visual_type', 'diagram')}")
    parts.append(f"Why: {advice.get('why_this_visual', '')}")
    parts.append(f"Layout: {advice.get('layout', '')}")
    parts.append(f"Color Scheme: {advice.get('color_scheme', '')}")
    parts.append(f"Max Objects on Screen: {advice.get('max_objects_on_screen', 8)}")

    manim_objs = advice.get("manim_objects", [])
    if manim_objs:
        parts.append(f"Manim Objects to Use: {', '.join(manim_objs)}")

    anim_steps = advice.get("animation_steps", [])
    if anim_steps:
        steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(anim_steps))
        parts.append(f"Recommended Animation Steps:\n{steps_str}")

    aha = advice.get("aha_visual", "")
    if aha:
        parts.append(f"AHA Moment Visual: {aha}")

    avoid = advice.get("avoid", [])
    if avoid:
        avoid_str = "\n".join(f"  - {a}" for a in avoid)
        parts.append(f"AVOID:\n{avoid_str}")

    formulas = advice.get("key_formulas", [])
    if formulas:
        parts.append(f"Key Formulas (MathTex): {', '.join(formulas)}")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────
# STRUCTURED 5-SEGMENT PLANNER PROMPT
# Used when query intent is "bare_topic" or "simple_explanation".
# Maps the teacher's 5-beat arc directly to 5 video segments.
# Enforces zero content overlap between segments.
# ─────────────────────────────────────────────────────────────────────

STRUCTURED_PLANNER_PROMPT = """You are an expert educational animation planner.
A student wants to learn about \"{query}\" and needs a structured educational video.

A curriculum expert has already broken this into a 5-beat teaching arc.
Your job is to convert each beat into a video segment plan.

--- TEACHER'S 5-BEAT ARC ---
{teacher_beats}
--- END BEATS ---

{topic_hints_block}

--- WORKING SEGMENT ADVISOR RECOMMENDATIONS ---
{working_advice_block}
--- END ADVISOR ---

═══ SEGMENT MAPPING RULES (STRICT — NO EXCEPTIONS) ═══

SEGMENT 1 — TOPIC NAME (maps to Beat 1: just the topic name)
- title_text: the topic name (LARGE, prominent — this is the FIRST thing viewers see).
- voiceover: Short intro like "Today we learn about [topic]." (5-12 words).
- NO definition. NO theory. NO use case. JUST the name and a one-line intro.
- Duration: 5-8 seconds.

SEGMENT 2 — THEORY (maps to Beat 2: what it IS + core theory)
- definition: The clear one-sentence definition from Beat 2.
- theory_points: The theory_points list from Beat 2 (3-5 points).
- Each point must be shown as on-screen text AND spoken by voiceover.
- voiceover: MUST read the definition aloud, then read each theory point aloud.
  ⚠️ VOICE-SCREEN RULE: Every word on screen is spoken. No silent text.
- This is TEXT-ONLY. No diagrams, no animation here.
- DO NOT include use cases or "why it's needed" — that's Segment 4.
- DO NOT include how-it-works steps — that's Segment 3.
- Duration: 10-15 seconds.

SEGMENT 3 — WORKING (maps to Beat 3: mechanism / core animation)
- This is the HEART of the video — the visual demonstration.
- Use the Working Advisor's recommendations for visual type, Manim objects, and layout.
- Each step from Beat 3 becomes one animation sub-step.
- Include the AHA MOMENT at aha_step_index.
- Split-screen: LEFT 60% for animation, RIGHT 40% for key text.
- Use each step's voiceover from step_voiceovers.
- The visual should be as RICH and DETAILED as possible.
- Follow the advisor's color scheme and layout recommendations.
- DO NOT repeat theory from Segment 2.
- DO NOT include use cases from Segment 4.
- Duration: 20-30 seconds.

SEGMENT 4 — NEED / USE CASE (maps to Beat 4: why it matters)
- need: Why this topic exists / what problem it solves.
- use_cases: 2-4 concrete real-world applications.
- voiceover: MUST read the need statement and each use case aloud.
  ⚠️ VOICE-SCREEN RULE: Every word on screen is spoken. No silent text.
- DO NOT re-define the topic (that was Segment 2).
- DO NOT re-explain HOW it works (that was Segment 3).
- Duration: 8-12 seconds.

SEGMENT 5 — SUMMARY (maps to Beat 5: key takeaway)
- Show the takeaway from Beat 5 inside a green rounded banner.
- voiceover: MUST read the takeaway text aloud.
  ⚠️ VOICE-SCREEN RULE: Voiceover matches the on-screen text.
- Duration: 5-8 seconds. End with self.wait(2.0).

═══ ANTI-OVERLAP RULE (CRITICAL) ═══
Audit the five segments before returning:
→ Segment 1 (topic name) has NO definition, NO theory, NO use case.
→ Segment 2 (theory) has NO use cases and NO mechanism steps.
→ Segment 3 (working) has NO re-definition and NO use cases.
→ Segment 4 (need/use case) does NOT re-explain what it is or how it works.
→ Segment 5 takeaway does NOT word-for-word repeat Segment 2 definition or Segment 4 need.

Return ONLY a JSON object in this exact format:
{{
    "title": "Topic name",
    "visual_style": "one of: diagram, flowchart, neural_network, timeline, array_boxes, bar_chart, graph_plot, equation_walkthrough, step_animation",
    "visual_metaphor": "Teacher's visual metaphor",
    "duration_seconds": 60,
    "segments": [
        {{
            "type": "topic_name",
            "title_text": "Topic Name (large, first thing shown)",
            "voiceover": "Today we learn about [topic]. (5-12 words)",
            "key_text": "[Topic] (2-4 words)",
            "duration_seconds": 6
        }},
        {{
            "type": "theory",
            "definition": "Clear one-sentence definition from Beat 2",
            "theory_points": ["Theory point 1", "Theory point 2", "Theory point 3"],
            "voiceover": "Read definition aloud, then read each theory point aloud (40-70 words)",
            "key_text": "What Is [Topic] (3-5 words)",
            "duration_seconds": 12
        }},
        {{
            "type": "working",
            "steps": ["Step 1 from Beat 3", "Step 2 from Beat 3", "Step 3 AHA from Beat 3"],
            "voiceovers": ["Beat 3 step_voiceover 1", "Beat 3 step_voiceover 2", "Beat 3 step_voiceover 3"],
            "key_texts": ["Step 1 key (3-8 words)", "Step 2 key (3-8 words)", "AHA key (3-8 words)"],
            "aha_step_index": 2,
            "visual_type": "Advisor's recommended visual type",
            "manim_objects": ["List of Manim classes from advisor"],
            "layout": "Advisor's layout recommendation",
            "color_scheme": "Advisor's color scheme",
            "key_formulas": ["Any formulas in LaTeX"],
            "duration_seconds": 25
        }},
        {{
            "type": "need_usecase",
            "need": "Why this topic exists (1 sentence from Beat 4)",
            "use_cases": ["Use case 1", "Use case 2", "Use case 3"],
            "voiceover": "Read need + each use case aloud (30-50 words)",
            "key_text": "Why It Matters (3-5 words)",
            "duration_seconds": 10
        }},
        {{
            "type": "summary",
            "takeaway": "Beat 5 takeaway (5-15 word memorable statement)",
            "voiceover": "Beat 5 voiceover — matches takeaway text",
            "key_text": "Remember: [key phrase]",
            "duration_seconds": 6
        }}
    ],
    "pedagogical_arc": {{
        "topic_intro": "Beat 1 topic name",
        "theory": "Beat 2 definition + theory points summary",
        "mechanism": "Beat 3 working steps summary",
        "need": "Beat 4 need/use case summary",
        "takeaway": "Beat 5 takeaway"
    }},
    "emotional_beats": ["curious", "informed", "following", "aha!", "confident"],
    "opening_scene": "Title (topic name) — just the name, large and prominent",
    "closing_scene": "Green banner with Beat 5 takeaway, then wait 2 seconds",
    "summary": "Beat 5 takeaway"
}}

Return ONLY the JSON. No explanation before or after."""

PLANNER_PROMPT = """You are an expert educational animation planner who thinks like a film director AND a teacher.
A teacher wants an animation for the following topic.
Your job is to plan exactly what the animation should show from start to end before any code is written.

A concept expert has already explained this topic clearly. Use this explanation as your source of truth:

--- TEACHER EXPLANATION ---
{teacher_explanation}
--- END TEACHER EXPLANATION ---

Using the teacher's breakdown above, plan a visually rich animation that follows the teaching flow.

═══ FREE-FORM SEGMENT STRUCTURE ═══
Create 3 to 7 segments based on what the user asked for.
You decide the number of segments and what each one contains.
Each segment is a distinct visual scene with its own voiceover.

GUIDELINES:
- If the user asked to SOLVE a problem → each segment is a solving step.
  Example for "solve x²+3x+2=0":
    Segment 1: "show_problem" — Display the equation prominently
    Segment 2: "factor" — Show factoring step with animation
    Segment 3: "find_roots" — Solve for x, show both roots
    Segment 4: "verify" — Plug roots back in to verify
    Segment 5: "solution" — Final answer displayed clearly

- If the user asked to EXPLAIN something complex → organize by sub-topics.
  Example for "explain how CNN works step by step":
    Segment 1: "introduction" — What is a CNN
    Segment 2: "convolution_layer" — How convolution works
    Segment 3: "pooling" — How pooling works
    Segment 4: "fully_connected" — Final classification layer
    Segment 5: "summary" — How they work together

- If the user asked to COMPARE → organize by comparison points.
- If the user asked to DRAW/PLOT → show construction step by step.

SEGMENT FORMAT:
Each segment must have:
- "type": a short snake_case label describing this segment (e.g. "show_problem", "step_1_factor", "verify_answer")
- "title_text": heading shown on screen for this segment
- "voiceover": what the narrator says (12-30 words)
- "key_text": a short 3-8 word key phrase shown on screen
- "duration_seconds": how long this segment lasts
- "visual_description": what the animation should show (be specific about Manim objects)

Optionally a segment can also include:
- "steps": array of sub-steps (for complex segments with multiple animations)
- "voiceovers": array of voiceovers for each sub-step
- "key_texts": array of key texts for each sub-step
- "aha_step_index": which sub-step is the AHA moment
- "key_formulas": array of LaTeX formulas to show (e.g. ["x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}"])

RULES:
- The FIRST segment should set up the problem or topic
- The LAST segment should show the final answer/conclusion/summary
- Each segment ends with FULL SCREEN CLEANUP
- Total duration across all segments should be 30-90 seconds
- Be SPECIFIC about visual actions — don't say "show the concept", say "draw a parabola from x=-3 to x=5"
- Voice and screen text must match ~90%

═══ SCREEN LAYOUT ═══
- For segments with both visual and text: split-screen LEFT 60% visual, RIGHT 40% key text.
- For segments that are mostly text: center the content.
- Text should APPEAR before voice starts. Voice reads what's already visible.
- NEVER overlap text on top of the main visual.

═══ CLEANUP BETWEEN SEGMENTS ═══
CRITICAL: Between EVERY segment, include:
`self.play(*[FadeOut(mob) for mob in self.mobjects])`
`self.wait(0.3)`

Return ONLY a JSON object in this exact format:
{{
    "title": "Short animation title",
    "visual_style": "one of: array_boxes, bar_chart, diagram, graph_plot, physics_motion, timeline, flowchart, neural_network, matrix_grid",
    "visual_metaphor": "The central visual theme",
    "duration_seconds": 60,
    "segments": [
        {{
            "type": "segment_type_name",
            "title_text": "Segment heading",
            "visual_description": "What to animate in this segment",
            "voiceover": "What the narrator says (12-30 words)",
            "key_text": "Short key phrase (3-8 words)",
            "duration_seconds": 10
        }}
    ],
    "pedagogical_arc": {{
        "hook": "How the video opens",
        "build": "How complexity layers",
        "aha_moment": "The key visual",
        "reinforce": "How we wrap up",
        "takeaway": "What viewer remembers"
    }},
    "emotional_beats": ["curious", "focused", "following", "aha!", "confident"],
    "opening_scene": "How the video opens",
    "closing_scene": "How it ends",
    "summary": "Final message"
}}

VOICEOVER QUALITY RULES:
- Each line must teach cause-and-effect, logic, or intuition for that step.
- Avoid placeholders like "Introduction", "Now we do step 2".
- Include specific terms, values, or relationships from the teacher's explanation.
- Keep each line concise: 12 to 28 words.

VOICE-SCREEN CONSISTENCY RULES:
- Text appears on screen BEFORE voice starts. Voice reads what viewer already sees.
- Voice and screen text must match ~90%.
- For animation-heavy segments, voice describes the visual; key_text panel must match.

SPATIAL BUDGET RULES:
- Never plan more than 8-10 objects on screen simultaneously.
- Reserve RIGHT ~40% for key text panels — visuals in LEFT ~60%.

VISUAL STYLE GUIDE:
- array_boxes: for search algorithms, data structures, memory
- bar_chart: for sorting algorithms, comparisons, statistics
- diagram: for biology, chemistry, networks, neural networks, processes
- graph_plot: for math functions, ML/AI concepts, data trends, loss curves
- physics_motion: for moving objects, forces, trajectories, waves
- timeline: for history, sequences, processes, lifecycles
- flowchart: for processes, decision trees, how systems work
- neural_network: for neural network architectures, layers, connections
- matrix_grid: for CNNs, image processing, convolution operations

Topic to plan: {query}

Return ONLY the JSON. No explanation before or after."""

PLAN_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "visual_style": {"type": "STRING"},
        "visual_metaphor": {"type": "STRING"},
        "duration_seconds": {"type": "INTEGER"},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING"},
                    "title_text": {"type": "STRING"},
                    "hook": {"type": "STRING"},
                    "definition": {"type": "STRING"},
                    "subtitle": {"type": "STRING"},
                    "theory_points": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "analogy": {"type": "STRING"},
                    "key_terms": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "steps": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "voiceovers": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "voiceover": {"type": "STRING"},
                    "key_texts": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "key_text": {"type": "STRING"},
                    "aha_step_index": {"type": "INTEGER"},
                    "visual_type": {"type": "STRING"},
                    "manim_objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "layout": {"type": "STRING"},
                    "color_scheme": {"type": "STRING"},
                    "key_formulas": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "need": {"type": "STRING"},
                    "use_cases": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "user_query_echo": {"type": "STRING"},
                    "direct_answer": {"type": "STRING"},
                    "takeaway": {"type": "STRING"},
                    "duration_seconds": {"type": "INTEGER"},
                },
                "required": ["type"],
            },
        },
        "pedagogical_arc": {
            "type": "OBJECT",
            "properties": {
                "hook": {"type": "STRING"},
                "topic_intro": {"type": "STRING"},
                "theory": {"type": "STRING"},
                "foundation": {"type": "STRING"},
                "build": {"type": "STRING"},
                "mechanism": {"type": "STRING"},
                "need": {"type": "STRING"},
                "aha_moment": {"type": "STRING"},
                "reinforce": {"type": "STRING"},
                "takeaway": {"type": "STRING"},
            },
        },
        "emotional_beats": {"type": "ARRAY", "items": {"type": "STRING"}},
        "opening_scene": {"type": "STRING"},
        "closing_scene": {"type": "STRING"},
        "summary": {"type": "STRING"},
    },
    "required": [
        "title", "visual_style", "visual_metaphor", "duration_seconds",
        "segments", "pedagogical_arc", "emotional_beats",
        "opening_scene", "closing_scene", "summary",
    ],
}


# ═══════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBILITY: Flatten segments into legacy steps/voiceovers
# so coder.py and app.py can still read the old format.
# ═══════════════════════════════════════════════════════════════════════

def _flatten_segments_to_legacy(plan: dict) -> dict:
    """Convert segment-based plan to legacy flat steps/voiceovers/key_texts.

    This allows the coder and app UI to continue working while we migrate.
    Handles both the 5-segment structured arc and the 5-segment detailed arc.
    """
    segments = plan.get("segments", [])
    if not segments:
        return plan

    flat_steps = []
    flat_voiceovers = []
    flat_key_texts = []
    full_script = []

    for seg in segments:
        seg_type = seg.get("type", "")

        if seg_type == "topic_name":
            step = f"TOPIC NAME: Show topic name '{seg.get('title_text', '')}' — large, prominent, visually striking."
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "theory":
            defn = seg.get("definition", "")
            points = "; ".join(seg.get("theory_points", []))
            step = f"THEORY: Show definition: {defn}. Theory points: {points}."
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "working":
            inner_steps = seg.get("steps", [])
            inner_vos = seg.get("voiceovers", [])
            inner_kts = seg.get("key_texts", [])
            for i, s in enumerate(inner_steps):
                vo = inner_vos[i] if i < len(inner_vos) else ""
                kt = inner_kts[i] if i < len(inner_kts) else ""
                prefix = "AHA MOMENT: " if i == seg.get("aha_step_index", -1) else "WORKING: "
                flat_steps.append(f"{prefix}{s}")
                flat_voiceovers.append(vo)
                flat_key_texts.append(kt)
                full_script.append(f"Beat {len(full_script)+1}: Visual - {prefix}{s}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "need_usecase":
            need = seg.get("need", "")
            use_cases = "; ".join(seg.get("use_cases", []))
            step = f"NEED / USE CASE: Why needed: {need}. Use cases: {use_cases}."
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        # ── Legacy segment types (from the detailed 5-segment path) ──
        elif seg_type == "introduction":
            step = f"TOPIC INTRODUCTION: Show topic name '{seg.get('title_text', '')}', definition: {seg.get('hook', '')}, use/need: {seg.get('subtitle', '')}."
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "theory_analogy":
            theory = "; ".join(seg.get("theory_points", []))
            analogy = seg.get("analogy", "")
            step = f"THEORY & ANALOGY: Show theory: {theory}. Analogy: {analogy}."
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "core_animation":
            inner_steps = seg.get("steps", [])
            inner_vos = seg.get("voiceovers", [])
            inner_kts = seg.get("key_texts", [])
            for i, s in enumerate(inner_steps):
                vo = inner_vos[i] if i < len(inner_vos) else ""
                kt = inner_kts[i] if i < len(inner_kts) else ""
                prefix = "AHA MOMENT: " if i == seg.get("aha_step_index", -1) else "ANIMATION: "
                flat_steps.append(f"{prefix}{s}")
                flat_voiceovers.append(vo)
                flat_key_texts.append(kt)
                full_script.append(f"Beat {len(full_script)+1}: Visual - {prefix}{s}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "user_message":
            echo = seg.get("user_query_echo", "")
            answer = seg.get("direct_answer", "")
            step = f"USER'S MESSAGE: {echo} Answer: {answer}"
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        elif seg_type == "summary":
            takeaway = seg.get("takeaway", "")
            step = f"SUMMARY: Show takeaway banner: {takeaway}"
            vo = seg.get("voiceover", "")
            kt = seg.get("key_text", "")
            flat_steps.append(step)
            flat_voiceovers.append(vo)
            flat_key_texts.append(kt)
            full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

        else:
            # ── Generic handler for any free-form segment type ──
            # This covers LLM-generated types like "show_problem", "factor",
            # "find_roots", "verify_answer", etc.
            title = seg.get("title_text", seg_type.replace("_", " ").title())
            desc = seg.get("visual_description", "")
            # If the segment has inner steps, flatten them
            inner_steps = seg.get("steps", [])
            if inner_steps:
                inner_vos = seg.get("voiceovers", [])
                inner_kts = seg.get("key_texts", [])
                for i, s in enumerate(inner_steps):
                    vo = inner_vos[i] if i < len(inner_vos) else ""
                    kt = inner_kts[i] if i < len(inner_kts) else ""
                    prefix = "AHA MOMENT: " if i == seg.get("aha_step_index", -1) else f"{title}: "
                    flat_steps.append(f"{prefix}{s}")
                    flat_voiceovers.append(vo)
                    flat_key_texts.append(kt)
                    full_script.append(f"Beat {len(full_script)+1}: Visual - {prefix}{s}. Key Text - \"{kt}\". Narration - {vo}")
            else:
                step = f"{title}: {desc}" if desc else f"{title}: Show content for this segment."
                vo = seg.get("voiceover", "")
                kt = seg.get("key_text", "")
                flat_steps.append(step)
                flat_voiceovers.append(vo)
                flat_key_texts.append(kt)
                full_script.append(f"Beat {len(full_script)+1}: Visual - {step}. Key Text - \"{kt}\". Narration - {vo}")

    plan["steps"] = flat_steps
    plan["voiceovers"] = flat_voiceovers
    plan["key_texts"] = flat_key_texts
    plan["full_script"] = full_script
    return plan


def _normalize_plan(plan: dict, query: str) -> dict:
    """Ensure the planner output is safe and structurally usable by the coder."""
    safe = dict(plan or {})

    safe["title"] = safe.get("title") or query[:40]
    safe["visual_style"] = safe.get("visual_style") or "diagram"
    safe["visual_metaphor"] = safe.get("visual_metaphor") or ""
    safe["opening_scene"] = safe.get("opening_scene") or f"Show the topic title about {query}"
    safe["closing_scene"] = safe.get("closing_scene") or "Show a summary takeaway banner, then wait 2 seconds"
    safe["summary"] = safe.get("summary") or f"Understanding {query}"
    safe["duration_seconds"] = int(safe.get("duration_seconds") or 60)

    # Pedagogical arc
    if not isinstance(safe.get("pedagogical_arc"), dict):
        safe["pedagogical_arc"] = {
            "hook": f"Show why {query} matters — what problem does it solve?",
            "foundation": "Introduce the simplest prerequisite concept",
            "build": "Layer complexity one concept at a time",
            "aha_moment": f"The key visual that makes {query} click",
            "reinforce": "Connect back to real-world application",
        }

    safe["emotional_beats"] = safe.get("emotional_beats") or ["curious", "grounded", "following", "aha!", "confident"]

    # ── Determine which segment set to expect ──
    segments = safe.get("segments")
    is_structured = safe.get("_arc_type") == "structured"

    if not isinstance(segments, list) or len(segments) < 3:
        # Build default segments
        if is_structured:
            segments = _build_default_structured_segments(safe, query)
        else:
            segments = _build_default_segments(safe, query)
        safe["segments"] = segments

    # Detect arc type from existing segments
    seg_types = [s.get("type") for s in segments]
    is_structured = "topic_name" in seg_types or "theory" in seg_types or "need_usecase" in seg_types

    if is_structured:
        # ── 5-segment structured arc — enforce canonical order ──
        expected_types = ["topic_name", "theory", "working", "need_usecase", "summary"]
        found_types = [s.get("type") for s in segments]
        for expected_type in expected_types:
            if expected_type not in found_types:
                segments.append(_default_segment(expected_type, query))

        # Re-sort to canonical order
        type_order = {t: i for i, t in enumerate(expected_types)}
        segments.sort(key=lambda s: type_order.get(s.get("type", ""), 99))
    else:
        # ── Free-form arc — accept whatever the LLM produced, just ensure basics ──
        # Ensure every segment has at minimum: type, voiceover, key_text, duration_seconds
        for seg in segments:
            seg.setdefault("type", "step")
            seg.setdefault("voiceover", "")
            seg.setdefault("key_text", "")
            seg.setdefault("duration_seconds", 10)
            seg.setdefault("title_text", seg.get("type", "step").replace("_", " ").title())

    safe["segments"] = segments

    # Validate working/core_animation segment
    for seg in segments:
        if seg.get("type") in ("working", "core_animation"):
            steps = seg.get("steps", [])
            vos = seg.get("voiceovers", [])
            kts = seg.get("key_texts", [])
            if not steps:
                seg["steps"] = [
                    f"Show the main visual for {query} with labeled elements",
                    "Animate the key process with smooth transitions",
                    f"AHA MOMENT: Highlight the key insight of {query}",
                ]
            if len(vos) < len(seg["steps"]):
                for i in range(len(vos), len(seg["steps"])):
                    vos.append(f"Watch how {query} works at this step.")
                seg["voiceovers"] = vos
            elif len(vos) > len(seg["steps"]):
                seg["voiceovers"] = vos[:len(seg["steps"])]
            if len(kts) < len(seg["steps"]):
                for i in range(len(kts), len(seg["steps"])):
                    words = seg["steps"][i].split()[:6]
                    kts.append(" ".join(words))
                seg["key_texts"] = kts
            elif len(kts) > len(seg["steps"]):
                seg["key_texts"] = kts[:len(seg["steps"])]
            if seg.get("aha_step_index") is None:
                seg["aha_step_index"] = max(0, len(seg["steps"]) - 1)

    # Polish voiceovers (remove generic fillers)
    generic_markers = ["introduction", "conclusion", "step", "now we", "visiting node", "this is happening"]
    for seg in segments:
        if seg.get("type") in ("working", "core_animation"):
            polished = []
            for line in seg.get("voiceovers", []):
                text = str(line).strip()
                lower = text.lower()
                if any(marker in lower for marker in generic_markers):
                    text = f"{text}. This is important in {query} because it changes what we should track next."
                elif len(text.split()) < 10:
                    text = f"{text}. This gives context for the next step in {query}."
                polished.append(text)
            seg["voiceovers"] = polished

    # ── Flatten segments to legacy format for backward compat ──
    safe = _flatten_segments_to_legacy(safe)

    return safe


def _default_segment(seg_type: str, query: str) -> dict:
    """Create a safe default segment of the given type."""
    # ── 5-segment structured types ──
    if seg_type == "topic_name":
        return {
            "type": "topic_name",
            "title_text": query[:40],
            "voiceover": f"Today, let's explore {query}.",
            "key_text": query[:20],
            "duration_seconds": 6,
        }
    elif seg_type == "theory":
        return {
            "type": "theory",
            "definition": f"{query} is a technique that helps solve a core problem in its domain.",
            "theory_points": [
                f"The fundamental principle of {query}",
                f"What distinguishes {query} from simpler approaches",
                f"The key mechanism that makes {query} work",
            ],
            "voiceover": f"{query} is a technique that helps solve a core problem. The fundamental principle is what makes it powerful. It is distinguished from simpler approaches by its unique mechanism.",
            "key_text": f"What Is {query[:20]}",
            "duration_seconds": 12,
        }
    elif seg_type == "working":
        return {
            "type": "working",
            "steps": [
                f"Show the setup for {query} with labeled visuals",
                "Animate the main process with smooth transitions",
                f"AHA MOMENT: Highlight the key insight of {query}",
            ],
            "voiceovers": [
                f"First, let us set up the main elements of {query}.",
                "Now watch the process unfold step by step.",
                f"Here is the key insight — notice how everything connects.",
            ],
            "key_texts": ["Setting Up", "The Process", "The Key Insight"],
            "aha_step_index": 2,
            "duration_seconds": 25,
        }
    elif seg_type == "need_usecase":
        return {
            "type": "need_usecase",
            "need": f"{query} exists because traditional approaches have limitations it overcomes.",
            "use_cases": [
                f"Used in real-world applications that require {query}",
                f"Applied in industry and research for solving complex problems",
            ],
            "voiceover": f"{query} exists because traditional approaches have limitations. It is used in real-world applications and applied in industry and research.",
            "key_text": "Why It Matters",
            "duration_seconds": 10,
        }
    # ── Legacy 5-segment types (detailed path) ──
    elif seg_type == "introduction":
        return {
            "type": "introduction",
            "title_text": query[:40],
            "hook": f"{query} is a technique that helps solve a core problem in its domain.",
            "subtitle": f"It is needed because traditional approaches have limitations that {query} overcomes.",
            "voiceover": f"{query} is a technique that helps solve a core problem in its domain. It is needed because traditional approaches have limitations.",
            "key_text": f"What Is {query[:20]}",
            "duration_seconds": 10,
        }
    elif seg_type == "theory_analogy":
        return {
            "type": "theory_analogy",
            "theory_points": [f"The core idea behind {query}"],
            "analogy": f"Think of {query} like a real-world process you already understand.",
            "key_terms": [],
            "voiceover": f"At its core, {query} is about solving a specific problem. Let us break it down.",
            "key_text": "The Core Idea",
            "duration_seconds": 18,
        }
    elif seg_type == "core_animation":
        return {
            "type": "core_animation",
            "steps": [
                f"Show the setup for {query} with labeled visuals",
                "Animate the main process with smooth transitions",
                f"AHA MOMENT: Highlight the key insight of {query}",
            ],
            "voiceovers": [
                f"First, let us set up the main elements of {query}.",
                "Now watch the process unfold step by step.",
                f"Here is the key insight — notice how everything connects.",
            ],
            "key_texts": ["Setting Up", "The Process", "The Key Insight"],
            "aha_step_index": 2,
            "duration_seconds": 25,
        }
    elif seg_type == "user_message":
        return {
            "type": "user_message",
            "user_query_echo": f"You asked about {query}.",
            "direct_answer": f"{query} is the process that achieves the result we just demonstrated.",
            "voiceover": f"So to answer your question: {query} works by following the steps we just saw.",
            "key_text": "Your Answer",
            "duration_seconds": 8,
        }
    elif seg_type == "summary":
        return {
            "type": "summary",
            "takeaway": f"The key idea behind {query} is that small, repeated steps lead to the best result.",
            "voiceover": f"Remember: {query} is all about this key insight. That is the takeaway.",
            "key_text": f"Remember: {query}",
            "duration_seconds": 6,
        }
    return {"type": seg_type, "duration_seconds": 5}


def _build_default_structured_segments(plan: dict, query: str) -> list:
    """Build default 5-segment structure for structured arc."""
    return [
        _default_segment("topic_name", query),
        _default_segment("theory", query),
        _default_segment("working", query),
        _default_segment("need_usecase", query),
        _default_segment("summary", query),
    ]


def _build_default_segments(plan: dict, query: str) -> list:
    """Build default free-form segments when LLM output is missing/broken."""
    segments = []

    # Check for legacy steps
    legacy_steps = plan.get("steps", [])
    legacy_vos = plan.get("voiceovers", [])
    legacy_kts = plan.get("key_texts", [])

    # Seg 1: Setup
    segments.append({
        "type": "setup",
        "title_text": query[:40],
        "visual_description": f"Show the topic/problem: {query}",
        "voiceover": f"Let's look at {query}.",
        "key_text": query[:20],
        "duration_seconds": 8,
    })

    # Seg 2-4: Steps (from legacy or default)
    if legacy_steps:
        for i, step in enumerate(legacy_steps[:5]):
            vo = legacy_vos[i] if i < len(legacy_vos) else f"Now let's work through step {i+1}."
            kt = legacy_kts[i] if i < len(legacy_kts) else f"Step {i+1}"
            segments.append({
                "type": f"step_{i+1}",
                "title_text": kt,
                "visual_description": step,
                "voiceover": vo,
                "key_text": kt,
                "duration_seconds": 12,
            })
    else:
        segments.append({
            "type": "main_content",
            "title_text": f"Working Through {query[:30]}",
            "visual_description": f"Show the main process for {query} with labeled visuals and step-by-step animation.",
            "voiceover": f"Let's work through the main process of {query} step by step.",
            "key_text": "The Process",
            "duration_seconds": 25,
        })

    # Final: Conclusion
    segments.append({
        "type": "conclusion",
        "title_text": "Summary",
        "takeaway": f"The key insight about {query}.",
        "voiceover": f"So that's {query} — remember the key insight.",
        "key_text": f"Remember: {query[:20]}",
        "duration_seconds": 6,
    })

    return segments


def _format_structured_explanation(explanation: dict) -> str:
    """Format the structured 5-beat teacher beats into readable text for the planner prompt."""
    if not explanation or "beats" not in explanation:
        return "No beats available."

    parts = []
    parts.append(f"Topic: {explanation.get('topic', '')}")
    parts.append(f"Difficulty: {explanation.get('difficulty_level', 'intermediate')} | Duration: {explanation.get('recommended_duration', 65)}s")
    parts.append(f"Visual Metaphor: {explanation.get('visual_metaphor', '')}")
    parts.append("")

    for beat in explanation.get("beats", []):
        b = beat.get("beat", "?")
        label = beat.get("label", "")
        parts.append(f"--- BEAT {b}: {label} ---")

        if beat.get("definition"):
            parts.append(f"  Definition: {beat['definition']}")
        if beat.get("on_screen_text"):
            parts.append(f"  On Screen: {beat['on_screen_text']}")
        if beat.get("theory_points"):
            parts.append(f"  Theory Points: {'; '.join(beat['theory_points'])}")
        if beat.get("suggested_visual"):
            parts.append(f"  Suggested Visual: {beat['suggested_visual']}")
        if beat.get("visual_description"):
            parts.append(f"  Visual Description: {beat['visual_description']}")
        if beat.get("steps"):
            for i, step in enumerate(beat["steps"]):
                parts.append(f"  Step {i+1}: {step}")
        if beat.get("step_voiceovers"):
            for i, vo in enumerate(beat["step_voiceovers"]):
                parts.append(f"  Step {i+1} Voiceover: {vo}")
        if beat.get("aha_step_index") is not None:
            parts.append(f"  AHA Step Index: {beat['aha_step_index']}")
        if beat.get("need"):
            parts.append(f"  Need: {beat['need']}")
        if beat.get("use_cases"):
            parts.append(f"  Use Cases: {'; '.join(beat['use_cases'])}")
        if beat.get("takeaway"):
            parts.append(f"  Takeaway: {beat['takeaway']}")
        if beat.get("voiceover"):
            parts.append(f"  Voiceover: {beat['voiceover']}")
        parts.append("")

    return "\n".join(parts)


def normalize_plan(plan: dict, query: str) -> dict:
    """Public wrapper used when plans are edited in the UI before execution."""
    return _normalize_plan(plan, query)


def _format_teacher_explanation(explanation: dict) -> str:
    """Format the teacher's structured explanation into readable text for the planner prompt."""
    if not explanation:
        return "No teacher explanation available."

    parts = []
    parts.append(f"Topic: {explanation.get('topic', '')}")
    parts.append(f"Core Idea: {explanation.get('core_idea', '')}")

    difficulty = explanation.get("difficulty_level", "intermediate")
    recommended_duration = explanation.get("recommended_duration", 55)
    visual_complexity = explanation.get("visual_complexity", "medium")
    max_objects = explanation.get("max_simultaneous_objects", 8)
    parts.append(f"Difficulty: {difficulty} | Recommended Duration: {recommended_duration}s")
    parts.append(f"Visual Complexity: {visual_complexity} (max {max_objects} simultaneous objects on screen)")

    prerequisites = explanation.get("prerequisites", [])
    if prerequisites:
        parts.append(f"Prerequisites (what the learner already knows): {', '.join(prerequisites)}")

    key_terms = explanation.get("key_terms", [])
    if key_terms:
        terms_str = "; ".join(
            f"{t.get('term', '')}: {t.get('meaning', '')}" for t in key_terms
        )
        parts.append(f"Key Terms: {terms_str}")

    steps = explanation.get("step_by_step", [])
    if steps:
        steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        parts.append(f"Step-by-Step Process:\n{steps_str}")

    analogy = explanation.get("analogy", "")
    if analogy:
        parts.append(f"Analogy: {analogy}")

    visual_metaphor = explanation.get("visual_metaphor", "")
    if visual_metaphor:
        parts.append(f"VISUAL METAPHOR (build the animation around this): {visual_metaphor}")

    aha_moment = explanation.get("aha_moment", "")
    if aha_moment:
        parts.append(f"AHA MOMENT (this should be the STAR scene): {aha_moment}")

    build_up = explanation.get("build_up_sequence", [])
    if build_up:
        build_str = " → ".join(build_up)
        parts.append(f"Build-Up Sequence (introduce in this order): {build_str}")

    what_to_emphasize = explanation.get("what_to_emphasize", "")
    if what_to_emphasize:
        parts.append(f"EMPHASIZE: {what_to_emphasize}")

    what_to_deemphasize = explanation.get("what_to_deemphasize", "")
    if what_to_deemphasize:
        parts.append(f"DE-EMPHASIZE (skip or minimize): {what_to_deemphasize}")

    misconception = explanation.get("misconception", "")
    if misconception:
        parts.append(f"Common Misconception: {misconception}")

    takeaway = explanation.get("takeaway", "")
    if takeaway:
        parts.append(f"Key Takeaway: {takeaway}")

    return "\n".join(parts)


def plan_animation(query: str, teacher_explanation: dict = None, intent: str = "detailed") -> dict:
    """
    Takes a teacher's request and an optional concept explanation,
    then returns a structured animation plan before any code is generated.

    Parameters
    ----------
    query               : The user's input.
    teacher_explanation : Structured output from teach_concept().
    intent              : ``"bare_topic"`` or ``"simple_explanation"`` uses the
                          strict 5-segment structured planner.
                          ``"detailed"`` (default) uses the standard 5-segment planner.
    """
    print(f"[Planner] Planning animation for: {query} (intent={intent})")

    # Look up topic-specific visual hints (dynamic + static fallback)
    hints = get_topic_hints(query)
    hints_text = format_hints_for_prompt(hints)
    if hints_text:
        print(f"[Planner] Found topic hints: {(hints or {}).get('visual_style', 'N/A')}")

    # ── Choose prompt template based on intent ──
    _raw_intent = (teacher_explanation or {}).get("_intent", intent)
    use_structured = _raw_intent == "structured" and "beats" in (teacher_explanation or {})

    if use_structured:
        explanation_text = _format_structured_explanation(teacher_explanation)
        topic_hints_block = f"Topic Visual Hints:\n{hints_text}" if hints_text else ""

        # ── Working Segment Advisor: ask LLM what to include in Segment 3 ──
        beats = teacher_explanation.get("beats", [])
        working_beat = next((b for b in beats if b.get("label", "").lower() in ("how it works",)), {})
        working_advice = generate_working_advice(query, working_beat)
        working_advice_block = _format_working_advice(working_advice)

        prompt = STRUCTURED_PLANNER_PROMPT.format(
            query=query,
            teacher_beats=explanation_text,
            topic_hints_block=topic_hints_block,
            working_advice_block=working_advice_block,
        )
        print("[Planner] Using STRUCTURED 5-segment planner prompt (zero-overlap)")
    else:
        explanation_text = _format_teacher_explanation(teacher_explanation)
        if hints_text:
            explanation_text += f"\n\nTopic Visual Hints (based on how this topic is typically animated):\n{hints_text}"
        prompt = PLANNER_PROMPT.format(query=query, teacher_explanation=explanation_text)
        print("[Planner] Using DETAILED 5-segment planner prompt")

    response = call_llm(
        prompt,
        max_tokens=4096,
        response_mime_type="application/json",
        response_schema=PLAN_RESPONSE_SCHEMA,
        preferred_model=PLANNER_MODEL,
    )

    # Parse JSON response with multiple fallback strategies
    try:
        clean = response.strip()
        try:
            plan = json.loads(clean)
            if use_structured:
                plan["_arc_type"] = "structured"
            plan = _normalize_plan(plan, query)
            _print_plan_summary(plan)
            return plan
        except json.JSONDecodeError:
            pass

        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        if "{" in clean and "}" in clean:
            start_idx = clean.find("{")
            end_idx = clean.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                clean = clean[start_idx:end_idx]

        plan = json.loads(clean)
        if use_structured:
            plan["_arc_type"] = "structured"
        plan = _normalize_plan(plan, query)
        _print_plan_summary(plan)
        return plan

    except Exception as e:
        print(f"[Planner] WARNING: JSON parse failed: {e}")
        print(f"[Planner] Retrying planner once with strict JSON repair prompt...")
        try:
            repair_prompt = (
                "Return a valid JSON animation plan only. "
                "Do not include markdown fences or explanations. "
                "Ensure keys: title, visual_style, visual_metaphor, duration_seconds, "
                "segments (array of segments), pedagogical_arc, emotional_beats, "
                "opening_scene, closing_scene, summary. "
                f"Topic: {query}"
            )
            repaired = call_llm(
                repair_prompt,
                max_tokens=4096,
                response_mime_type="application/json",
                response_schema=PLAN_RESPONSE_SCHEMA,
                preferred_model=PLANNER_MODEL,
            )

            repaired_plan = json.loads(repaired.strip())
            if use_structured:
                repaired_plan["_arc_type"] = "structured"
            repaired_plan = _normalize_plan(repaired_plan, query)
            _print_plan_summary(repaired_plan)
            return repaired_plan
        except Exception as second_error:
            print(f"[Planner] WARNING: Retry failed: {second_error}")
            print(f"[Planner] Using fallback plan")

        # Return a safe default plan
        fallback = {
            "title": query[:40],
            "visual_style": "diagram",
            "visual_metaphor": "",
            "duration_seconds": 60,
            "segments": _build_default_structured_segments({}, query) if use_structured else _build_default_segments({}, query),
            "pedagogical_arc": {
                "topic_intro": f"Show {query} as a title",
                "theory": f"Define {query} and explain core theory",
                "mechanism": f"Show how {query} works visually",
                "need": f"Explain why {query} matters",
                "takeaway": f"Key takeaway about {query}",
            },
            "emotional_beats": ["curious", "informed", "following", "aha!", "confident"],
            "opening_scene": f"Show topic title about {query}",
            "closing_scene": "Show summary takeaway banner, then wait 2 seconds",
            "summary": f"Understanding {query}",
        }
        if use_structured:
            fallback["_arc_type"] = "structured"
        return _normalize_plan(fallback, query)


def _print_plan_summary(plan: dict):
    """Print concise plan summary for debugging."""
    print(f"[Planner] Visual style: {plan.get('visual_style')}")
    segments = plan.get("segments", [])
    print(f"[Planner] Segments: {len(segments)} ({', '.join(s.get('type', '?') for s in segments)})")
    print(f"[Planner] Duration: {plan.get('duration_seconds')}s")
    print(f"[Planner] Visual metaphor: {plan.get('visual_metaphor', '')[:60]}")
    # Count total steps across all segments
    total_steps = len(plan.get("steps", []))
    print(f"[Planner] Total flat steps: {total_steps}")
