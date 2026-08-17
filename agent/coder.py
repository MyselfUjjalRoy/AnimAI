import re
import json
import os
from agent.llm import call_llm_detailed
from agent.feedback import get_learned_examples
from agent.validator import validate_video_structure, format_validation_for_prompt
from rag.retriever import retrieve as rag_retrieve

CODER_MODEL = os.getenv("GEMINI_CODER_MODEL", "gemini-2.5-flash")
MAX_GENERATION_ATTEMPTS = 3
MAX_CONTINUATION_ATTEMPTS = 2
TTS_PROVIDER_DEFAULT = os.getenv("TTS_PROVIDER", "azure").strip().lower()
TTS_FALLBACK_DEFAULT = os.getenv("TTS_FALLBACK_PROVIDER", "gtts").strip().lower()
AZURE_TTS_VOICE_DEFAULT = os.getenv("AZURE_TTS_VOICE", "en-IN-NeerjaNeural")
AZURE_TTS_STYLE_DEFAULT = os.getenv("AZURE_TTS_STYLE", "general")

# ---- Known deprecated/broken API patterns → safe replacements ----
_PREVENTIVE_SUBSTITUTIONS = [
    # Axes.get_graph() deprecated in Manim 0.18+
    (r'\.get_graph\(', '.plot('),
    # Axes.get_vertical_line_to_graph() deprecated
    (r'\.get_vertical_line_to_graph\(', '.get_vertical_line('),
    # .to_center() does not exist
    (r'\.to_center\(\)', '.move_to(ORIGIN)'),
    # SurroundingRoundedRectangle does NOT exist — use SurroundingRectangle
    # NOTE: the plain string replacement below in _apply_preventive_fixes is the
    # belt-and-suspenders companion to this regex, in case \b fails.
    (r'\bSurroundingRoundedRectangle\b', 'SurroundingRectangle'),
]


def _strip_prosody_kwargs(code: str) -> str:
    """Remove prosody={{...}} kwargs from ALL self.voiceover() calls.

    GTTSService does NOT accept a ``prosody`` argument — passing it raises:
        TypeError: gTTS.__init__() got an unexpected keyword argument 'prosody'

    This function is called whenever the active TTS provider is gTTS (either
    as the primary provider or as an Azure-fallback after a network failure).
    """
    # Handles both dict literals: prosody={"rate": "-15%"} and prosody={{"rate": "-15%"}}
    # and multiline variants by using a non-greedy match inside the braces.
    return re.sub(
        r',\s*prosody\s*=\s*\{\{[^}]*\}\}|,\s*prosody\s*=\s*\{[^}]*\}',
        '',
        code,
    )


def _build_coder_plan_payload(plan: dict) -> dict:
    """Keep only essential planner fields to reduce prompt size and truncation risk."""
    safe = plan or {}
    keys = [
        "title",
        "visual_style",
        "opening_scene",
        "steps",
        "voiceovers",
        "key_texts",
        "closing_scene",
        "summary",
        "duration_seconds",
        "visual_metaphor",
        "pedagogical_arc",
        "segments",
    ]
    compact = {k: safe.get(k) for k in keys if k in safe}

    # Cap step/voiceover/key_text lengths to avoid massive prompts.
    steps = [str(s)[:220] for s in (compact.get("steps") or [])]
    voiceovers = [str(v)[:220] for v in (compact.get("voiceovers") or [])]
    key_texts = [str(t)[:80] for t in (compact.get("key_texts") or [])]
    compact["steps"] = steps[:10]
    compact["voiceovers"] = voiceovers[:10]
    compact["key_texts"] = key_texts[:10]

    # Trim segments for prompt size
    segments = compact.get("segments", [])
    if segments:
        trimmed_segments = []
        for seg in segments[:5]:
            trimmed = dict(seg)
            # Trim inner steps/voiceovers
            if "steps" in trimmed:
                trimmed["steps"] = [str(s)[:200] for s in trimmed["steps"][:5]]
            if "voiceovers" in trimmed:
                trimmed["voiceovers"] = [str(v)[:200] for v in trimmed["voiceovers"][:5]]
            trimmed_segments.append(trimmed)
        compact["segments"] = trimmed_segments

    return compact


def _build_tts_prompt_context() -> dict:
    """Build prompt fragments for the currently configured TTS strategy."""
    provider = TTS_PROVIDER_DEFAULT or "azure"
    fallback = TTS_FALLBACK_DEFAULT or "gtts"

    if provider == "azure":
        imports_block = (
            "2. Also import VoiceoverScene and speech services:\n"
            "   - `import os`\n"
            "   - `from manim_voiceover import VoiceoverScene`\n"
            "   - `from manim_voiceover.services.azure import AzureService`\n"
            "   - `from manim_voiceover.services.gtts import GTTSService`"
        )
        setup_block = (
            "4. In `construct()`, configure Azure speech FIRST and keep gTTS fallback exactly once before narration:\n"
            "   ```python\n"
            f"   provider = os.getenv(\"TTS_PROVIDER\", \"{provider}\").lower()\n"
            f"   fallback_provider = os.getenv(\"TTS_FALLBACK_PROVIDER\", \"{fallback}\").lower()\n"
            f"   azure_voice = os.getenv(\"AZURE_TTS_VOICE\", \"{AZURE_TTS_VOICE_DEFAULT}\")\n"
            f"   azure_style = os.getenv(\"AZURE_TTS_STYLE\", \"{AZURE_TTS_STYLE_DEFAULT}\")\n"
            "\n"
            "   if provider == \"azure\":\n"
            "       azure_kwargs = {{\"voice\": azure_voice}}\n"
            "       if azure_style and azure_style.lower() != \"none\":\n"
            "           azure_kwargs[\"style\"] = azure_style\n"
            "       try:\n"
            "           self.set_speech_service(AzureService(**azure_kwargs))\n"
            "       except Exception:\n"
            "           if fallback_provider == \"gtts\":\n"
            "               self.set_speech_service(GTTSService(lang=\"en\"))\n"
            "           else:\n"
            "               raise\n"
            "   else:\n"
            "       self.set_speech_service(GTTSService(lang=\"en\"))\n"
            "   ```"
        )

        voiceover_sync_rules = (
            "VOICEOVER SYNC RULES — CRITICAL:\n"
            "- ✅ Prefer `tracker.duration` and `tracker.get_remaining_duration()` for reliable timing across providers.\n"
            "- ✅ Each `with self.voiceover(text=...)` block should manage its own timing.\n"
            "- ⚠️ Avoid bookmark-driven timing when using gTTS fallback (`self.wait_until_bookmark(...)`) because gTTS does not support it.\n"
            "- ✅ If you need compatibility with fallback mode, do NOT use `<bookmark .../>` tags."
        )
    else:
        imports_block = (
            "2. Also import `VoiceoverScene` and `GTTSService`:\n"
            "    - `from manim_voiceover import VoiceoverScene`\n"
            "    - `from manim_voiceover.services.gtts import GTTSService`"
        )
        setup_block = "4. In `construct()`, call `self.set_speech_service(GTTSService(lang=\"en\"))` before any narration"
        voiceover_sync_rules = (
            "VOICEOVER SYNC RULES — CRITICAL:\n"
            "- ❌ NEVER USE `<bookmark mark='...'/>` TAGS IN VOICEOVER TEXT. EVER.\n"
            "  GTTSService does NOT support word-boundary transcription — any call to\n"
            "  `self.wait_until_bookmark()` will crash with:\n"
            "  \"Word boundaries are required for timing with bookmarks.\"\n"
            "- ❌ NEVER call `self.wait_until_bookmark()` — it will always raise an Exception with GTTSService.\n"
            "- ✅ USE `tracker.duration` to match total animation length to speech length.\n"
            "- ✅ USE `tracker.get_remaining_duration()` to fill remaining time in a voiceover.\n"
            "- ✅ Sequence animations using fractions of `tracker.duration`:"
        )

    return {
        "tts_provider": provider,
        "tts_import_rules": imports_block,
        "tts_setup_rules": setup_block,
        "voiceover_sync_rules": voiceover_sync_rules,
    }

CODER_PROMPT = """You are an expert Manim animator creating professional educational animations for AnimAI Studio.

═══ VIDEO SEGMENT STRUCTURE ═══
The plan below contains a `segments` array. Generate code for ALL segments in the plan.
Each segment MUST be separated by FULL SCREEN CLEANUP.

The plan may use EITHER a 5-segment structured arc OR a 5-segment detailed arc:

── 5-SEGMENT STRUCTURED ARC (segment types: topic_name, theory, working, need_usecase, summary) ──

SEGMENT: topic_name (comment: # ═══ SEGMENT 1: TOPIC NAME ═══)
- Show ONLY the topic name — LARGE, prominent, visually striking
- Use Write() animation for the title, add an underline
- Voiceover is a short intro like "Today we learn about X"
- Duration: 5-8 seconds
- End with FULL SCREEN CLEANUP

SEGMENT: theory (comment: # ═══ SEGMENT 2: THEORY ═══)
- Show the definition clearly as a heading
- Show theory points as BULLETED LIST with "•" prefix on each point
- Each bullet point should appear one at a time (LaggedStart)
- Voiceover READS the definition, THEN reads each theory point aloud
- ⚠️ Everything on screen MUST be spoken — no silent text
- This is TEXT-ONLY — no diagrams, no animation
- DO NOT include "why it's needed" here — that's Segment 4
- Duration: 10-15 seconds
- End with FULL SCREEN CLEANUP

SEGMENT: working (comment: # ═══ SEGMENT 3: HOW IT WORKS ═══)
- The main visual demonstration — the HEART of the video
- Follow the plan's inner steps, voiceovers, and key_texts
- Use the plan's visual_type, manim_objects, layout, color_scheme, key_formulas
- Each sub-step must clean up previous elements (FadeOut specific groups)
- Use split-screen: main visual LEFT 60%, key text RIGHT 40%
- Include the AHA MOMENT with special visual treatment (glow, scale, GOLD color)
- Make this as RICH and DETAILED as possible — use real data, formulas, concrete examples
- Duration: 20-30 seconds
- End with FULL SCREEN CLEANUP

SEGMENT: need_usecase (comment: # ═══ SEGMENT 4: NEED / USE CASE ═══)
- Show the "need" sentence as a heading
- Show use cases as BULLETED LIST with "•" prefix
- Voiceover READS the need statement, THEN reads each use case aloud
- ⚠️ Everything on screen MUST be spoken — no silent text
- DO NOT re-define the topic (that was Segment 2)
- Duration: 8-12 seconds
- End with FULL SCREEN CLEANUP

SEGMENT: summary (comment: # ═══ SEGMENT 5: SUMMARY ═══)
- Show a summary banner (RoundedRectangle) with the key takeaway
- Use green color (#6AB04C) for the banner
- End with self.wait(2.0) for screen retention

── FREE-FORM SEGMENTS (any segment types the planner decides) ──

For plans that are NOT using the structured arc, the segments can have ANY type name
(e.g. "show_problem", "factor", "find_roots", "verify_answer", "conclusion").

For each free-form segment:
- Read the segment's "type", "title_text", "voiceover", "key_text", "visual_description"
- Show the title_text as a heading (font_size=36, color=#4FACFE, weight=BOLD)
- If the segment has "visual_description", create Manim visuals matching that description
- If the segment has "key_formulas", render them with MathTex
- If the segment has inner "steps"/"voiceovers"/"key_texts" arrays, handle them like the working/core_animation segment
- If the segment has NO visual_description, treat it as text-only:
  Show title + voiceover text on screen, voice reads it
- End with FULL SCREEN CLEANUP
- Use comment: `# ═══ SEGMENT N: {{TYPE_NAME}} ═══` (uppercase)

For the LAST segment (often "conclusion" or "solution"):
- If it has a "takeaway" field, show it as a summary banner (like the structured summary)
- End with self.wait(2.0)

STRICT CODE RULES:
1. Always use `from manim import *`
{tts_import_rules}
3. Class must extend `VoiceoverScene` and be named `GeneratedScene`
{tts_setup_rules}
5. Add # VOICEOVER: comment and matching `with self.voiceover(text=...)` block before every meaningful animation block
6. Always set: self.camera.background_color = "#0F0F1A"
7. NEVER use filesystem operations.

{voiceover_sync_rules}
- ✅ Sequence animations using fractions of `tracker.duration`:
  ```
  with self.voiceover(text="First we show X, then Y appears.", prosody={{"rate": "-15%"}}) as tracker:
      self.play(FadeIn(x_element), run_time=0.3)
      self.play(Create(decorative_visual), run_time=tracker.get_remaining_duration())
  ```
- Each `with self.voiceover(text=...)` block handles its own timing automatically.

═══ VOICE PACING RULES — CRITICAL ═══
- ❌ NEVER use default speed for voiceover — it is TOO FAST for educational content.
- ✅ ALWAYS add `prosody={{"rate": "-15%"}}` to EVERY `self.voiceover()` call:
  ```python
  with self.voiceover(text="...", prosody={{"rate": "-15%"}}) as tracker:
  ```
- This slows the voice by 15%, making it clearer and more digestible for learners.
- The prosody parameter works with both AzureService and GTTSService.
- For especially dense theory segments, you can go slower: `prosody={{"rate": "-20%"}}`

═══ BULLETED LIST RULES — TEXT SEGMENTS ═══
- For theory points, use cases, and any list of items, ALWAYS prefix with bullet markers:
  ```python
  tp1 = Text("• Core principle one — concrete detail", font_size=24, color=WHITE)
  tp2 = Text("• Core principle two — concrete detail", font_size=24, color=WHITE)
  points = VGroup(tp1, tp2).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
  ```
- Use LaggedStart for staggered bullet appearance:
  ```python
  self.play(LaggedStart(*[FadeIn(p, shift=RIGHT*0.3) for p in points], lag_ratio=0.3), run_time=1.5)
  ```
- Make definition text slightly larger (font_size=30) and theory points smaller (font_size=24)

═══ VOICE-TEXT SYNCHRONIZATION RULES — CRITICAL ═══
The voiceover audio starts IMMEDIATELY when entering `with self.voiceover(...)`.
Therefore: TEXT MUST APPEAR ON SCREEN **BEFORE** THE VOICE STARTS. NO EXCEPTIONS.

TEXT-FIRST-THEN-VOICE PATTERN (mandatory for ALL segments):
1. Build ALL Text/Paragraph objects BEFORE the voiceover block.
2. Show them on screen with `self.play(FadeIn(...), run_time=0.5)` BEFORE the voiceover block.
3. THEN enter `with self.voiceover(...)` — the text is already visible when voice begins.
4. Inside the voiceover block, do animations, transitions, or highlights with tracker timing.

CORRECT PATTERN:
```python
# 1. Build text BEFORE voiceover
theory_title = Text("The cost function measures error", font_size=30, color="#4FACFE", weight=BOLD)
theory_title.to_edge(UP, buff=0.5)
theory_points = Paragraph("It sums the squared differences...", font_size=22, color=WHITE)
theory_points.next_to(theory_title, DOWN, buff=0.5)

# 2. SHOW text on screen FIRST (before voice starts)
self.play(FadeIn(theory_title), FadeIn(theory_points), run_time=0.5)

# 3. THEN voice reads what's already visible
with self.voiceover(text="The cost function measures error. It sums the squared differences.") as tracker:
    self.play(Indicate(theory_title, color=YELLOW), run_time=tracker.duration * 0.3)
    self.play(Create(decorative_visual), run_time=tracker.get_remaining_duration())
```

WRONG PATTERNS:
```python
# ❌ WRONG: Voice starts before text appears
with self.voiceover(text="The cost function...") as tracker:
    self.play(Write(title), run_time=3)  # Voice plays for 3s before text is done

# ❌ WRONG: Text constructed inside voiceover block
with self.voiceover(text="...") as tracker:
    my_text = Text("Details", ...)  # Construction delay inside voiceover
    self.play(FadeIn(my_text))

# ❌ WRONG: FadeIn inside voiceover (still has slight delay)
with self.voiceover(text="The theory is...") as tracker:
    self.play(FadeIn(theory_text), run_time=0.5)  # 0.5s delay, voice already ahead
```

═══ VOICE-SCREEN CONTENT MATCHING RULES — CRITICAL ═══
Rules differ by segment type:

SEGMENT 1 (INTRODUCTION), 2 (THEORY), 4 (USER MESSAGE), 5 (SUMMARY):
- On-screen text MUST be a ~90% LITERAL MATCH of the voiceover narration.
- ✅ If the voice says "The cost function measures how wrong our predictions are",
  the screen text should say: "The cost function measures how wrong our predictions are."
- ✅ If showing bullet points, the voice MUST READ or PARAPHRASE those bullets IN ORDER.
- ✅ The key_text panel must contain the EXACT key phrase the voice emphasizes.
- ❌ NEVER show on-screen text that says something different from the voice.

SEGMENT 3 (CORE ANIMATION):
- The voice CAN describe what is being VISUALIZED without needing matching on-screen text.
- The voice narrates the animation — the visual IS the content.
- BUT: the KEY TEXT PANEL on the right MUST still match what the voice emphasizes.
- ✅ Voice says "Watch the ball roll downhill" while animation shows ball rolling — OK, no text needed.
- ✅ Key text panel shows "Ball Rolls Downhill" — matches the voice emphasis.
- ❌ Key text panel shows "Convergence Theory" while voice says "ball rolling" — mismatch.

RULE OF THUMB: If someone muted the video, the on-screen text should teach the same thing
that someone hearing only the audio would learn.

TEXT / PARAGRAPH API RULES — CRITICAL (Manim v0.20.1):
- ❌ NEVER pass `alignment=` to `Text()`. Text does NOT have this parameter.
  → To left-align multiple Text items, use: `VGroup(t1, t2).arrange(DOWN, aligned_edge=LEFT)`
- ❌ NEVER pass `max_width=` to `Text()`. Text does NOT have this parameter.
  → To limit text width, create the Text first, THEN call: `my_text.scale_to_fit_width(desired_width)`
- ❌ NEVER pass `justify=` to `Text()`. Text does NOT have this parameter.
- ✅ Text() VALID parameters: text, font, font_size, color, weight, slant, line_spacing,
  fill_opacity, stroke_width, tab_width, disable_ligatures
- ✅ For multi-line text with alignment/wrapping, split into separate `Text()` objects in a VGroup.
- ❌ NEVER use `Paragraph(... width=X ...)` — the `width` parameter is NOT a layout constraint
  in Manim v0.20.1; it is either ignored or causes a DeprecationWarning. The text will OVERFLOW.
  → Replace Paragraph with a VGroup of separate Text lines + call `.scale_to_fit_width(X)` on the VGroup:
  ```python
  # WRONG — width= is ignored, text overflows:
  para = Paragraph("Long sentence here.", font_size=20, width=config.frame_width * 0.35)

  # RIGHT — explicit lines + guarded width:
  l1 = Text("Long sentence", font_size=20, color=WHITE)
  l2 = Text("here.", font_size=20, color=WHITE)
  para = VGroup(l1, l2).arrange(DOWN, aligned_edge=LEFT)
  para.scale_to_fit_width(config.frame_width * 0.42)
  ```
- ❌ NEVER call `.get_part_by_text("some string")` on a Paragraph or Text object.
  This crashes: `Mobject.__getattr__.<locals>.getter() takes 1 positional argument but 2 were given`
  → split into separate Text() objects and Indicate() each variable directly.
- ❌ NEVER call `.set_width(X)` on a Paragraph or Text object — deprecated.
  → Use `.scale_to_fit_width(X)` after creation instead.

TEXT LENGTH LIMITS — CRITICAL FOR READABILITY:
- A single `Text()` object MUST NOT exceed 70 characters. Long text is illegible on screen.
  → Split long sentences across multiple `Text()` lines inside a VGroup:
  ```python
  # WRONG — 120+ chars, overflows or renders as illegible line:
  tp = Text("It's a Divide and Conquer algorithm: break a problem into smaller subproblems, solve them, and combine their solutions.", font_size=24)

  # RIGHT — split into readable lines:
  tp_l1 = Text("• Divide and Conquer algorithm:", font_size=24, color=WHITE)
  tp_l2 = Text("  Break → Solve → Combine", font_size=22, color="#E0E0E0")
  tp = VGroup(tp_l1, tp_l2).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
  ```
- A single `voiceover()` block SHOULD NOT exceed 200 characters (~15 seconds of speech).
  For long theory or use-case segments, split into 2-3 voiceover blocks with visual transitions between them.
- Prefer SHORT, PUNCHY bullet points: "• Splits in half recursively" not "• The divide step recursively splits the list in half until each sublist has only one element, which is inherently sorted."

VOICEOVER BLOCK CONSTRUCTION RULES — CRITICAL:
- ❌ NEVER construct Manim objects (Text, Circle, Arrow, etc.) INSIDE a `with self.voiceover(...)` block.
  Objects built inside the block violate the text-first-then-voice pattern and can cause timing errors.
  → Build ALL visual objects BEFORE the `with self.voiceover(...)` block begins.
  ```python
  # WRONG — object built inside voiceover:
  with self.voiceover(text="...") as tracker:
      label = Text("result", ...)          # ← never do this
      self.play(FadeIn(label), ...)

  # RIGHT — build first, then voice:
  label = Text("result", ...)              # ← build before voiceover
  with self.voiceover(text="...") as tracker:
      self.play(FadeIn(label), ...)
  ```
- ❌ NEVER call `self.play(...)` AFTER `tracker.get_remaining_duration()` has already been used
  as a `run_time=` value in the SAME voiceover block. The tracker budget is already exhausted.
  ```python
  # WRONG — Indicate runs after budget is gone:
  with self.voiceover(text="...") as tracker:
      self.play(A.animate.move_to(B), run_time=max(0.05, tracker.get_remaining_duration()))
      self.play(Indicate(C), run_time=0.5)  # ← CRASH or desync

  # RIGHT — merge into the same self.play() call:
  with self.voiceover(text="...") as tracker:
      self.play(
          A.animate.move_to(B),
          Indicate(C),
          run_time=max(0.05, tracker.get_remaining_duration())
      )
  ```

SANDBOX CONSTRAINT RULES — CRITICAL:
- The code runs inside a Docker container with ONLY scene.py mounted. NO external files exist.
- NEVER use ImageMobject, SVGMobject, or any file-loading class.
- NEVER reference external files like .png, .svg, .jpg, .gif, .ico, or any asset path.
- ALWAYS create ALL visuals programmatically using Manim primitives.
- `import os` is allowed only for reading TTS environment variables.
- NEVER use filesystem operations.

DESIGN RULES — ALWAYS FOLLOW:
- Background: "#0F0F1A" (dark navy)
- Use RoundedRectangle for boxes, Circle for nodes
- Use LaggedStart for staggered entrance animations
- Use smooth run_time=0.6 transitions
- Always add title with underline at top in SEGMENT 1
- NEVER use `.to_center()` — it does not exist. Use `.move_to(ORIGIN)` instead.
- NEVER use `Axes.get_graph()` — deprecated in Manim 0.18+. ALWAYS use `Axes.plot()`.
- NEVER use `Axes.get_vertical_line_to_graph()` — use `Axes.get_vertical_line()`.

═══ ANIMATION VARIETY RULES — CRITICAL FOR QUALITY ═══
Using `Indicate()` as the ONLY animation makes the video feel like a slideshow. Viewers
need VISUAL VARIETY to stay engaged. Follow these rules:

❌ NEVER use `Indicate()` as the sole animation for more than 2 consecutive voiceover blocks.
❌ NEVER use `LaggedStart(*[Indicate(elem) for elem in group])` — it is Indicate()-spam on a loop.
  → Replace with a single `Flash()` burst or a color sweep: `LaggedStart(*[elem[0].animate.set_fill(GREEN_E, opacity=1) for elem in group], lag_ratio=0.1)`
✅ Inside voiceover blocks, use AT LEAST 2 different animation types across the full video.

ANIMATION PALETTE — pick the right tool for the job:
- HIGHLIGHTING (draw attention to an element already on screen):
  `Circumscribe(obj, color=YELLOW)` — draws a shape around the object, then fades
  `Flash(obj.get_center(), color=YELLOW, line_length=0.4, num_lines=12)` — quick flash effect
  `Indicate(obj, scale_factor=1.1, color=YELLOW)` — brief scale+color pulse
  `Wiggle(obj)` — playful wiggle motion
- COLOR TRANSITIONS (change an element's appearance in-place):
  `obj.animate.set_color(GREEN)` — smoothly transition color
  `obj.animate.set_fill(GREEN, opacity=0.8)` — change fill color
  `SurroundingRectangle(obj, color=YELLOW, corner_radius=0.1)` — add a highlight box
- REVEALING (introduce a new element on screen):
  `FadeIn(obj, shift=UP*0.3)` — fade in from a direction
  `GrowFromCenter(obj)` — grow from a point
  `DrawBorderThenFill(obj)` — draw outline, then fill in
  `Write(text_obj)` — write text character by character
  `Create(line_or_shape)` — draw a line/shape progressively
- MOVEMENT (reposition elements to tell a story):
  `obj.animate.shift(RIGHT * 2)` — slide an element
  `obj.animate.move_to(target.get_center())` — move to another element
  `obj.animate.scale(1.3)` — grow/shrink in place
- TRANSITIONS (transform one element into another):
  `Transform(old, new)` — morph old shape into new shape
  `ReplacementTransform(old, new)` — replace old with new (old is removed)
  `TransformMatchingShapes(old_group, new_group)` — match by shape geometry

SEGMENT-SPECIFIC ANIMATION GUIDELINES:
- Segment 1 (Topic Name): Use `Write()` + `Create()` for title+underline. Inside voiceover, use `Circumscribe()` or subtle `scale_factor=1.05`.
- Segment 2 (Theory): Use `SurroundingRectangle` to highlight definition, then `.animate.set_color()` to color each bullet as voice reads it. NOT Indicate()-spam.
- Segment 3 (Core Animation): The RICHEST segment — use `.animate.move_to()`, arrows, color transitions,
  `GrowFromCenter()`, `Create(arrow)`. This is the visual heart; it should feel ALIVE with movement.
- Segment 4 (Need/Use Case): Visually DIFFERENT from Segment 2. Use numbered icon circles + text, or
  progressive reveal with `DrawBorderThenFill()`, NOT the same bullet pattern as Segment 2.
  ❌ NEVER do a mass color-flash on all icons at once (e.g., all 3 icons→YELLOW simultaneously).
  ✅ Each icon should flash/grow individually AS the voice mentions it.
- Segment 5 (Summary): Use `FadeIn(scale=1.08)` for the banner, then `Circumscribe()` for text.
  ✅ ALWAYS include 2 text lines in the summary banner (sum_l1 + sum_l2) — single-line banners look thin.

ALGORITHM-SPECIFIC VISUAL PATTERNS:
- SORTING — CRITICAL RULES (Quick Sort, Merge Sort, Bubble Sort, etc.):
  ❌ NEVER just FadeIn a new array after a partition/swap step. That's "array teleportation" and looks broken.
  ✅ ALWAYS animate elements PHYSICALLY MOVING between positions using `.animate.move_to()`:
    ```python
    # RIGHT — elements physically travel to their new positions:
    self.play(
        elem_a.animate.move_to(elem_b.get_center()),
        elem_b.animate.move_to(elem_a.get_center()),
        run_time=0.8
    )
    ```
  ✅ Show a COMPARISON POINTER — an Arrow or Dot scanning left-to-right over the array:
    ```python
    pointer = Arrow(start=arr[0].get_top()+UP*0.3, end=arr[0].get_top()+UP*0.1,
                    color=YELLOW, buff=0, max_tip_length_to_length_ratio=0.4)
    for i, elem in enumerate(arr_elements):
        self.play(pointer.animate.move_to(elem.get_top() + UP*0.4), run_time=0.2)
        if value[i] < pivot_value:
            self.play(elem[0].animate.set_fill(BLUE_D, opacity=0.9), run_time=0.15)
        else:
            self.play(elem[0].animate.set_fill(RED_D, opacity=0.9), run_time=0.15)
    ```
  ✅ Color elements GREEN permanently once they are in their FINAL sorted position.
  ✅ AHA MOMENT (pivot placement) MUST use: Flash() burst + Circumscribe() + color→GREEN_E.

- SEARCHING: Show pointer Arrow moving across array. Highlight active search region vs inactive (dim with `.animate.set_opacity(0.3)`).
- TREES/GRAPHS: Animate traversal by coloring visited nodes (`.animate.set_fill(GREEN, opacity=1)`). Show edge highlighting with `Create(colored_edge)`.
- MATH/EQUATIONS: Use `MathTex` with `.animate.set_color()` to highlight terms. Transform equations step by step.
- NEVER recreate elements from scratch between animation steps — use `Transform()` or `.animate` on existing
  elements. Recreating breaks visual continuity and makes the animation feel disconnected.
- Add step counter or progress indicator for multi-step algorithms (e.g., "Step 2/5" as a small Text in corner).

HALLUCINATED API RULES — CRITICAL (these classes/methods DO NOT EXIST):
- ❌ `SurroundingRoundedRectangle` — DOES NOT EXIST. Crashes with NameError.
  ✅ Use `SurroundingRectangle(mobject, corner_radius=0.1, color=YELLOW)` instead.
- ❌ `Text.set_text("new text")` — Text objects are IMMUTABLE in Manim CE. Calling set_text() silently fails or crashes.
  ✅ Create a NEW `Text("new text", ...)` object and use `Transform(old_text, new_text)` to animate the change.
  ```python
  # WRONG — set_text does not update the visual:
  label.set_text("new value")

  # RIGHT — create new Text + Transform:
  new_label = Text("new value", font_size=28, color=WHITE).move_to(label)
  self.play(Transform(label, new_label), run_time=0.3)
  ```

OPACITY API RULES — CRITICAL:
- ❌ `Line(start, end, color=BLUE, opacity=0.3)` — CRASHES with:
  `TypeError: Mobject.__init__() got an unexpected keyword argument 'opacity'`
- Line, Arrow, DashedLine, Arc, and other geometry mobjects do NOT accept bare `opacity=`.
- ✅ For stroke-based objects (Line, Arrow, DashedLine):
  Use `stroke_opacity=` in the constructor: `Line(start, end, color=BLUE, stroke_opacity=0.3)`
  Or call `.set_opacity(0.3)` after creation.
- ✅ For fill-based objects (Circle, Rectangle, RoundedRectangle):
  Use `fill_opacity=` in the constructor: `Circle(radius=1, fill_opacity=0.5)`
  Or call `.set_fill(opacity=0.5)` after creation.
- NOTE: `.set_fill(opacity=X)` and `.set_stroke(opacity=X)` DO accept `opacity=` — only
  the geometry CONSTRUCTORS reject it.

RECTANGLE vs ROUNDEDRECTANGLE API RULES — CRITICAL:
- `Rectangle()` does NOT accept `corner_radius=`. It will crash with:
  `TypeError: Mobject.__init__() got an unexpected keyword argument 'corner_radius'`
- ❌ WRONG: `Rectangle(width=4, height=2, corner_radius=0.2)` — THIS CRASHES.
- ✅ CORRECT: `RoundedRectangle(corner_radius=0.2, width=4, height=2)` — use this instead.
- For sharp-cornered rectangles: use `Rectangle(width=4, height=2)` — NO corner_radius.
- For rounded rectangles (gates, boxes, banners): ALWAYS use `RoundedRectangle(corner_radius=X, ...)`.

AXIS LABEL API RULES — CRITICAL:
- `get_x_axis_label()` and `get_y_axis_label()` do NOT accept `color=`, `font_size=`, or `weight=`.
  These cause: `TypeError: _get_axis_label() got an unexpected keyword argument 'color'`
- ❌ WRONG: `axes.get_x_axis_label(Tex(r"$x$"), edge=DOWN, buff=0.4, color=WHITE)`
- ✅ CORRECT: `axes.get_x_axis_label(Tex(r"$x$", color=WHITE), edge=DOWN, buff=0.4)`
  → Pass `color` / `font_size` to the `Tex()` / `Text()` object, NOT to get_x_axis_label().
- `get_axis_labels()` only takes `x_label` and `y_label` as strings, MathTex, or Tex objects.

3D SCENE RULES — CRITICAL:
- ❌ NEVER call `self.set_camera_orientation()` — this method only exists on `ThreeDScene`,
  NOT on `VoiceoverScene`. Calling it will crash with AttributeError every time.
- ❌ NEVER use `ThreeDAxes`, `Surface`, `Sphere`, or any 3D mobject — they require `ThreeDScene`
  and are incompatible with `VoiceoverScene`.
- ✅ For 3D-like visuals, simulate depth using 2D Axes, perspective scaling, or layered VGroups.
- ✅ For gradient descent surfaces, use `Axes.plot()` on a 2D parabola instead of a 3D surface.

COLOR RULES — CRITICAL FOR VISIBILITY:
- Colors WITH suffix variants (_A through _E): RED, BLUE, GREEN, YELLOW, GOLD, TEAL, PURPLE, MAROON, GREY/GRAY.
  Example: RED_A, RED_B, RED_C, RED_D, RED_E — all valid.
- Colors WITHOUT suffix variants (base name ONLY): ORANGE, PINK, WHITE, BLACK.
  - ❌ NEVER use ORANGE_A, ORANGE_B, ORANGE_C, ORANGE_D, ORANGE_E — they DO NOT EXIST and crash with NameError.
  - ❌ NEVER use PINK_A, PINK_B, PINK_C, PINK_D, PINK_E — they DO NOT EXIST.
  - ✅ Use ORANGE (no suffix) or a hex string like "#FF8C00" for dark orange.
  - ✅ Use PINK (no suffix) or a hex string like "#FF69B4" for hot pink.
- For ANY other color, ALWAYS use hex strings like "#8B4513" (brown), "#00CED1" (cyan).
- Background is #0F0F1A. Every color MUST have strong contrast against it.
- NEVER use BLACK, DARK_BLUE, "#000000" — they are INVISIBLE on the dark background.
- For text: use WHITE, "#E0E0E0", or bright colors.
- For highlights: use bright saturated colors (YELLOW, "#00FFFF", "#6AB04C").

VOICEOVER TIMING BUDGET RULES — CRITICAL:
- The TOTAL `run_time` of ALL animations inside ONE `with self.voiceover(...) as tracker:` block
  MUST be LESS than `tracker.duration`. Exceeding it makes `get_remaining_duration()` return
  zero or negative, crashing with `ValueError: wait() duration <= 0`.
- ✅ SAFE PATTERN — fractions sum to ≤ 0.85:
  ```python
  with self.voiceover(text="...") as tracker:
      self.play(FadeIn(item1), run_time=tracker.duration * 0.3)
      self.play(Create(item2), run_time=tracker.duration * 0.3)
      self.play(Indicate(item3), run_time=tracker.duration * 0.25)
      # total = 0.85 — safe
  ```
- ❌ CRASH PATTERN — 7 × 0.2 = 1.4 > 1.0:
  ```python
  with self.voiceover(text="...") as tracker:
      self.play(..., run_time=tracker.duration * 0.2)  # x7 = 1.4 CRASH!
  ```
- RULE: Count your fractions. If N animations use `tracker.duration * X`, ensure N × X < 1.0.
- Always guard remaining time: `self.wait(max(0.05, tracker.get_remaining_duration()))`
- Always guard run_time: `run_time=max(0.05, tracker.get_remaining_duration())`
- If you need many animations inside one voiceover, GROUP them or use smaller fractions.

LOOP INSIDE VOICEOVER — CRITICAL TIMING RULES:
- When a `for` loop runs INSIDE a `with self.voiceover(...)` block, the total time is
  ITERATIONS × PER_STEP_TIME. This EASILY exceeds tracker.duration.
- ❌ NEVER use `tracker.duration * X` inside a loop — each iteration burns X of the budget:
  ```python
  # WRONG — 4 iterations × 0.15 = 0.6, plus setup = overflow!
  for i in range(4):
      self.play(..., run_time=tracker.duration * 0.15)  # CRASH!
      self.play(..., run_time=0.1)  # extra fixed time per iteration
  ```
- ✅ SAFE PATTERN — pre-calculate per-step budget from total:
  ```python
  with self.voiceover(text="...") as tracker:
      self.play(FadeIn(setup), run_time=tracker.duration * 0.2)
      n_steps = 4
      per_step = tracker.duration * 0.15 / n_steps  # divide remaining among steps
      for i in range(n_steps):
          self.play(..., run_time=per_step)
      self.wait(max(0.05, tracker.get_remaining_duration()))
  ```
- ❌ NEVER call `self.wait()` or `self.play()` AFTER `get_remaining_duration()` was used as
  a `run_time=` in the SAME voiceover block. The budget is already fully consumed.
  ```python
  # WRONG — self.wait AFTER budget exhausted:
  with self.voiceover(text="...") as tracker:
      self.play(..., run_time=max(0.05, tracker.get_remaining_duration()))
      self.wait(0.5)  # ← DESYNC! Budget already consumed above
  ```

HORIZONTAL PIPELINE LAYOUT — CRITICAL OVERFLOW PREVENTION:
- When placing 3+ elements in a LEFT→RIGHT pipeline (e.g., Input→Filter→FeatureMap→Pooled→FC→Output),
  chaining `.next_to(prev, RIGHT, buff=1.5)` WITHOUT a width guard causes OVERFLOW past x=6.5.
- ❌ NEVER chain `.next_to(RIGHT)` for more than 2 elements without a total-width guard:
  ```python
  # WRONG — element 3+ flies off-screen:
  filter.next_to(input_grid, RIGHT, buff=1.5)
  feature_map.next_to(filter, RIGHT, buff=1.5)    # already at x≈5
  pooled.next_to(feature_map, RIGHT, buff=1.5)     # x≈8 — OFF SCREEN!
  fc_layer.next_to(pooled, RIGHT, buff=1.5)        # x≈11 — INVISIBLE!
  ```
- ✅ SAFE PATTERN — group + arrange + scale_to_fit_width:
  ```python
  # RIGHT — fit entire pipeline within safe zone:
  pipeline = VGroup(input_grid, arrow1, feature_map, arrow2, pooled, arrow3, fc_layer, arrow4, output)
  pipeline.arrange(RIGHT, buff=0.4)
  pipeline.scale_to_fit_width(config.frame_width - 2.0)  # fits in safe zone
  pipeline.move_to(ORIGIN)  # or LEFT * MAIN_X for split-screen
  ```
- ✅ For stepped reveals (show one piece at a time), build the FULL pipeline first with
  `.arrange()` + `.scale_to_fit_width()`, then FadeIn each piece sequentially.
- RULE: If more than 2 elements go left-to-right, ALWAYS use arrange() + scale_to_fit_width().
- VERTICAL stacking: same rule applies for DOWN — use arrange(DOWN) + scale_to_fit_height(config.frame_height - 2.0).
- ABSOLUTE OFFSETS: Never use `obj.get_center() + RIGHT * N` when N > 2.0 — the target is likely outside the safe zone.

═══ SCENE CLEANUP RULES — CRITICAL FOR PREVENTING OVERLAP ═══

BETWEEN SEGMENTS (full cleanup):
```python
self.play(*[FadeOut(mob) for mob in self.mobjects])
self.wait(0.3)
```

WITHIN SEGMENT 3 STEPS (partial cleanup):
```python
# Clean up previous step's elements
if 'step_group' in dir() and step_group is not None:
    self.play(FadeOut(step_group))
    self.wait(0.3)
```

RULES:
- Every segment MUST start with a full cleanup of previous segment's elements.
- Every step within Segment 3 MUST clean up its predecessor's elements.
- NEVER let elements from step N remain visible during step N+1 unless explicitly needed.
- Track elements in named groups (e.g., step1_group, step2_group) for targeted cleanup.

LAYOUT RULES — PREVENT OVERLAP AND CLIPPING:
- Frame is 14.2 units wide × 8 units tall. SAFE ZONE: x=[-6.5, 6.5], y=[-3.5, 3.5].
- Use `.arrange(direction, buff=0.3)` for groups. Never buff < 0.25.
- After building ANY title, long Text, or VGroup, ALWAYS call `.scale_to_fit_width()`:
  - Title: `title.scale_to_fit_width(config.frame_width - 1.4)`
  - Hook/subtitle: `hook.scale_to_fit_width(config.frame_width - 2.2)`
  - Theory/Analogy column: `if group.width > COL_WIDTH: group.scale_to_fit_width(COL_WIDTH)`
  - Banner texts: `banner_text.scale_to_fit_width(banner.get_width() - 1.2)`
  - Answer blocks: `answer_block.scale_to_fit_width(config.frame_width - 2.0)`
- For text labels: font_size=22 or smaller in dense scenes. Never exceed font_size=28 with >3 elements.
- Use `.to_edge()` with `buff=0.5` for elements at screen edges.

SPLIT-SCREEN COORDINATE CONSTANTS — ALWAYS DEFINE AT TOP OF construct():
```python
MAIN_X  = -3.2   # x-centre of left visual area (left 58%)
KEY_X   =  4.6   # x-centre of right key-text panel (right 42%)
KEY_TOP =  1.8   # y for key headline
KEY_MID =  0.8   # y for sub key text

# ── COLUMN BOUNDARIES (Segment 2 split-screen) ──
LEFT_COL_LEFT  = -6.2   # left edge of left column
RIGHT_COL_LEFT =  0.4   # left edge of right column (just past centre divider)
COL_WIDTH      =  5.5   # max width for each column's content
# Left column occupies:  x = [-6.2, -0.7]    (5.5 wide)
# Right column occupies: x = [ 0.4,  5.9]    (5.5 wide)
# Gap at centre:         x = [-0.7,  0.4]    (1.1 wide — safe for divider)
```
- ❌ NEVER create `Rectangle(...)` / `main_visual_area` / `key_text_panel` objects as invisible layout guides.
  They pollute `self.mobjects`, appear in `FadeOut(*self.mobjects)`, and cause ghost overlap in the next segment.
- ✅ Use the coordinate constants directly: `key1.move_to([KEY_X, KEY_TOP, 0])`

SPLIT SCREEN LAYOUT for Segments 2 and 3 — OVERLAP PREVENTION PROTOCOL:

⚠️ THE #1 CAUSE OF OVERLAP: Using `move_to(LEFT * 3.5)` to CENTER a title, then `align_to(title, LEFT)`.
  For a long title (e.g., "Analogy: Collaborative Learning" ~7 units wide), centered at x=3.5,
  the LEFT EDGE is at x=0.0 — which is PAST THE CENTRE DIVIDER. Content aligned to this edge overflows!

Step-by-step MANDATORY positioning order (follow EXACTLY, no deviation):
```python
# ── COLUMN BOUNDARY CONSTANTS (defined at top of construct) ──
LEFT_COL_LEFT  = -6.2
RIGHT_COL_LEFT =  0.4
COL_WIDTH      =  5.5

# ── LEFT COLUMN (Theory) ──────────────────────────────────────────────────────
theory_title = Text("The Key Idea", font_size=30, color="#4FACFE", weight=BOLD)
theory_title.move_to(UP * 3.0)                                      # STEP 1: set Y position
if theory_title.width > COL_WIDTH:                                   # STEP 2: constrain title width
    theory_title.scale_to_fit_width(COL_WIDTH)
theory_title.align_to([LEFT_COL_LEFT, 0, 0], LEFT)                  # STEP 3: anchor LEFT EDGE

tp1 = Text("Point one.", font_size=21, color=WHITE)
tp2 = Text("Point two.", font_size=21, color=WHITE)
theory_pts = VGroup(tp1, tp2).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
theory_pts.next_to(theory_title, DOWN, buff=0.4, aligned_edge=LEFT) # STEP 4: place below title
if theory_pts.width > COL_WIDTH:                                     # STEP 5: constrain width
    theory_pts.scale_to_fit_width(COL_WIDTH)
theory_pts.align_to([LEFT_COL_LEFT, 0, 0], LEFT)                    # STEP 6: anchor to COLUMN EDGE

# ── RIGHT COLUMN (Analogy) ────────────────────────────────────────────────────
analogy_title = Text("Analogy", font_size=26, color="#F9CA24", weight=BOLD)
analogy_title.move_to(UP * 3.0)                                      # STEP 1: set Y position
if analogy_title.width > COL_WIDTH:                                   # STEP 2: constrain title width
    analogy_title.scale_to_fit_width(COL_WIDTH)
analogy_title.align_to([RIGHT_COL_LEFT, 0, 0], LEFT)                 # STEP 3: anchor LEFT EDGE

al1 = Text("Analogy line 1.", font_size=19, color="#E0E0E0")
al2 = Text("Analogy line 2.", font_size=19, color="#E0E0E0")
analogy_pts = VGroup(al1, al2).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
analogy_pts.next_to(analogy_title, DOWN, buff=0.4, aligned_edge=LEFT) # STEP 4
if analogy_pts.width > COL_WIDTH:                                      # STEP 5
    analogy_pts.scale_to_fit_width(COL_WIDTH)
analogy_pts.align_to([RIGHT_COL_LEFT, 0, 0], LEFT)                    # STEP 6: anchor to COLUMN EDGE
```

Column rules:
  - ❌ NEVER use `move_to(LEFT * 3.5)` or `move_to(RIGHT * 3.5)` for titles — this sets the CENTER,
    and a wide title's LEFT EDGE drifts past centre, causing content aligned to it to overlap.
  - ❌ NEVER use `align_to(title, LEFT)` — the title's left edge depends on its rendered width,
    making it unpredictable. Long titles push content into the opposite column.
  - ❌ NEVER use unconditional `scale_to_fit_width()` — it scales SHORT text UP, making it wider
    than necessary. Always guard: `if group.width > COL_WIDTH: group.scale_to_fit_width(COL_WIDTH)`
  - ✅ ALWAYS anchor to FIXED column edge: `.align_to([LEFT_COL_LEFT, 0, 0], LEFT)` or
    `.align_to([RIGHT_COL_LEFT, 0, 0], LEFT)` — this guarantees columns never overlap.
  - ✅ ALWAYS constrain BOTH titles AND content to COL_WIDTH.
  - ✅ Add a `DashedLine(UP*3.5, DOWN*3.5, color=GRAY, stroke_width=1.5).move_to(ORIGIN)` divider.

KEY TEXT PANEL RULES:
- For EVERY voiceover step, show a KEY TEXT using `KEY_X` + `KEY_TOP` coordinates.
- Key text should be SHORT (3-8 words) from plan.key_texts.
- Example: `key1 = Text("Key phrase", font_size=22, color=YELLOW).move_to([KEY_X, KEY_TOP, 0])`
- FadeIn before voiceover block, FadeOut during segment cleanup.
- For AHA MOMENT: font_size=26, color=GOLD, weight=BOLD.

═══ GOLDEN CODE TEMPLATE — FOLLOW THIS STRUCTURE ═══
CRITICAL PATTERN: Text is shown on screen BEFORE the voiceover block starts.
Voice reads what the viewer has ALREADY seen. On-screen text matches voiceover ~90%.
In Segment 3 (animation), voice describes the visual; only key_text panel must match.
```python
from manim import *
import os
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.azure import AzureService
from manim_voiceover.services.gtts import GTTSService

class GeneratedScene(VoiceoverScene):
    def construct(self):
        provider = os.getenv("TTS_PROVIDER", "azure").lower()
        fallback_provider = os.getenv("TTS_FALLBACK_PROVIDER", "gtts").lower()
        azure_voice = os.getenv("AZURE_TTS_VOICE", "en-IN-NeerjaNeural")
        azure_style = os.getenv("AZURE_TTS_STYLE", "general")

        if provider == "azure":
            azure_kwargs = {{"voice": azure_voice}}
            if azure_style and azure_style.lower() != "none":
                azure_kwargs["style"] = azure_style
            try:
                self.set_speech_service(AzureService(**azure_kwargs))
            except Exception:
                if fallback_provider == "gtts":
                    self.set_speech_service(GTTSService(lang="en"))
                else:
                    raise
        else:
            self.set_speech_service(GTTSService(lang="en"))
        self.camera.background_color = "#0F0F1A"

        # ── Split-screen coordinate constants (replaces invisible Rectangle panels) ──
        # ❌ NEVER create Rectangle() as invisible layout guides — they pollute self.mobjects
        MAIN_X  = -3.2   # x-centre of left visual area
        KEY_X   =  4.6   # x-centre of right key-text panel
        KEY_TOP =  1.8   # y for key headline
        KEY_MID =  0.8   # y for sub key text
        # ── Column boundary constants for Segment 2 split-screen (ALWAYS include these) ──
        LEFT_COL_LEFT  = -6.2   # left edge of left column  → content x = [-6.2, -0.7]
        RIGHT_COL_LEFT =  0.4   # left edge of right column → content x = [ 0.4,  5.9]
        COL_WIDTH      =  5.5   # max width for each column's content

        # ═══ SEGMENT 1: TOPIC NAME ═══
        title = Text("Topic Title", font_size=64, color="#4FACFE", weight=BOLD)
        title.to_edge(UP, buff=0.45)
        title.scale_to_fit_width(config.frame_width - 1.4)
        underline = Line(LEFT * 5.5, RIGHT * 5.5, color="#4FACFE", stroke_width=3)
        underline.next_to(title, DOWN, buff=0.2)

        self.play(Write(title), Create(underline), run_time=1.0)

        with self.voiceover(text="Today we learn about Topic Title.", prosody={{"rate": "-15%"}}) as tracker:
            self.play(Indicate(title, scale_factor=1.05), run_time=tracker.duration)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.3)

        # ═══ SEGMENT 2: THEORY ═══
        # DEFINITION heading + BULLETED theory points — all spoken aloud
        # ✅ Use VARIED animations: SurroundingRectangle highlight, color transitions — NOT Indicate()-spam
        theory_title = Text("What Is Topic Title?", font_size=36, color="#4FACFE", weight=BOLD)
        theory_title.to_edge(UP, buff=0.75)
        if theory_title.width > config.frame_width - 2.0:
            theory_title.scale_to_fit_width(config.frame_width - 2.0)

        definition_text = Text(
            "Clear one-sentence definition (max 70 chars).",
            font_size=28, color=WHITE, line_spacing=1.2
        )
        definition_text.next_to(theory_title, DOWN, buff=0.6)
        if definition_text.width > config.frame_width - 2.0:
            definition_text.scale_to_fit_width(config.frame_width - 2.0)

        # ✅ SHORT bullet points — max 70 chars each
        tp1 = Text("• Core principle one — short and specific.", font_size=24, color="#E0E0E0", line_spacing=1.2)
        tp2 = Text("• Core principle two — short and specific.", font_size=24, color="#E0E0E0", line_spacing=1.2)
        tp3 = Text("• Core principle three — what makes it special.", font_size=24, color="#E0E0E0", line_spacing=1.2)

        theory_points = VGroup(tp1, tp2, tp3).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        theory_points.next_to(definition_text, DOWN, buff=0.6)
        if theory_points.width > config.frame_width - 2.0:
            theory_points.scale_to_fit_width(config.frame_width - 2.0)

        # Show text FIRST — before voice starts
        self.play(FadeIn(theory_title), run_time=0.5)
        self.play(FadeIn(definition_text), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(tp, shift=RIGHT*0.3) for tp in theory_points], lag_ratio=0.3), run_time=1.5)

        # ✅ Voice reads what's on screen — use VARIED highlight animations
        def_highlight = SurroundingRectangle(definition_text, color=YELLOW, corner_radius=0.1, buff=0.12)
        with self.voiceover(text="Topic Title is [definition]. [Theory point 1].", prosody={{"rate": "-15%"}}) as tracker:
            self.play(Create(def_highlight), run_time=tracker.duration * 0.3)
            self.play(tp1.animate.set_color(YELLOW), run_time=tracker.duration * 0.25)
            self.play(tp1.animate.set_color("#E0E0E0"), FadeOut(def_highlight), run_time=max(0.05, tracker.get_remaining_duration()))

        with self.voiceover(text="[Theory point 2]. [Theory point 3].", prosody={{"rate": "-15%"}}) as tracker:
            self.play(Circumscribe(tp2, color="#00FFFF"), run_time=tracker.duration * 0.4)
            self.play(Circumscribe(tp3, color="#00FFFF"), run_time=max(0.05, tracker.get_remaining_duration()))

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.3)

        # ═══ SEGMENT 3: HOW IT WORKS (CORE ANIMATION) ═══
        # ❌ NEVER create Rectangle() panels — use MAIN_X / KEY_X constants instead
        # ✅ Build ALL objects BEFORE voiceover blocks
        # ✅ Use RICH animations: GrowFromCenter, .animate.move_to(), Create(Arrow), color transitions
        # ✅ NEVER recreate elements from scratch between steps — transform existing elements
        # This is the HEART of the video — make it feel ALIVE with movement

        # --- Step 1: Build key text + initial visual BEFORE voiceover ---
        step_counter = Text("Step 1/2", font_size=18, color=GREY).to_corner(UR, buff=0.3)
        key1 = Text("Set up the elements", font_size=22, color=YELLOW)
        key1.move_to([KEY_X, KEY_TOP, 0])

        box1 = RoundedRectangle(corner_radius=0.1, width=1.2, height=1.0, color=BLUE_D)
        box1.set_fill(BLUE_D, opacity=0.8).move_to([MAIN_X - 1.5, 0, 0])
        label1 = Text("A", font_size=24, color=WHITE).move_to(box1.get_center())

        box2 = RoundedRectangle(corner_radius=0.1, width=1.2, height=1.0, color=BLUE_D)
        box2.set_fill(BLUE_D, opacity=0.8).move_to([MAIN_X + 1.5, 0, 0])
        label2 = Text("B", font_size=24, color=WHITE).move_to(box2.get_center())

        compare_arrow = Arrow(box1.get_right(), box2.get_left(), color="#00FFFF", buff=0.15)

        self.play(FadeIn(step_counter), FadeIn(key1), run_time=0.3)
        with self.voiceover(text="First we set up two elements side by side and compare them.", prosody={{"rate": "-15%"}}) as tracker:
            self.play(GrowFromCenter(VGroup(box1, label1)), run_time=tracker.duration * 0.25)
            self.play(GrowFromCenter(VGroup(box2, label2)), run_time=tracker.duration * 0.25)
            self.play(Create(compare_arrow), run_time=tracker.duration * 0.2)
            self.play(Circumscribe(VGroup(box1, label1), color=YELLOW), run_time=max(0.05, tracker.get_remaining_duration()))

        step1_group = VGroup(box1, label1, box2, label2, compare_arrow, key1, step_counter)
        self.play(FadeOut(step1_group))
        self.wait(0.3)

        # --- Step 2 (AHA MOMENT): Transform existing elements ---
        # ✅ AHA MOMENT PROTOCOL — use ALL THREE: Circumscribe + Flash + color→GREEN_E
        step_counter2 = Text("Step 2/2", font_size=18, color=GREY).to_corner(UR, buff=0.3)
        key2 = Text("AHA — Key insight!", font_size=26, color=GOLD, weight=BOLD)
        key2.move_to([KEY_X, KEY_TOP, 0])

        result_box = RoundedRectangle(corner_radius=0.15, width=3.0, height=1.2, color=GREEN_D)
        result_box.set_fill(GREEN_D, opacity=0.8).move_to([MAIN_X, 0, 0])
        result_label = Text("Result", font_size=28, color=WHITE, weight=BOLD).move_to(result_box.get_center())
        glow = SurroundingRectangle(result_box, color=GOLD, buff=0.15, corner_radius=0.2)

        self.play(FadeIn(step_counter2), FadeIn(key2), run_time=0.3)
        with self.voiceover(text="And here is the key insight — this is why it works.", prosody={{"rate": "-15%"}}) as tracker:
            self.play(DrawBorderThenFill(result_box), FadeIn(result_label), run_time=tracker.duration * 0.30)
            # AHA MOMENT: Circumscribe + Flash burst simultaneously for maximum impact
            self.play(
                Circumscribe(result_box, color=GOLD, buff=0.1),
                Flash(result_box.get_center(), color=GOLD, line_length=0.4, num_lines=12),
                run_time=tracker.duration * 0.35
            )
            self.play(
                result_box.animate.set_fill(GREEN_E, opacity=1),
                Create(glow),
                run_time=max(0.05, tracker.get_remaining_duration())
            )

        # FULL CLEANUP before Segment 4
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.3)

        # ═══ SEGMENT 4: NEED / USE CASE ═══
        # ✅ Visually DIFFERENT from Segment 2 — use NUMBERED ICONS instead of bullets
        need_title = Text("Why Topic Title Matters", font_size=36, color="#F9CA24", weight=BOLD)
        need_title.to_edge(UP, buff=0.75)
        if need_title.width > config.frame_width - 2.0:
            need_title.scale_to_fit_width(config.frame_width - 2.0)

        need_text = Text(
            "Short need statement (max 70 chars).",
            font_size=28, color=WHITE, line_spacing=1.2
        )
        need_text.next_to(need_title, DOWN, buff=0.6)
        if need_text.width > config.frame_width - 2.0:
            need_text.scale_to_fit_width(config.frame_width - 2.0)

        # ✅ NUMBERED ICON use cases — visually different from Segment 2 bullets
        def make_usecase_row(num, text_str, icon_color):
            icon = Circle(radius=0.22, color=icon_color).set_fill(icon_color, opacity=0.9)
            num_label = Text(str(num), font_size=18, color=WHITE, weight=BOLD).move_to(icon)
            icon_group = VGroup(icon, num_label)
            desc = Text(text_str, font_size=22, color="#E0E0E0")
            row = VGroup(icon_group, desc).arrange(RIGHT, buff=0.3)
            return row

        uc1 = make_usecase_row(1, "Use case 1 — short example.", "#4FACFE")
        uc2 = make_usecase_row(2, "Use case 2 — short example.", "#6AB04C")
        uc3 = make_usecase_row(3, "Use case 3 — short example.", "#F9CA24")

        use_cases = VGroup(uc1, uc2, uc3).arrange(DOWN, aligned_edge=LEFT, buff=0.45)
        use_cases.next_to(need_text, DOWN, buff=0.6)
        if use_cases.width > config.frame_width - 2.0:
            use_cases.scale_to_fit_width(config.frame_width - 2.0)

        # Show title + need text FIRST
        self.play(FadeIn(need_title), run_time=0.5)
        self.play(FadeIn(need_text), run_time=0.5)

        # ✅ Voice reads need, then reveals each use case one at a time
        with self.voiceover(text="Topic Title exists because [need]. First, [use case 1].", prosody={{"rate": "-15%"}}) as tracker:
            need_hl = SurroundingRectangle(need_text, color="#F9CA24", corner_radius=0.1, buff=0.1)
            self.play(Create(need_hl), run_time=tracker.duration * 0.3)
            self.play(FadeOut(need_hl), GrowFromCenter(uc1), run_time=max(0.05, tracker.get_remaining_duration()))

        with self.voiceover(text="Second, [use case 2]. And third, [use case 3].", prosody={{"rate": "-15%"}}) as tracker:
            self.play(GrowFromCenter(uc2), run_time=tracker.duration * 0.35)
            self.play(GrowFromCenter(uc3), run_time=tracker.duration * 0.3)
            self.play(
                uc1[0][0].animate.set_fill(YELLOW, opacity=0.9),
                uc2[0][0].animate.set_fill(YELLOW, opacity=0.9),
                uc3[0][0].animate.set_fill(YELLOW, opacity=0.9),
                run_time=max(0.05, tracker.get_remaining_duration())
            )

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(0.3)

        # ═══ SEGMENT 5: SUMMARY ═══
        # ✅ ALWAYS use 2 lines in the summary banner — single-line banners look thin.
        # sum_l1: the memorable takeaway (5-15 words)
        # sum_l2: supporting detail, formula, or complexity (e.g. MathTex for O(n log n))
        banner = RoundedRectangle(corner_radius=0.25, width=12.5, height=2.8)
        banner.set_fill("#0A2A0A", opacity=1).set_stroke("#6AB04C", width=3)
        sum_l1 = Text("[Key takeaway — punchy, 5-15 words]", font_size=28, color="#6AB04C", weight=BOLD)
        sum_l2 = Text("[Supporting detail, complexity, or formula]", font_size=22, color="#6AB04C")
        summary_block = VGroup(sum_l1, sum_l2).arrange(DOWN, buff=0.22)
        summary_block.move_to(banner)
        summary_block.scale_to_fit_width(banner.get_width() - 1.2)
        self.play(FadeIn(VGroup(banner, summary_block), scale=1.08), run_time=0.5)

        with self.voiceover(text="[Exactly matches sum_l1.] [Exactly matches sum_l2.]", prosody={{"rate": "-15%"}}) as tracker:
            # ✅ Use Circumscribe for line 1 — more visual than Indicate for the final payoff
            self.play(Circumscribe(sum_l1, color=YELLOW), run_time=tracker.duration * 0.48)
            self.play(sum_l2.animate.set_color(YELLOW), run_time=max(0.05, tracker.get_remaining_duration()))

        self.wait(2.0)
```

RELEVANT MANIM API PATTERNS (use these as reference for correct API usage):
{manim_docs}

{feedback_examples}

ANIMATION PLAN (use this as the source of truth):
{plan_json}

PLAN USAGE RULES:
- The plan has `segments` array. Generate code for ALL segments in order.
- The plan may have 5 segments (topic_name, theory, working, need_usecase, summary) — the STRUCTURED arc.
  OR 3-7 segments with any type names — the FREE-FORM arc.
- To detect which: if segment types include "topic_name" or "theory" → structured; otherwise → free-form.
- For STRUCTURED plans, use the golden template above as your guide.
- For FREE-FORM plans, generate code for each segment based on its type, title_text, voiceover, and visual_description.
- For segments with inner "steps"/"voiceovers"/"key_texts" arrays, follow them step by step.
- For "working" segments, also use visual_type, manim_objects, layout, color_scheme, key_formulas.
- If the plan has a `visual_metaphor`, use it as the primary visual element.
- If the plan has `pedagogical_arc.aha_moment`, make that step visually SPECIAL.
- The plan also has legacy flat `steps` and `voiceovers` — use segments first, fall back to flat.
- ⚠️ ALWAYS add `prosody={{"rate": "-15%"}}` to EVERY `self.voiceover()` call — this is MANDATORY.
- Follow the "Remove:" and "Keep:" hints in each step for scene cleanup.

Now generate a complete, stunning, professional Manim animation for:
{query}

Use the plan's segment structure. Generate code for ALL segments with proper cleanup between them.
Make it visually rich with colors, smooth animations, and clear educational value.
Return ONLY the Python code. No explanation."""


REVISION_PROMPT = """You are an expert Manim code editor.
You are given existing working Manim code and user-requested changes.

TASK:
- Apply the requested changes to the existing code.
- Preserve all working parts unless they conflict with the requested changes.
- Keep class name `GeneratedScene` and `VoiceoverScene` usage.
- Keep the existing speech-service setup pattern (Azure-first with optional gTTS fallback).
- Keep or improve existing voiceover blocks.
- Return the FULL updated Python file from imports to end of `construct()`.
- PRESERVE the 5-segment video structure (intro, theory, animation, user message, summary).

QUALITY RULES:
- Ensure no clipping/overlap where possible.
- Keep animations educational and readable.
- Maintain full screen cleanup between segments.

VOICE-TEXT SYNCHRONIZATION RULES — CRITICAL:
- Voice audio starts IMMEDIATELY when entering `with self.voiceover(...)`.
- TEXT MUST APPEAR ON SCREEN **BEFORE** THE VOICE STARTS. NO EXCEPTIONS.
- Build ALL Text/Paragraph objects BEFORE the voiceover block.
- Show them on screen with `self.play(FadeIn(...), run_time=0.5)` BEFORE the voiceover block.
- THEN enter `with self.voiceover(...)` — text is already visible when voice begins.
- NEVER start a voiceover block with slow Write() or Create() for text the voice describes.

VOICE-SCREEN CONTENT MATCHING — CRITICAL:
- For Segments 1, 2, 4, 5: On-screen text MUST be ~90% literal match of voiceover narration.
- For Segment 3 (animation): voice narrates the visual; only key_text panel must match.
- Key text panels must echo the exact key phrase the voice emphasizes.
- NEVER show on-screen text that contradicts what the voice is saying.

SCENE TRANSITION RULES:
- Between segments: self.play(*[FadeOut(mob) for mob in self.mobjects])
- Within steps: FadeOut specific element groups.
- NEVER let elements from step N remain visible during step N+1.

LAYOUT RULES:
- Safe zone: x=[-6.5, 6.5], y=[-3.5, 3.5]. Frame is 14.2 wide x 8 tall.
- Use .arrange(direction, buff=0.3) for groups. Never buff < 0.25.
- After building VGroups with 3+ elements, call .scale_to_fit_width(config.frame_width - 2).
- Split-screen: LEFT 60% for visuals, RIGHT 40% for key text.

RELEVANT MANIM API PATTERNS:
{manim_docs}

TOPIC:
{query}

PLAN JSON:
{plan_json}

USER REQUESTED CHANGES:
{change_request}

EXISTING CODE:
{existing_code}

Return ONLY Python code. No markdown or commentary.
"""


def _likely_truncated_tail(code: str) -> bool:
    stripped = (code or "").rstrip()
    if not stripped:
        return True
    bad_tails = ("=", "(", "[", "{", ",", ".", ":", "\\")
    if stripped.endswith(bad_tails):
        return True
    return False


def _binary_tree_completeness_error(code: str, query: str) -> str:
    """Heuristic guard against half-visible or incomplete binary tree renders."""
    text = (query or "").lower()
    if "binary tree" not in text:
        return ""

    if "Graph(" in code:
        return ""

    node_count = len(re.findall(r"Circle\(|Dot\(|RoundedRectangle\(", code))
    edge_count = len(re.findall(r"Line\(|Arrow\(", code))

    if node_count < 6 or edge_count < 4:
        return "binary tree appears incomplete; ensure full tree nodes and edges are visible"

    return ""


def _validate_scene_cleanup(code: str) -> str:
    """Check that voiceover blocks have proper cleanup between them.

    Returns empty string if cleanup looks adequate, else a short suggestion.
    """
    voiceover_positions = [m.start() for m in re.finditer(r'with\s+self\.voiceover\s*\(', code)]
    if len(voiceover_positions) < 3:
        return ""  # Too short to need extensive cleanup checks

    # Check for full cleanup patterns between voiceover blocks
    full_cleanup_pattern = re.compile(
        r'self\.play\s*\(\s*\*\s*\[\s*FadeOut\s*\(\s*mob\s*\)\s*for\s+mob\s+in\s+self\.mobjects\s*\]'
    )
    full_cleanups = list(full_cleanup_pattern.finditer(code))

    if len(full_cleanups) < 2 and len(voiceover_positions) >= 5:
        return (
            "CLEANUP WARNING: Found fewer than 2 full screen cleanups for a 5+ voiceover video. "
            "Add `self.play(*[FadeOut(mob) for mob in self.mobjects])` between major segments "
            "to prevent element pile-up."
        )

    return ""


def _validate_voiceover_sync(code: str) -> str:
    """Check that voiceover blocks follow the text-first-then-voice pattern.

    Validates:
    1. Text/Paragraph objects should be built BEFORE voiceover blocks, not inside.
    2. The first self.play() inside voiceover blocks should NOT be FadeIn of text
       (text should already be visible BEFORE the voiceover block).
    3. Write/Create of text inside voiceover blocks is a sync violation.

    Returns empty string if OK, else a warning string.
    """
    warnings = []

    # Find all voiceover blocks and the code between them
    vo_pattern = re.compile(
        r'with\s+self\.voiceover\s*\(.*?\)\s*as\s+\w+\s*:(.*?)(?=\n\s*(?:with\s+self\.voiceover|# ═══|self\.play\s*\(\s*\*\s*\[|self\.wait\s*\(\s*\d)|\Z)',
        re.DOTALL,
    )
    matches = list(vo_pattern.finditer(code))

    for idx, m in enumerate(matches):
        block = m.group(1)
        if not block:
            continue

        # Check 1: Text/Paragraph constructed INSIDE voiceover block (all blocks)
        text_construction = re.findall(
            r'=\s*(?:Text|Paragraph|MathTex|Tex)\s*\(', block
        )
        if text_construction:
            warnings.append(
                f"SYNC WARNING: Voiceover block {idx + 1} constructs "
                f"{len(text_construction)} text object(s) INSIDE the block. "
                f"Build text BEFORE the voiceover block and show it on screen first."
            )

        # Check 2: First self.play() uses Write/Create (slow) — text should be pre-shown
        first_play = re.search(r'self\.play\s*\(([^)]*(?:\([^)]*\))*[^)]*)\)', block)
        if first_play:
            play_args = first_play.group(1)
            if re.search(r'\bWrite\s*\(', play_args) or re.search(r'\bCreate\s*\(', play_args):
                if not re.search(r'\bFadeIn\s*\(', play_args):
                    warnings.append(
                        f"SYNC WARNING: Voiceover block {idx + 1} starts with "
                        f"Write/Create animation. Show text BEFORE the voiceover "
                        f"block so it's visible when voice starts."
                    )

    return "; ".join(warnings) if warnings else ""

def _validate_timing_budget(code: str) -> str:
    """Check that voiceover blocks don't overdraw tracker.duration with sequential animations.

    A single `with self.voiceover(...) as tracker:` block that chains multiple
    `run_time=tracker.duration * X` calls will crash when the fractions sum > 1.0,
    because `tracker.get_remaining_duration()` returns 0 or negative.

    Returns empty string if OK, else a warning string.
    """
    warnings = []
    # Regex to find voiceover blocks and extract all tracker.duration * N fractions
    vo_pattern = re.compile(
        r'with\s+self\.voiceover\s*\(.*?\)\s*as\s+\w+\s*:(.*?)(?=\n\s*(?:with\s+self\.voiceover|# ═══|self\.play\s*\(\s*\*\s*\[|self\.wait\s*\(\s*\d)|\Z)',
        re.DOTALL,
    )
    fraction_pattern = re.compile(r'tracker\.duration\s*\*\s*([0-9]*\.?[0-9]+)')

    for idx, m in enumerate(vo_pattern.finditer(code)):
        block = m.group(1)
        fractions = [float(f) for f in fraction_pattern.findall(block)]
        if fractions:
            total = sum(fractions)
            if total > 0.95:
                warnings.append(
                    f"TIMING WARNING: Voiceover block {idx + 1} uses fractions summing to "
                    f"{total:.2f} of tracker.duration (limit is 0.95). "
                    f"Fractions found: {fractions}. "
                    f"Reduce individual fractions so they sum to ≤ 0.85 to leave room for "
                    f"get_remaining_duration() to return a positive value."
                )
    return "; ".join(warnings) if warnings else ""



def _validate_segment2_layout(code: str) -> str:
    """Detect Segment 2 split-screen overlap anti-patterns.

    Checks:
    1. Columns using config.frame_width * 0.5 (too wide).
    2. theory_pts/analogy_pts missing scale_to_fit_width.
    3. scale_to_fit_width called BEFORE next_to (wrong order).
    4. Missing align_to clamp.
    5. NEW: Center-based title anchor (move_to LEFT/RIGHT * 3.x) — #1 overlap cause.
    6. NEW: align_to(title, LEFT) — title-width-dependent, unpredictable.
    7. NEW: Missing LEFT_COL_LEFT / RIGHT_COL_LEFT boundary constants.

    Returns empty string if layout looks safe, else a description of the issue.
    """
    issues = []

    # Find the Segment 2 block
    seg2_match = re.search(
        r'# ═══ SEGMENT 2.*?(?=# ═══ SEGMENT 3|\Z)',
        code,
        re.DOTALL,
    )
    if not seg2_match:
        return ""  # No Segment 2 found — skip

    seg2 = seg2_match.group(0)

    # 1. Columns wider than 0.44 (old pattern)
    wide_col = re.findall(r'scale_to_fit_width\s*\(\s*config\.frame_width\s*\*\s*0\.([5-9]\d*)', seg2)
    if wide_col:
        issues.append(
            f"Column(s) use scale_to_fit_width(config.frame_width * 0.{wide_col[0]}...) — "
            f"must be <= 0.44 to avoid overlap. Use COL_WIDTH=5.5 instead."
        )

    # 2. Theory/analogy points missing scale_to_fit_width guard
    for var in ("theory_pts", "analogy_pts"):
        if var in seg2 and f"{var}.scale_to_fit_width" not in seg2:
            issues.append(
                f"{var} is missing width guard — "
                f"add: if {var}.width > COL_WIDTH: {var}.scale_to_fit_width(COL_WIDTH)"
            )

    # 3. scale_to_fit_width called BEFORE next_to (wrong order)
    for var in ("theory_pts", "analogy_pts"):
        scale_pos = [m.start() for m in re.finditer(rf'{var}\.scale_to_fit_width', seg2)]
        nexto_pos = [m.start() for m in re.finditer(rf'{var}\.next_to', seg2)]
        if scale_pos and nexto_pos and scale_pos[0] < nexto_pos[0]:
            issues.append(
                f"{var}: scale_to_fit_width() called BEFORE next_to() — next_to() repositions "
                f"the group, defeating the width guard. Order must be: "
                f"next_to() then scale_to_fit_width() then align_to()."
            )

    # 4. Missing align_to clamp
    for var in ("theory_pts", "analogy_pts"):
        if var in seg2 and f"{var}.align_to" not in seg2:
            issues.append(
                f"{var} is missing .align_to([FIXED_EDGE, 0, 0], LEFT) — "
                f"without a fixed-edge anchor, text can drift past the centre divider. "
                f"Use: theory_pts.align_to([LEFT_COL_LEFT, 0, 0], LEFT) or "
                f"analogy_pts.align_to([RIGHT_COL_LEFT, 0, 0], LEFT)"
            )

    # 5. Center-based title anchor — the #1 root cause of overlap.
    # move_to(LEFT * 3.5 + UP * Y) centers the title at x=-3.5.
    # A wide title's left edge can be at x=-5.25 → content spans to x=+0.25 (past divider!).
    center_anchor = re.findall(
        r'move_to\s*\(\s*(?:LEFT|RIGHT)\s*\*\s*3\.[0-9]+\s*\+\s*UP',
        seg2
    )
    if center_anchor:
        issues.append(
            f"CRITICAL: {len(center_anchor)} column title(s) use move_to(LEFT/RIGHT * 3.x + UP) "
            f"which CENTERS the title — a long title's left edge drifts past the divider. "
            f"Fix: title.move_to(UP * Y) then title.align_to([LEFT_COL_LEFT, 0, 0], LEFT)"
        )

    # 6. align_to(title, LEFT) — title-width-dependent, unpredictable.
    # For a long analogy title, its left edge can be at x=0 (AT the divider) so content
    # aligned to it overlaps from x=0 to x=5.5.
    title_relative_align = re.findall(
        r'\.align_to\s*\(\s*(?:theory_title|analogy_title)\s*,\s*LEFT\s*\)',
        seg2
    )
    if title_relative_align:
        issues.append(
            f"CRITICAL: {len(title_relative_align)} align_to(title, LEFT) call(s) found — "
            f"anchors content to the TITLE'S left edge, which varies with title length. "
            f"A long right-column title centered at x=3.5 has left edge at x~0 (past divider!). "
            f"Replace with: .align_to([LEFT_COL_LEFT, 0, 0], LEFT) or "
            f".align_to([RIGHT_COL_LEFT, 0, 0], LEFT)"
        )

    # 7. Missing column boundary constants
    if 'theory_title' in seg2 or 'analogy_title' in seg2:
        missing = [c for c in ('LEFT_COL_LEFT', 'RIGHT_COL_LEFT', 'COL_WIDTH')
                   if c not in code]
        if missing:
            issues.append(
                f"Missing column boundary constants: {', '.join(missing)}. "
                f"Define at top of construct(): "
                f"LEFT_COL_LEFT=-6.2, RIGHT_COL_LEFT=0.4, COL_WIDTH=5.5"
            )

    return "; ".join(issues) if issues else ""


# ── Colors that DO NOT have suffix variants in Manim CE 0.20 ──
_INVALID_COLOR_CONSTANTS = set()
for _base in ("ORANGE", "PINK", "WHITE", "BLACK"):
    for _suffix in ("_A", "_B", "_C", "_D", "_E"):
        _INVALID_COLOR_CONSTANTS.add(f"{_base}{_suffix}")

# ── Undefined color name → hex/base replacement map (shared with debugger.py) ──
_COLOR_MAP = {
    "BROWN": '"#8B4513"', "LIGHT_BROWN": '"#CD853F"', "DARK_BROWN": '"#5C3317"',
    "CYAN": '"#00FFFF"', "DARK_GREEN": '"#006400"', "LIGHT_GREEN": '"#90EE90"',
    "DARK_RED": '"#8B0000"', "LIGHT_BLUE": '"#ADD8E6"', "DARK_GREY": '"#555555"',
    "LIGHT_GREY": '"#AAAAAA"', "MAGENTA": '"#FF00FF"', "LIME": '"#00FF00"',
    "NAVY": '"#000080"', "OLIVE": '"#808000"', "BEIGE": '"#F5F5DC"',
    "TURQUOISE": '"#40E0D0"', "VIOLET": '"#EE82EE"', "INDIGO": '"#4B0082"',
    "CORAL": '"#FF7F50"', "SALMON": '"#FA8072"', "TAN": '"#D2B48C"',
    "KHAKI": '"#F0E68C"', "AQUA": '"#00FFFF"', "CRIMSON": '"#DC143C"',
    "IVORY": '"#FFFFF0"', "LAVENDER": '"#E6E6FA"', "SILVER": '"#C0C0C0"',
    "SKYBLUE": '"#87CEEB"',
    # ORANGE has NO suffix variants in Manim CE
    "ORANGE_A": 'ORANGE', "ORANGE_B": 'ORANGE', "ORANGE_C": 'ORANGE',
    "ORANGE_D": 'ORANGE', "ORANGE_E": 'ORANGE',
    # PINK has NO suffix variants
    "PINK_A": 'PINK', "PINK_B": 'PINK', "PINK_C": 'PINK',
    "PINK_D": 'PINK', "PINK_E": 'PINK',
    # WHITE/BLACK have NO suffix variants
    "WHITE_A": 'WHITE', "WHITE_B": 'WHITE', "WHITE_C": 'WHITE',
    "WHITE_D": 'WHITE', "WHITE_E": 'WHITE',
    "BLACK_A": 'WHITE', "BLACK_B": '"#333333"', "BLACK_C": '"#555555"',
    "BLACK_D": '"#777777"', "BLACK_E": '"#999999"',
}



def _validate_color_constants(code: str) -> str:
    """Detect invalid color constants that will cause NameError at runtime.

    Manim only provides _A through _E suffix variants for specific color families:
    RED, BLUE, GREEN, YELLOW, GOLD, TEAL, PURPLE, MAROON, GREY/GRAY.

    Colors like ORANGE, PINK, WHITE, BLACK do NOT have suffix variants.
    Using e.g. ORANGE_E crashes with: NameError: name 'ORANGE_E' is not defined.
    """
    found = []
    for color in _INVALID_COLOR_CONSTANTS:
        if re.search(r'\b' + color + r'\b', code):
            found.append(color)

    if found:
        return (
            f"Invalid color constant(s): {', '.join(sorted(found))}. "
            f"ORANGE, PINK, WHITE, BLACK do NOT have _A through _E suffix variants in Manim. "
            f"Use the base name (e.g. ORANGE) or a hex string instead."
        )
    return ""


def _validate_hallucinated_apis(code: str) -> str:
    """Detect usage of non-existent Manim APIs that will crash at runtime."""
    issues = []

    # SurroundingRoundedRectangle does NOT exist — use SurroundingRectangle
    if re.search(r'\bSurroundingRoundedRectangle\b', code):
        issues.append(
            "SurroundingRoundedRectangle does NOT exist in Manim. "
            "Use SurroundingRectangle(mobject, corner_radius=0.1) instead."
        )

    # Text.set_text() — Text objects are immutable in Manim CE
    if re.search(r'\.\bset_text\s*\(', code):
        issues.append(
            "Text.set_text() does not reliably update visuals in Manim CE. "
            "Create a new Text() and use Transform(old, new) instead."
        )

    # Bare opacity= in geometry constructors (Line, Arrow, Circle, etc.)
    _GEOM_CONSTRUCTORS = r'(?:Line|Arrow|DashedLine|Circle|Dot|Rectangle|RoundedRectangle|Arc|Square|Polygon|Triangle)\('
    for i, line in enumerate(code.split('\n'), 1):
        if re.search(_GEOM_CONSTRUCTORS, line) and re.search(r'\bopacity\s*=', line):
            if 'stroke_opacity' not in line and 'fill_opacity' not in line:
                issues.append(
                    f"Line {i}: bare opacity= in geometry constructor — "
                    f"use stroke_opacity= or fill_opacity= instead. "
                    f"Manim geometry constructors do NOT accept bare opacity=."
                )

    return "; ".join(issues) if issues else ""


def _validate_horizontal_overflow(code: str) -> str:
    """Detect horizontal layout patterns likely to overflow beyond the safe zone.

    Catches:
    - Chained .next_to(..., RIGHT, buff=X) calls (3+ in succession)
    - Large absolute offsets like obj.get_center() + RIGHT * N where N > 2.0
    """
    issues = []

    # Pattern 1: Count .next_to(..., RIGHT, ...) calls in Segment 3
    seg3_match = re.search(
        r'# ═══ SEGMENT 3.*?(?=# ═══ SEGMENT 4|\Z)',
        code,
        re.DOTALL,
    )
    if seg3_match:
        seg3 = seg3_match.group(0)
        right_chains = re.findall(
            r'\.next_to\s*\([^)]*,\s*RIGHT\s*,\s*buff\s*=\s*([\d.]+)',
            seg3
        )
        if len(right_chains) >= 3:
            total_buff = sum(float(b) for b in right_chains)
            issues.append(
                f"{len(right_chains)} chained .next_to(RIGHT) calls found in Segment 3 "
                f"(total buff={total_buff:.1f}). Elements WILL overflow past x=6.5. "
                f"Use VGroup(...).arrange(RIGHT, buff=0.4).scale_to_fit_width(config.frame_width - 2.0) instead."
            )

    # Pattern 2: Large absolute offsets (obj.get_center() + RIGHT * N with N > 2.0)
    large_offsets = re.findall(
        r'\.get_center\s*\(\s*\)\s*\+\s*RIGHT\s*\*\s*([\d.]+)',
        code
    )
    for offset in large_offsets:
        if float(offset) > 2.0:
            issues.append(
                f"Large absolute offset: get_center() + RIGHT * {offset} — "
                f"target is likely outside the safe zone (x > 6.5). "
                f"Use .next_to() with a scale_to_fit_width guard instead."
            )

    return "; ".join(issues) if issues else ""


def _validate_voiceover_loop_timing(code: str) -> str:
    """Detect for-loops inside voiceover blocks that may overflow the timing budget.

    Catches:
    - `tracker.duration * X` inside a for loop body (each iteration re-burns X)
    - self.wait() or self.play() after get_remaining_duration() was already used
    """
    issues = []

    # Find all voiceover blocks
    voiceover_blocks = re.finditer(
        r'with\s+self\.voiceover\s*\([^)]*\)\s*as\s+(\w+)\s*:(.*?)(?=\n    # |\n        # ═══|\Z)',
        code,
        re.DOTALL,
    )
    for block_match in voiceover_blocks:
        tracker_var = block_match.group(1)
        block_body = block_match.group(2)

        # Pattern 1: for loop with tracker.duration inside
        for_loops = re.finditer(
            r'for\s+\w+\s+in\s+range\s*\((\d+)\)(.*?)(?=\n        (?:self\.|#|\Z))',
            block_body,
            re.DOTALL,
        )
        for loop_match in for_loops:
            iterations = int(loop_match.group(1))
            loop_body = loop_match.group(2)
            # Count tracker.duration * X usages inside the loop
            fractions = re.findall(
                rf'{tracker_var}\.duration\s*\*\s*([\d.]+)',
                loop_body
            )
            if fractions:
                total = sum(float(f) for f in fractions) * iterations
                if total > 0.85:
                    issues.append(
                        f"for loop ({iterations} iters) inside voiceover uses "
                        f"tracker.duration * {'+'.join(fractions)} per iteration — "
                        f"total = {total:.2f} × tracker.duration which EXCEEDS budget. "
                        f"Pre-divide: per_step = tracker.duration * X / {iterations}"
                    )

            # Also check for fixed-time animations inside the loop
            fixed_times = re.findall(r'run_time\s*=\s*([\d.]+)', loop_body)
            if fixed_times:
                fixed_total = sum(float(t) for t in fixed_times) * iterations
                if fixed_total > 3.0:  # More than 3 seconds of fixed animation in a loop
                    issues.append(
                        f"for loop ({iterations} iters) has {len(fixed_times)} "
                        f"fixed-time animations totalling {fixed_total:.1f}s — "
                        f"this likely exceeds the voiceover duration."
                    )

        # Pattern 2: self.wait() or self.play() AFTER get_remaining_duration()
        remaining_pos = block_body.rfind('get_remaining_duration()')
        if remaining_pos > 0:
            after_remaining = block_body[remaining_pos:]
            # Check if there's a self.wait( or self.play( AFTER the LAST get_remaining_duration
            # but only if get_remaining_duration was used as a run_time=
            if re.search(r'run_time\s*=\s*max\([^)]*get_remaining_duration', after_remaining[:80]):
                trailing = after_remaining[80:]
                if re.search(r'self\.(wait|play)\s*\(', trailing):
                    issues.append(
                        "self.wait() or self.play() found AFTER get_remaining_duration() was used "
                        "as a run_time — budget is already consumed, this causes audio desync."
                    )

    return "; ".join(issues) if issues else ""


def _validate_generated_code(code: str, expected_steps: int, query: str = "") -> str:
    """Return empty string when code looks complete, else a short validation error."""
    if not code.strip():
        return "empty output"

    if "class GeneratedScene(VoiceoverScene):" not in code:
        return "missing GeneratedScene class"

    if "def construct(self):" not in code:
        return "missing construct method"

    if "self.set_speech_service(" not in code:
        return "missing speech service setup"
    if "GTTSService" not in code and "AzureService" not in code:
        return "missing supported speech service import"

    if "with self.voiceover(" not in code:
        return "missing voiceover blocks"

    # Sandbox constraint checks
    if "ImageMobject" in code:
        return "ImageMobject is forbidden — no image files exist in the sandbox"
    if "SVGMobject" in code:
        return "SVGMobject is forbidden — no SVG files exist in the sandbox"
    if ".to_center()" in code:
        return ".to_center() does not exist in Manim — use .move_to(ORIGIN) instead"

    # Hard-fail on hallucinated API that crashes Manim immediately
    if "SurroundingRoundedRectangle" in code:
        return (
            "SurroundingRoundedRectangle does NOT exist in Manim CE — "
            "replace ALL occurrences with SurroundingRectangle(obj, corner_radius=0.1, color=YELLOW)"
        )

    if _likely_truncated_tail(code):
        return "output appears truncated near the end"

    if expected_steps > 3 and len(code.splitlines()) < 45:
        return "output too short for requested step count"

    tree_error = _binary_tree_completeness_error(code, query)
    if tree_error:
        return tree_error

    # Check cleanup between scenes
    cleanup_warning = _validate_scene_cleanup(code)
    if cleanup_warning:
        # Not a hard failure, but log it
        print(f"[Coder] ⚠️ {cleanup_warning}")

    # Check voice-text sync pattern
    sync_warning = _validate_voiceover_sync(code)
    if sync_warning:
        print(f"[Coder] ⚠️ {sync_warning}")

    # Check timing budget
    timing_warning = _validate_timing_budget(code)
    if timing_warning:
        print(f"[Coder] ⚠️ {timing_warning}")

    # Check for static self.wait(tracker.duration) calls inside voiceover blocks
    _static_wait_pat = re.compile(
        r'with\s+self\.voiceover\s*\(.*?\)\s*as\s+\w+\s*:.*?self\.wait\(tracker\.duration\)',
        re.DOTALL
    )
    if _static_wait_pat.search(code):
        print("[Coder] ⚠️ STATIC WAIT detected: self.wait(tracker.duration) inside voiceover block "
              "freezes screen with no animation. Replace with Indicate() or other animated call.")

    # Check for invisible panel Rectangles that pollute self.mobjects
    if re.search(r'(main_visual_area|key_text_panel)\s*=\s*Rectangle\(', code):
        print("[Coder] ⚠️ INVISIBLE PANEL detected: main_visual_area/key_text_panel Rectangle objects "
              "pollute self.mobjects and cause ghost overlap. Use coordinate constants instead.")

    # Check for Segment 2 overlap anti-patterns
    seg2_overlap = _validate_segment2_layout(code)
    if seg2_overlap:
        print(f"[Coder] ⚠️ SEGMENT 2 OVERLAP RISK: {seg2_overlap}")

    # Check for invalid color constants (ORANGE_E, PINK_A, etc.)
    color_warning = _validate_color_constants(code)
    if color_warning:
        print(f"[Coder] ⚠️ COLOR ERROR: {color_warning}")

    # Check for hallucinated APIs
    api_warning = _validate_hallucinated_apis(code)
    if api_warning:
        print(f"[Coder] ⚠️ HALLUCINATED API: {api_warning}")

    # Check for horizontal pipeline overflow
    overflow_warning = _validate_horizontal_overflow(code)
    if overflow_warning:
        print(f"[Coder] ⚠️ OVERFLOW RISK: {overflow_warning}")

    # Check for voiceover loop timing issues
    loop_warning = _validate_voiceover_loop_timing(code)
    if loop_warning:
        print(f"[Coder] ⚠️ LOOP TIMING: {loop_warning}")

    try:
        compile(code, "scene.py", "exec")
    except SyntaxError as e:
        return f"syntax error: {e.msg} at line {e.lineno}"

    return ""


def _strip_kwarg(code: str, kwarg_name: str) -> str:
    """Remove a keyword argument from any function call in code.

    Handles complex value expressions with nested parentheses,
    e.g. ``max_width=banner.get_width() * 0.8``.
    """
    result: list[str] = []
    i = 0
    pattern = re.compile(r',\s*' + kwarg_name + r'\s*=\s*')

    while i < len(code):
        m = pattern.search(code, i)
        if not m:
            result.append(code[i:])
            break

        # Keep everything before the match
        result.append(code[i:m.start()])

        # Skip the value expression after "kwarg_name="
        j = m.end()
        depth = 0
        in_string = None
        while j < len(code):
            ch = code[j]
            if in_string:
                if ch == in_string and (j == 0 or code[j - 1] != '\\'):
                    in_string = None
            elif ch in ('"', "'"):
                in_string = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                if depth == 0:
                    break  # closing paren of enclosing call
                depth -= 1
            elif ch == ',' and depth == 0:
                break  # start of next argument
            j += 1

        i = j  # continue from the delimiter we stopped at

    return ''.join(result)


def _apply_preventive_fixes(code: str) -> str:
    """Apply deterministic fixes for known deprecated/broken API patterns."""
    fixed = code
    for pattern, replacement in _PREVENTIVE_SUBSTITUTIONS:
        fixed = re.sub(pattern, replacement, fixed)

    # ── CRITICAL: Strip voiceover bookmarks ──
    # GTTSService does NOT support word-boundary transcription, so any
    # <bookmark .../> tag or wait_until_bookmark() call will crash at runtime.
    fixed = re.sub(r'<bookmark\s+mark=[\'"][^\'"]*[\'"]\s*/>', '', fixed)
    fixed = re.sub(r'\bself\.wait_until_bookmark\s*\([^)]*\)\s*\n?', '', fixed)



    # ── BELT-AND-SUSPENDERS: plain string replacement for SurroundingRoundedRectangle ──

    # The _PREVENTIVE_SUBSTITUTIONS regex uses \b word boundaries which should always fire,

    # but as a safety net we also do a plain string replace so no edge-case can slip through.

    fixed = fixed.replace('SurroundingRoundedRectangle', 'SurroundingRectangle')

    # ── CRITICAL: Strip hallucinated Text() parameters ──
    # Text() in Manim v0.20.1 does NOT accept alignment, max_width, or justify.
    for bad_kw in ("alignment", "max_width", "justify"):
        fixed = _strip_kwarg(fixed, bad_kw)

    # ── CRITICAL: Strip 3D scene methods incompatible with VoiceoverScene ──
    # set_camera_orientation() only exists on ThreeDScene, NOT VoiceoverScene.
    # Replace with a no-op comment so the rest of the code is preserved.
    fixed = re.sub(
        r'\bself\.set_camera_orientation\s*\([^)]*\)\s*\n?',
        '# [REMOVED: set_camera_orientation not available on VoiceoverScene]\n',
        fixed
    )
    # ThreeDAxes → replace with regular Axes (2D)
    fixed = re.sub(r'\bThreeDAxes\b', 'Axes', fixed)

    # ── CRITICAL: Convert Rectangle(corner_radius=X) → RoundedRectangle(corner_radius=X) ──
    # Rectangle() in Manim v0.20.1 does NOT accept corner_radius; only RoundedRectangle does.
    if 'corner_radius' in fixed:
        def _fix_cr_line(line: str) -> str:
            if 'corner_radius' in line and 'Rectangle(' in line and 'RoundedRectangle' not in line:
                line = line.replace('Rectangle(', 'RoundedRectangle(')
            return line
        fixed = '\n'.join(_fix_cr_line(ln) for ln in fixed.split('\n'))

    # ── CRITICAL: Convert bare opacity= to stroke_opacity=/fill_opacity= in geometry constructors ──
    # Line, Arrow, DashedLine, Arc etc. do NOT accept opacity= — only stroke_opacity= or fill_opacity=.
    # This is a targeted fix: only modify lines containing geometry constructors + bare opacity=.
    _STROKE_GEOM = ('Line(', 'Arrow(', 'DashedLine(', 'DashedVMobject(', 'Arc(',
                    'CurvedArrow(', 'DoubleArrow(', 'TangentLine(')
    _FILL_GEOM = ('Circle(', 'Dot(', 'Square(', 'Polygon(', 'Triangle(',
                  'Ellipse(', 'Annulus(', 'AnnularSector(', 'Sector(')
    if ', opacity=' in fixed or ',opacity=' in fixed:
        def _fix_opacity_line(line: str) -> str:
            # Skip lines that already use stroke_opacity or fill_opacity
            if 'stroke_opacity' in line or 'fill_opacity' in line:
                return line
            # Skip .set_fill(opacity=...) and .set_stroke(opacity=...) — those are valid
            if '.set_fill(' in line or '.set_stroke(' in line or '.set_opacity(' in line:
                return line
            # Check if this line has a geometry constructor with bare opacity=
            for geom in _STROKE_GEOM:
                if geom in line and re.search(r'\bopacity\s*=', line):
                    line = re.sub(r'\bopacity\s*=', 'stroke_opacity=', line)
                    return line
            for geom in _FILL_GEOM:
                if geom in line and re.search(r'\bopacity\s*=', line):
                    line = re.sub(r'\bopacity\s*=', 'fill_opacity=', line)
                    return line
            # For Rectangle/RoundedRectangle — could be either fill or stroke
            for geom in ('Rectangle(', 'RoundedRectangle('):
                if geom in line and re.search(r'\bopacity\s*=', line):
                    line = re.sub(r'\bopacity\s*=', 'fill_opacity=', line)
                    return line
            return line
        fixed = '\n'.join(_fix_opacity_line(ln) for ln in fixed.split('\n'))

    # ── CRITICAL: Guard tracker.get_remaining_duration() against zero/negative ──
    # Wraps bare calls so they never produce a non-positive run_time.
    fixed = re.sub(
        r'self\.wait\(tracker\.get_remaining_duration\(\)\)',
        'self.wait(max(0.05, tracker.get_remaining_duration()))',
        fixed
    )
    fixed = re.sub(
        r'run_time\s*=\s*tracker\.get_remaining_duration\(\)',
        'run_time=max(0.05, tracker.get_remaining_duration())',
        fixed
    )

    # ── CRITICAL: Strip invalid kwargs from get_x/y_axis_label() calls only ──
    # These methods do NOT accept color=, font_size=, weight=, stroke_width=.
    # Uses a targeted regex that only touches lines containing get_x/y_axis_label.
    def _strip_axis_label_kwargs(code_str: str) -> str:
        lines = code_str.split('\n')
        out = []
        for line in lines:
            if 'get_x_axis_label' in line or 'get_y_axis_label' in line or 'get_axis_labels' in line:
                for bad_kw in ('color', 'font_size', 'weight', 'stroke_width'):
                    line = re.sub(r',\s*' + bad_kw + r'\s*=[^,)]+', '', line)
            out.append(line)
        return '\n'.join(out)
    fixed = _strip_axis_label_kwargs(fixed)

    # Replace undefined bare color constants with hex equivalents (module-level _COLOR_MAP)
    for color_name, hex_val in _COLOR_MAP.items():
        fixed = re.sub(r'\b' + color_name + r'\b(?!\s*=)', hex_val, fixed)

    # ── Strip Paragraph(... width=X ...) — width= is NOT a layout constraint in Manim v0.20 ──
    # Paragraph accepts width= as a text initializer internally but it does NOT constrain
    # the rendered width — the text will overflow. Strip it and let scale_to_fit_width handle it.
    # NOTE: Do NOT use _strip_kwarg(fixed, 'width') here — it strips width= from ALL calls
    # including Rectangle(width=4) and RoundedRectangle(width=4), breaking valid code.
    # Use only the targeted Paragraph regex below.
    fixed = re.sub(
        r'(Paragraph\([^)]*),\s*width\s*=\s*[^,)]+([^)]*\))',
        r'\1\2',
        fixed
    )

    # ── Guard: replace self.wait(tracker.duration) static pauses inside voiceover blocks ──
    # A bare self.wait(tracker.duration) freezes the screen with no animation.
    # Replace it with a comment reminder — the LLM must add an animation.
    # We can't auto-fix this perfectly (no object to Indicate), so just guard the trivial case.
    fixed = re.sub(
        r'\bself\.wait\(tracker\.duration\)',
        'self.wait(max(0.05, tracker.get_remaining_duration()))  # guard: was static wait',
        fixed
    )


    # ── Strip prosody= kwarg when active provider is gTTS ──
    # GTTSService does NOT support prosody=; calling self.voiceover(prosody={...}) with
    # gTTS raises: TypeError: gTTS.__init__() got an unexpected keyword argument 'prosody'.
    # Strip prosody from generated code whenever gTTS is the *primary* provider so the
    # code is always safe. When Azure is primary, prosody stays (Azure supports it).
    # The orchestrator's _force_gtts_fallback() also strips prosody on Azure failures.
    _active_provider = os.getenv('TTS_PROVIDER', 'azure').strip().lower()
    if _active_provider != 'azure':
        fixed = _strip_prosody_kwargs(fixed)
    return fixed


def _stitch_continuation(base_code: str, continuation: str) -> str:
    """Append continuation safely to existing code."""
    left = (base_code or "").rstrip()
    right = (continuation or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n{right}"


def _continuation_prompt(query: str, plan_json: str, partial_code: str) -> str:
    tail_lines = "\n".join((partial_code or "").splitlines()[-60:])
    return (
        "You are continuing an incomplete Python file for a Manim Voiceover scene.\n"
        "Do NOT restart from imports or class declaration.\n"
        "Return ONLY the remaining lines that should come after the partial tail.\n"
        "Do not include markdown fences or explanations.\n"
        "IMPORTANT: Ensure ALL 5 segments are completed (intro, theory, animation, user message, summary).\n"
        "Each segment must start with a full cleanup: self.play(*[FadeOut(mob) for mob in self.mobjects])\n\n"
        f"Topic: {query}\n"
        f"Plan JSON: {plan_json}\n\n"
        "Partial file tail (continue from here):\n"
        f"{tail_lines}\n"
    )


def _build_rag_context(query: str) -> str:
    """Retrieve relevant Manim doc patterns via RAG."""
    try:
        context = rag_retrieve(query, k=3)
        if context:
            print(f"[Coder] RAG injected {len(context)} chars of Manim API context")
        return context or "(no RAG context available)"
    except Exception as e:
        print(f"[Coder] RAG retrieval failed: {e}")
        return "(no RAG context available)"


def _build_feedback_section(category: str) -> str:
    """Build few-shot section from user-approved examples."""
    try:
        learned = get_learned_examples(category, k=2)
        if not learned:
            return ""
        parts = ["PROVEN EXAMPLES (these compiled successfully and were approved by users — follow their patterns):"]
        for ex in learned:
            code_snippet = (ex.get("code") or "")[:1000]
            parts.append(f"Query: {ex.get('query', '')}\nCode snippet:\n{code_snippet}\n---")
        section = "\n".join(parts)
        print(f"[Coder] Injected {len(learned)} feedback example(s) for category '{category}'")
        return section
    except Exception as e:
        print(f"[Coder] Feedback retrieval failed: {e}")
        return ""


def generate_manim_code(query: str, plan: dict = None) -> str:
    """
    Calls Gemini to generate Manim code from the planner output.
    Guards against truncation and enforces completeness.
    Injects RAG context and feedback examples for higher quality.
    Validates video structure after generation.
    """
    print(f"[Coder] Generating code for: {query}")

    plan = plan or {}
    plan_payload = _build_coder_plan_payload(plan)
    plan_json = json.dumps(plan_payload, indent=2)

    # Retrieve relevant Manim API context via RAG
    manim_docs = _build_rag_context(query)

    # Retrieve user-approved feedback examples
    category = plan.get("visual_style", "diagram")
    feedback_examples = _build_feedback_section(category)
    tts_ctx = _build_tts_prompt_context()

    prompt = CODER_PROMPT.format(
        query=query,
        plan_json=plan_json,
        manim_docs=manim_docs,
        feedback_examples=feedback_examples,
        tts_import_rules=tts_ctx["tts_import_rules"],
        tts_setup_rules=tts_ctx["tts_setup_rules"],
        voiceover_sync_rules=tts_ctx["voiceover_sync_rules"],
    )
    expected_steps = len(plan.get("steps", []))
    last_code = ""
    last_validation_error = ""

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt_for_attempt = prompt
        if attempt > 1:
            prompt_for_attempt += (
                "\n\nIMPORTANT RETRY INSTRUCTION:\n"
                "Previous output was incomplete or invalid. "
                "Return the FULL file from imports to the final wait call. "
                "Do not stop mid-line or mid-block.\n"
                "ENSURE ALL SEGMENTS FROM THE PLAN ARE PRESENT. "
                "Check the plan's segments array and generate code for EVERY segment type listed.\n"
                f"Validation failure to fix: {last_validation_error}\n"
            )

        response_details = call_llm_detailed(
            prompt_for_attempt,
            max_tokens=16384,
            preferred_model=CODER_MODEL,
            disable_thinking=True,
        )

        code = extract_code(response_details.get("text", ""))
        finish_reason = str(response_details.get("finish_reason", "")).upper()
        last_code = code

        # Apply preventive fixes BEFORE validation — auto-substitution catches known bad patterns
        # (e.g. SurroundingRoundedRectangle, .get_graph(), etc.) before they trigger a hard-fail.
        code = _apply_preventive_fixes(code)
        last_code = code

        validation_error = _validate_generated_code(code, expected_steps=expected_steps, query=query)

        if validation_error and "MAX_TOKENS" in finish_reason:
            continued_code = code
            continuation_error = validation_error
            for hop in range(1, MAX_CONTINUATION_ATTEMPTS + 1):
                cont_prompt = _continuation_prompt(query, plan_json, continued_code)
                cont_details = call_llm_detailed(
                    cont_prompt,
                    max_tokens=8192,
                    preferred_model=CODER_MODEL,
                    disable_thinking=True,
                )
                cont_piece = extract_code(cont_details.get("text", ""))
                if not cont_piece.strip():
                    break

                continued_code = _stitch_continuation(continued_code, cont_piece)
                continuation_error = _validate_generated_code(
                    continued_code,
                    expected_steps=expected_steps,
                    query=query,
                )
                if not continuation_error:
                    print(f"[Coder] ✅ Continuation succeeded after {hop} hop(s)")
                    code = continued_code
                    validation_error = ""
                    break

                cont_finish = str(cont_details.get("finish_reason", "")).upper()
                if "MAX_TOKENS" not in cont_finish:
                    break

            last_code = code if not validation_error else continued_code

        if not validation_error:
            break

        last_validation_error = validation_error
        print(f"[Coder] ⚠️  Generation attempt {attempt} failed validation: {validation_error}")
    else:
        print("[Coder] ⚠️  Returning last generated code after retries; debugger may be required.")

    code = last_code

    # Apply preventive fixes for deprecated/broken APIs
    code = _apply_preventive_fixes(code)

    # ── Post-generation structure validation ──
    report = validate_video_structure(code, plan)
    code_lines = len(code.splitlines())
    print(f"[Coder] Generated {code_lines} lines of code")
    print(f"[Coder] Structure: voiceovers={report['voiceover_count']}, "
          f"cleanups={report['cleanup_count']}, "
          f"full_cleanups={report.get('full_cleanup_count', 0)}, "
          f"title={report['has_title']}, summary={report['has_summary']}")

    if report.get("issues"):
        print(f"[Coder] ⚠️ Structure issues: {report['issues']}")
    if report.get("suggestions"):
        for sug in report["suggestions"][:2]:
            print(f"[Coder] 💡 {sug}")

    return code


def revise_manim_code(
    query: str,
    plan: dict,
    existing_code: str,
    change_request: str,
    max_attempts: int = 3,
) -> str:
    """Apply user-requested changes to existing code with validation and retry."""
    print(f"[Coder] Revising code with user feedback: {change_request[:120]}")

    plan_payload = _build_coder_plan_payload(plan or {})
    plan_json = json.dumps(plan_payload, indent=2)

    manim_docs = _build_rag_context(query)

    prompt = REVISION_PROMPT.format(
        query=query,
        plan_json=plan_json,
        manim_docs=manim_docs,
        change_request=change_request,
        existing_code=existing_code,
    )

    expected_steps = len((plan or {}).get("steps", []))
    last_code = existing_code
    last_validation_error = ""

    for attempt in range(1, max_attempts + 1):
        prompt_for_attempt = prompt
        if attempt > 1:
            prompt_for_attempt += (
                "\n\nIMPORTANT RETRY INSTRUCTION:\n"
                "The previous revision was invalid. Return the full corrected file.\n"
                f"Validation failure to fix: {last_validation_error}\n"
            )

        details = call_llm_detailed(
            prompt_for_attempt,
            max_tokens=16384,
            preferred_model=CODER_MODEL,
            disable_thinking=True,
        )

        candidate = extract_code(details.get("text", ""))
        finish_reason = str(details.get("finish_reason", "")).upper()
        if not candidate.strip():
            candidate = last_code

        # Apply preventive fixes BEFORE validation
        candidate = _apply_preventive_fixes(candidate)

        validation_error = _validate_generated_code(
            candidate,
            expected_steps=expected_steps,
            query=query,
        )

        if validation_error and "MAX_TOKENS" in finish_reason:
            continued_code = candidate
            for _ in range(MAX_CONTINUATION_ATTEMPTS):
                cont_prompt = _continuation_prompt(query, plan_json, continued_code)
                cont = call_llm_detailed(
                    cont_prompt,
                    max_tokens=4096,
                    preferred_model=CODER_MODEL,
                    disable_thinking=True,
                )
                cont_piece = extract_code(cont.get("text", ""))
                if not cont_piece.strip():
                    break
                continued_code = _stitch_continuation(continued_code, cont_piece)
                validation_error = _validate_generated_code(
                    continued_code,
                    expected_steps=expected_steps,
                    query=query,
                )
                if not validation_error:
                    candidate = continued_code
                    break
                if "MAX_TOKENS" not in str(cont.get("finish_reason", "")).upper():
                    break

        if not validation_error:
            print(f"[Coder] Revision successful on attempt {attempt}")
            return _apply_preventive_fixes(candidate)

        last_code = candidate
        last_validation_error = validation_error
        print(f"[Coder] ⚠️  Revision attempt {attempt} failed validation: {validation_error}")

    print("[Coder] ⚠️  Returning last revised code after max attempts")
    return _apply_preventive_fixes(last_code)


def extract_code(text: str) -> str:
    """
    Extracts Python code from LLM response.
    Handles cases where model wraps in ```python blocks or not.
    """
    clean = (text or "").strip()

    # Try to find ```python ... ``` block
    match = re.search(r'```python\s*(.*?)```', clean, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try plain ``` block
    match = re.search(r'```\s*(.*?)```', clean, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Handle incomplete fences
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines:
            lines = lines[1:]
        clean = "\n".join(lines).strip()

    # Remove trailing unmatched closing fence
    if clean.endswith("```"):
        clean = clean[:-3].rstrip()

    return clean