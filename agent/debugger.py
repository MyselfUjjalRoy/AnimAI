import os
import re

from agent.llm import call_llm
from agent.coder import extract_code, _strip_kwarg, _COLOR_MAP
from rag.retriever import retrieve as rag_retrieve

DEBUGGER_MODEL = os.getenv("GEMINI_DEBUGGER_MODEL", "gemini-2.5-flash")

DEBUGGER_PROMPT = """You are an expert Manim debugger.
Your ONLY job: fix the specific compilation error with minimal changes.

BROKEN CODE:
{code}

ERROR:
{error}

⚠️  CRITICAL RULES:
1. PRESERVE all animation logic, visual elements, and scene structure.
2. Fix ONLY the line or block causing the error — do NOT rewrite the whole scene.
3. If the error is a truncated/incomplete line (like `ELEMENT_SP` with no value), complete it intelligently.
4. If lines are cut off mid-definition, infer the intent and complete them.
5. NEVER replace working code with placeholder animations like 'Hello World'.
6. Return the complete, fixed file from `from manim import *` to the end of `construct()`.
7. Ensure all parentheses, brackets, and string quotes are balanced.
8. The fixed code must pass `compile(code, 'scene.py', 'exec')` check.
9. If the code uses `VoiceoverScene` with AzureService/GTTSService, preserve speech-service imports and `self.set_speech_service(...)` setup.
10. For Manim positioning APIs, use callable getters like `get_center()`, `get_top()`, `get_bottom()` when passing points to `move_to`, `next_to`, or vector math.

LAYOUT AWARENESS RULES:
11. If the error involves objects going off-screen or clipping, add `.scale_to_fit_width(config.frame_width - 2)` or reduce `font_size` values.
12. If the error shows leftover elements or pile-up, add `self.play(FadeOut(*old_elements))` before the failing animation block.
13. Common layout fixes:
    - "off-screen" or "clipping" → add scale_to_fit_width/height after positioning
    - "overlap" → increase buff values in .arrange() calls to at least 0.3, or reduce font_size
    - elements piling up → add FadeOut of previous step's elements before the new step
    - content too wide → add `.scale_to_fit_width(config.frame_width - 2)` on the VGroup
    - content too tall → add `.scale_to_fit_height(config.frame_height - 1.5)` on the VGroup

SANDBOX CONSTRAINT RULES — CRITICAL:
14. The code runs inside Docker with ONLY scene.py mounted. NO external files (images, SVGs, assets) exist.
15. If the error involves a missing file (e.g., FileNotFoundError for .png, .svg, .jpg), REPLACE the ImageMobject/SVGMobject with programmatic Manim shapes (Rectangle, Circle, Triangle, Polygon, etc.).
16. `import os` is allowed only for reading TTS environment variables; never add filesystem operations.
17. NEVER use `.to_center()` — it does not exist in Manim. Use `.move_to(ORIGIN)` instead.

Common code fixes:
- Incomplete lines: `ELEMENT_SP` → `ELEMENT_SPACING = 0.5`
- Missing imports: add before class definition
- Unmatched brackets/parens: close them properly
- Undefined variables: check context and add missing definitions
- `.center` used as property → change to `.get_center()`
- `.get_top` / `.get_bottom` used without `()` → add `()`
- `ImageMobject("file.png")` → replace with shapes (e.g., Rectangle + Circle)
- `.to_center()` → `.move_to(ORIGIN)`
- `axes.get_graph(lambda ...)` → `axes.plot(lambda ...)` (get_graph is deprecated in Manim 0.18+)
- `axes.get_vertical_line_to_graph(...)` → `axes.get_vertical_line(...)` (deprecated)
- `getter() takes 1 positional argument but 2 were given` → deprecated method; replace with modern equivalent
- `Word boundaries are required for timing with bookmarks` → GTTSService does NOT support bookmarks.
  Remove ALL `<bookmark mark='...'/>` tags and ALL `self.wait_until_bookmark(...)` calls.
- `'GeneratedScene' object has no attribute 'set_camera_orientation'` → VoiceoverScene DOES NOT
  inherit from ThreeDScene. Remove ALL `self.set_camera_orientation(...)` calls. Replace any
  `ThreeDAxes` with `Axes`. Replace `Surface(...)` and `Sphere(...)` with 2D equivalents.
- `Text() got unexpected keyword argument 'alignment'` → Remove `alignment=` from Text() calls.
  Use `VGroup(...).arrange(DOWN, aligned_edge=LEFT)` for alignment.
- `Text() got unexpected keyword argument 'max_width'` → Remove `max_width=` from Text() calls.
  Use `.scale_to_fit_width(X)` after creation instead.
- `_get_axis_label() got an unexpected keyword argument 'color'` → `get_x_axis_label()` and
  `get_y_axis_label()` do NOT accept `color=`, `font_size=`, or `weight=`. Move those kwargs
  inside the `Tex()`/`Text()` object: `axes.get_x_axis_label(Tex(r"$x$", color=WHITE), edge=DOWN)`.
- `Rectangle() got unexpected keyword argument 'corner_radius'` → `Rectangle` does NOT accept
  `corner_radius`. Use `RoundedRectangle(corner_radius=X, width=W, height=H)` instead.
  Replace every `Rectangle(..., corner_radius=X)` → `RoundedRectangle(corner_radius=X, ...)`.
- `wait() has a duration of ... <= 0 seconds` → Too many `run_time=tracker.duration * N` calls
  inside a single voiceover block exceeded `tracker.duration` total. Fix: replace
  `self.wait(tracker.get_remaining_duration())` → `self.wait(max(0.05, tracker.get_remaining_duration()))`
  AND replace `run_time=tracker.get_remaining_duration()` → `run_time=max(0.05, tracker.get_remaining_duration())`.
  Also check: the sum of all `run_time=tracker.duration * X` must be < 1.0 (i.e., total fraction < 1).
  Reduce individual fractions so they sum to at most 0.85.
- `Mobject.__getattr__.<locals>.getter() takes 1 positional argument but 2 were given` →
  This is caused by calling `.get_part_by_text("some string")` on a `Paragraph` or `Text`.
  These getter methods do NOT accept string arguments in this Manim version.
  REMOVE all `obj.get_part_by_text("...")` calls and replace with `Indicate(obj, ...)` on the
  whole object, or break the Paragraph into separate Text() objects (one per line) and
  Indicate the specific Text object directly.
- LAYOUT: Never use `Rectangle()` as invisible layout guides named `main_visual_area` or
  `key_text_panel`. They pollute `self.mobjects` and cause ghost elements between segments.
  Replace with coordinate constants: `MAIN_X = -3.2`, `KEY_X = 4.6`, `KEY_TOP = 1.8`.
  Position objects with explicit coords: `key1.move_to([KEY_X, KEY_TOP, 0])`.
- OVERLAP: Two-column layout (Seg 2/3) MUST have each column width ≤ `config.frame_width * 0.44`.
  Using `0.5` causes the columns to overlap in the centre. Always call
  `col.scale_to_fit_width(config.frame_width * 0.44)` on both left and right columns.
- Paragraph(width=...) is NOT a layout constraint in Manim v0.20.1 — text will overflow.
  Replace Paragraph with VGroup of separate Text() lines + call `.scale_to_fit_width(X)`.
- `NameError: name 'ORANGE_E' is not defined` (or ORANGE_A/B/C/D, PINK_A-E, WHITE_A-E, BLACK_A-E) →
  ORANGE, PINK, WHITE, and BLACK do NOT have _A through _E suffix variants in Manim.
  Replace with the base name: `ORANGE_E` → `ORANGE`, `PINK_C` → `PINK`, etc.
  Or use a hex string: `ORANGE_E` → `"#FF8C00"`.
- `NameError: name 'SurroundingRoundedRectangle' is not defined` →
  `SurroundingRoundedRectangle` DOES NOT EXIST in Manim. Replace with:
  `SurroundingRectangle(mobject, corner_radius=0.1, color=YELLOW)`
- `TypeError: Mobject.__init__() got an unexpected keyword argument 'opacity'` →
  Line, Arrow, Circle and other geometry constructors do NOT accept bare `opacity=`.
  Replace `opacity=X` with `stroke_opacity=X` for stroke objects (Line, Arrow, DashedLine)
  or `fill_opacity=X` for fill objects (Circle, Rectangle, RoundedRectangle).
  NOTE: `.set_fill(opacity=X)` and `.set_stroke(opacity=X)` ARE valid — only constructors reject it.
- `Text.set_text("new text")` → Text objects are IMMUTABLE. Create a new Text() and
  use `Transform(old, new)` to animate the change.
- HORIZONTAL OVERFLOW: If elements go off-screen to the right (x > 6.5), the code is
  chaining `.next_to(RIGHT)` without bounds. Fix: put ALL pipeline elements into a
  VGroup, use `.arrange(RIGHT, buff=0.4)`, then `.scale_to_fit_width(config.frame_width - 2.0)`.
- VOICEOVER LOOP TIMING: If a `for` loop inside a voiceover block uses
  `tracker.duration * X` per iteration, total = N*X which easily exceeds 1.0.
  Fix: pre-divide `per_step = tracker.duration * X / N`.

RELEVANT MANIM API REFERENCE (use to verify correct API usage):
{manim_context}

Return ONLY the complete, syntactically valid Python code. No commentary.
"""



def _apply_common_manim_runtime_fixes(code: str, error: str) -> str:
    """Apply deterministic fixes for frequent Manim runtime mistakes."""
    fixed = code
    err = (error or "").lower()

    # Common LLM mistake: `obj.center` used as a point instead of `obj.get_center()`.
    if "unsupported operand type(s) for -: 'method' and 'float'" in err or "has no attribute 'center'" in err:
        fixed = re.sub(r'\.center\b(?!\()', '.get_center()', fixed)

    # Common: using get_top/get_bottom as properties instead of methods
    if "'method'" in err:
        fixed = re.sub(r'\.get_top\b(?!\()', '.get_top()', fixed)
        fixed = re.sub(r'\.get_bottom\b(?!\()', '.get_bottom()', fixed)
        fixed = re.sub(r'\.get_left\b(?!\()', '.get_left()', fixed)
        fixed = re.sub(r'\.get_right\b(?!\()', '.get_right()', fixed)

    # Common: .to_center() does not exist in Manim, use .move_to(ORIGIN)
    if "has no attribute 'to_center'" in err:
        fixed = re.sub(r'\.to_center\(\)', '.move_to(ORIGIN)', fixed)

    # Common: SurroundingRoundedRectangle does NOT exist in Manim — use SurroundingRectangle
    if 'surroundingroundedrectangle' in err or 'SurroundingRoundedRectangle' in fixed:
        fixed = re.sub(r'\bSurroundingRoundedRectangle\b', 'SurroundingRectangle', fixed)

    # Common: bare opacity= in geometry constructors (Line, Arrow, Circle, etc.)
    # These do NOT accept opacity= — use stroke_opacity= or fill_opacity= instead.
    if "unexpected keyword argument 'opacity'" in err or (
        re.search(r'(?:Line|Arrow|DashedLine|Circle|Dot|Rectangle|RoundedRectangle|Arc)\(', fixed)
        and ', opacity=' in fixed
    ):
        _stroke_geom = ('Line(', 'Arrow(', 'DashedLine(', 'Arc(', 'CurvedArrow(', 'DoubleArrow(')
        _fill_geom = ('Circle(', 'Dot(', 'Square(', 'Polygon(', 'Rectangle(', 'RoundedRectangle(')
        def _fix_opacity(line):
            if 'stroke_opacity' in line or 'fill_opacity' in line:
                return line
            if '.set_fill(' in line or '.set_stroke(' in line:
                return line
            for g in _stroke_geom:
                if g in line and re.search(r'\bopacity\s*=', line):
                    return re.sub(r'\bopacity\s*=', 'stroke_opacity=', line)
            for g in _fill_geom:
                if g in line and re.search(r'\bopacity\s*=', line):
                    return re.sub(r'\bopacity\s*=', 'fill_opacity=', line)
            return line
        fixed = '\n'.join(_fix_opacity(ln) for ln in fixed.split('\n'))

    # Common: Axes.get_graph() deprecated in Manim 0.18+ → use Axes.plot()
    if "getter() takes 1 positional argument but 2 were given" in err or "get_graph" in err:
        fixed = re.sub(r'\.get_graph\(', '.plot(', fixed)

    # Common: Axes.get_vertical_line_to_graph() deprecated → use Axes.get_vertical_line()
    if "get_vertical_line_to_graph" in err:
        fixed = re.sub(r'\.get_vertical_line_to_graph\(', '.get_vertical_line(', fixed)

    # ── CRITICAL: Strip voiceover bookmarks (always applied) ──
    # GTTSService does NOT support transcription-based word boundaries.
    # <bookmark .../> tags + wait_until_bookmark() calls always crash.
    fixed = re.sub(r'<bookmark\s+mark=[\'"][^\'"]*[\'"]\s*/>', '', fixed)
    fixed = re.sub(r'\bself\.wait_until_bookmark\s*\([^)]*\)\s*\n?', '', fixed)

    # ── CRITICAL: Strip hallucinated Text() parameters (always applied) ──
    for bad_kw in ("alignment", "max_width", "justify"):
        fixed = _strip_kwarg(fixed, bad_kw)

    # ── CRITICAL: Strip 3D scene methods incompatible with VoiceoverScene ──
    # set_camera_orientation() only exists on ThreeDScene, NOT VoiceoverScene.
    if "set_camera_orientation" in err or "set_camera_orientation" in fixed:
        fixed = re.sub(
            r'\bself\.set_camera_orientation\s*\([^)]*\)\s*\n?',
            '# [REMOVED: set_camera_orientation not available on VoiceoverScene]\n',
            fixed
        )
        fixed = re.sub(r'\bThreeDAxes\b', 'Axes', fixed)

    # ── CRITICAL: Strip invalid kwargs from get_x/y_axis_label() calls only ──
    # These methods do NOT accept color=, font_size=, weight=, stroke_width=.
    if 'get_axis_label' in err or ('get_x_axis_label' in fixed) or ('get_y_axis_label' in fixed):
        def _fix_axis_label_line(line):
            if 'get_x_axis_label' in line or 'get_y_axis_label' in line or 'get_axis_labels' in line:
                for bad_kw in ('color', 'font_size', 'weight', 'stroke_width'):
                    line = re.sub(r',\s*' + bad_kw + r'\s*=[^,)]+', '', line)
            return line
        fixed = '\n'.join(_fix_axis_label_line(ln) for ln in fixed.split('\n'))

    # ── CRITICAL: Convert Rectangle(corner_radius=X) to RoundedRectangle(corner_radius=X) ──
    # Rectangle does NOT accept corner_radius — only RoundedRectangle does.
    if "corner_radius" in err or ("corner_radius" in fixed and "Rectangle(" in fixed):
        def _fix_corner_radius_line(line):
            if 'corner_radius' in line and 'Rectangle(' in line and 'RoundedRectangle' not in line:
                line = line.replace('Rectangle(', 'RoundedRectangle(')
            return line
        fixed = '\n'.join(_fix_corner_radius_line(ln) for ln in fixed.split('\n'))

    # ── CRITICAL: Guard tracker.get_remaining_duration() against zero/negative values ──
    # When too many run_time=tracker.duration*N fractions are used inside one voiceover block,
    # get_remaining_duration() can return 0 or negative, crashing self.wait().
    if "<= 0 seconds" in err or "get_remaining_duration" in fixed:
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

    # ── CRITICAL: Remove get_part_by_text() calls (method doesn't accept string args) ──
    if "getter() takes 1 positional argument" in err or "get_part_by_text" in fixed:
        fixed = re.sub(
            r'(\w+)\.get_part_by_text\([^)]+\)',
            r'\1',
            fixed
        )

    # ── Suppress deprecated set_width() on Paragraph/Text ──
    if ".set_width(" in fixed:
        fixed = re.sub(
            r'(Paragraph\([^)]+\))\.set_width\(([^)]+)\)',
            r'\1',
            fixed
        )

    # ── Strip Paragraph(... width=X ...) — not a layout constraint in Manim v0.20.1 ──
    if 'Paragraph(' in fixed and 'width=' in fixed:
        fixed = re.sub(
            r'(Paragraph\([^)]*),\s*width\s*=\s*[^,)]+([^)]*\))',
            r'\1\2',
            fixed
        )

    # ── Replace self.wait(tracker.duration) static pauses in voiceover blocks ──
    if 'self.wait(tracker.duration)' in fixed:
        fixed = re.sub(
            r'\bself\.wait\(tracker\.duration\)',
            'self.wait(max(0.05, tracker.get_remaining_duration()))  # was: static wait',
            fixed
        )

    # If the error specifically mentions unexpected keyword argument, strip it
    kw_match = re.search(r"unexpected keyword argument '(\w+)'", err)
    if kw_match:
        kw = kw_match.group(1)
        if kw not in ('color', 'font_size', 'weight', 'stroke_width', 'corner_radius'):
            fixed = _strip_kwarg(fixed, kw)

    # Common: undefined color constants — replace with hex equivalents
    if "is not defined" in err:
        for color_name, hex_val in _COLOR_MAP.items():
            if color_name.lower() in err:
                fixed = re.sub(r'\b' + color_name + r'\b', hex_val, fixed)

    # GTTSService does NOT support the prosody= kwarg — strip it from every voiceover call.
    # This error manifests as: TypeError: gTTS.__init__() got an unexpected keyword argument 'prosody'
    if "'prosody'" in err or ("unexpected keyword argument" in err and "prosody" in err):
        fixed = re.sub(
            r',\s*prosody\s*=\s*\{\{[^}]*\}\}|,\s*prosody\s*=\s*\{[^}]*\}',
            '',
            fixed,
        )
        print("[Debugger] Stripped prosody= kwargs (GTTSService does not support prosody)")

    return fixed


def debug_manim_code(code: str, error: str) -> str:
    """
    Takes broken code + error message.
    Asks Claude to fix it surgically (preserve logic, don't rewrite).
    Returns corrected code.
    """
    print(f"[Debugger] Analyzing error: {error[:100]}...")

    deterministic_fix = _apply_common_manim_runtime_fixes(code, error)
    if deterministic_fix != code:
        print("[Debugger] Applied deterministic Manim runtime fix")
        return deterministic_fix

    # Retrieve relevant Manim docs to help the LLM fix the error correctly
    try:
        rag_query = f"manim {error[:200]}"
        manim_context = rag_retrieve(rag_query, k=2)
        if manim_context:
            print(f"[Debugger] RAG injected {len(manim_context)} chars of API context")
        else:
            manim_context = "(no RAG context available)"
    except Exception:
        manim_context = "(no RAG context available)"

    prompt = DEBUGGER_PROMPT.format(code=code, error=error, manim_context=manim_context)
    response = call_llm(
        prompt,
        max_tokens=16384,
        preferred_model=DEBUGGER_MODEL,
        disable_thinking=True,
    )
    fixed_code = extract_code(response)

    print(f"[Debugger] Fixed code generated ({len(fixed_code.splitlines())} lines)")
    
    # Sanity check: fixed code should not be drastically shorter (surgical fix, not rewrite)
    if len(fixed_code.strip()) < len(code.strip()) * 0.4:
        print(f"[Debugger] ⚠️  WARNING: Fixed code is much shorter than original.")
        print(f"[Debugger] Original: {len(code.splitlines())} lines, Fixed: {len(fixed_code.splitlines())} lines")
        print(f"[Debugger] Debugger may have over-simplified or lost animation logic.")
    
    return fixed_code