"""
Post-generation video structure validator.

Checks generated Manim code for:
- Presence of video segments (7-segment structured or 5-segment detailed arc)
- Scene cleanup between segments
- Minimum voiceover count
- Overlapping element risk
- Title and summary presence
"""

import re
from typing import List


def validate_video_structure(code: str, plan: dict = None) -> dict:
    """Validate that generated code follows the expected video segment arc.

    Returns a dict with:
        valid: bool — True if no critical issues
        voiceover_count: int — number of voiceover blocks
        cleanup_count: int — number of FadeOut cleanup calls
        has_title: bool — whether a title/intro is present
        has_summary: bool — whether a summary/banner is present
        segments_detected: dict — which segment types were detected
        issues: list — list of issue strings
        suggestions: list — list of improvement suggestions
    """
    issues: List[str] = []
    suggestions: List[str] = []

    if not code or not code.strip():
        return {
            "valid": False,
            "voiceover_count": 0,
            "cleanup_count": 0,
            "has_title": False,
            "has_summary": False,
            "segments_detected": {},
            "issues": ["Code is empty"],
            "suggestions": [],
        }

    code_lower = code.lower()

    # ── Count voiceover blocks ──
    voiceover_blocks = re.findall(r'with\s+self\.voiceover\s*\(', code)
    voiceover_count = len(voiceover_blocks)

    if voiceover_count < 3:
        issues.append(
            f"Only {voiceover_count} voiceover blocks found — "
            f"need at least 4-5 for a complete educational video "
            f"(intro, theory, animation steps, summary)."
        )
    elif voiceover_count < 5:
        suggestions.append(
            f"Found {voiceover_count} voiceover blocks — aim for 5+ "
            f"to cover all video segments."
        )

    # ── Count cleanup calls ──
    # Full cleanup: self.play(*[FadeOut(mob) for mob in self.mobjects])
    full_cleanups = len(re.findall(
        r'self\.play\s*\(\s*\*\s*\[\s*FadeOut\s*\(\s*mob\s*\)\s*for\s+mob\s+in\s+self\.mobjects\s*\]',
        code
    ))
    # Partial cleanup: FadeOut(specific_group)
    partial_fadeouts = len(re.findall(r'FadeOut\s*\(', code))
    cleanup_count = partial_fadeouts

    if voiceover_count >= 4 and full_cleanups < 2:
        issues.append(
            f"Only {full_cleanups} full screen cleanup(s) found between segments. "
            f"Need at least 2-3 full cleanups (self.play(*[FadeOut(mob) for mob in self.mobjects])) "
            f"to prevent element pile-up between major sections."
        )

    if voiceover_count >= 3 and partial_fadeouts < voiceover_count - 1:
        suggestions.append(
            f"Found {partial_fadeouts} FadeOut calls for {voiceover_count} voiceover blocks. "
            f"Each step (except the first) should clean up the previous step's elements."
        )

    # ── Check for title/introduction ──
    has_title = bool(
        re.search(r'title\s*=\s*Text\s*\(', code, re.IGNORECASE)
        or re.search(r'INTRODUCTION|SEGMENT\s*1|intro', code_lower)
        or (re.search(r'Write\s*\(\s*title', code) and re.search(r'underline', code_lower))
    )

    if not has_title:
        issues.append(
            "No title/introduction segment detected. "
            "Videos should start with a title and hook question."
        )

    # ── Check for summary/closing ──
    has_summary = bool(
        re.search(r'banner|takeaway|summary|SEGMENT\s*5', code_lower)
        or re.search(r'RoundedRectangle.*set_fill.*#0A2A0A', code)
        or re.search(r'key\s*takeaway|remember', code_lower)
    )

    if not has_summary:
        issues.append(
            "No summary/takeaway banner detected. "
            "Videos should end with a clear takeaway message."
        )

    # ── Check for theory/key idea segment ──
    has_theory = bool(
        re.search(r'theory|key.?idea|SEGMENT\s*[23]|core\s*idea', code_lower)
        or re.search(r'theory_|key_idea', code_lower)
    )

    # ── Check for analogy segment ──
    has_analogy = bool(
        re.search(r'analogy|SEGMENT\s*[24]', code_lower)
        or re.search(r'analogy_|think.*like', code_lower)
    )

    # ── Check for working/animation segment ──
    has_working = bool(
        re.search(r'working|how.*works|core.*animation|SEGMENT\s*[35]', code_lower)
        or voiceover_count >= 3
    )

    # ── Check for user message / extra info segment ──
    has_user_or_extra = bool(
        re.search(r'user.*message|your.*question|you\s+asked|SEGMENT\s*[46]|extra.*info|misconception', code_lower)
        or re.search(r'user_query|direct_answer|your.*answer|nuance', code_lower)
    )

    # ── Detect which segments are present ──
    segments_detected = {
        "title_or_intro": has_title,
        "theory_or_key_idea": has_theory,
        "analogy": has_analogy,
        "working_or_animation": has_working,
        "extra_or_user_message": has_user_or_extra,
        "summary": has_summary,
    }

    missing_segments = [k for k, v in segments_detected.items() if not v]
    if missing_segments:
        suggestions.append(
            f"Missing video segments: {', '.join(missing_segments)}. "
            f"A complete video should cover all planned segments."
        )

    # ── Check for self.wait at the end ──
    last_lines = code.strip().splitlines()[-5:]
    has_final_wait = any("self.wait" in line for line in last_lines)
    if not has_final_wait:
        suggestions.append(
            "No self.wait() at the end of the video. "
            "Add self.wait(2.0) after the summary banner for screen retention."
        )

    # ── Check for background color ──
    has_bg = "background_color" in code
    if not has_bg:
        suggestions.append(
            "No background_color set. "
            'Add self.camera.background_color = "#0F0F1A" at the start of construct().'
        )

    # ── Check for split-screen layout ──
    has_split = bool(
        re.search(r'to_edge\s*\(\s*LEFT', code)
        and re.search(r'to_edge\s*\(\s*RIGHT', code)
    )
    if not has_split and voiceover_count >= 3:
        suggestions.append(
            "No split-screen layout detected. "
            "Use LEFT 60% for visuals and RIGHT 40% for key text panels."
        )

    valid = len(issues) == 0

    return {
        "valid": valid,
        "voiceover_count": voiceover_count,
        "cleanup_count": cleanup_count,
        "full_cleanup_count": full_cleanups,
        "has_title": has_title,
        "has_summary": has_summary,
        "segments_detected": segments_detected,
        "issues": issues,
        "suggestions": suggestions,
    }


def format_validation_for_prompt(report: dict) -> str:
    """Format a validation report into text that can be injected into a
    coder/debugger prompt for targeted fixes."""
    parts = []

    if report.get("issues"):
        parts.append("CRITICAL ISSUES (must fix):")
        for issue in report["issues"]:
            parts.append(f"  ❌ {issue}")

    if report.get("suggestions"):
        parts.append("SUGGESTIONS (should fix):")
        for sug in report["suggestions"]:
            parts.append(f"  ⚠️ {sug}")

    segments = report.get("segments_detected", {})
    missing = [k for k, v in segments.items() if not v]
    if missing:
        parts.append(f"MISSING SEGMENTS: {', '.join(missing)}")
        parts.append(
            "Add these segments to the code. Each segment should have its own "
            "voiceover block and be preceded by a full screen cleanup."
        )

    return "\n".join(parts)
