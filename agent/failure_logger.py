"""
failure_logger.py
─────────────────
Persistent failure logging for AnimAI Studio.

Every time the Manim sandbox fails, this module:
  1. Writes a timestamped JSON bundle to outputs/failure_logs/<timestamp>_<type>.json
  2. Appends a one-line summary to outputs/failure_logs/index.jsonl for quick grep/scan
  3. Keeps the two legacy flat files (last_failed_scene.py / last_failed_log.txt) intact

Bundle schema
─────────────
{
  "id":           "20260331_164500_bookmark_error",
  "timestamp":    "2026-03-31T16:45:00+05:30",
  "failure_type": "bookmark_error",
  "query":        "Explain LSTMs",
  "attempt":      2,
  "error":        "Exception: Word boundaries are required ...",
  "code":         "from manim import *\\n...",
  "code_lines":   358,
  "error_lines":  294,
  "tags":         ["GTTSService", "bookmark", "wait_until_bookmark"]
}
"""

import json
import os
import re
from datetime import datetime, timezone

FAILURE_LOG_DIR = os.path.join("outputs", "failure_logs")
FAILURE_INDEX   = os.path.join(FAILURE_LOG_DIR, "index.jsonl")


# ── Tag extraction ────────────────────────────────────────────────────────────

_TAG_PATTERNS = [
    (r"bookmark",                        "bookmark"),
    (r"wait_until_bookmark",             "wait_until_bookmark"),
    (r"AzureService",                    "AzureService"),
    (r"GTTSService",                     "GTTSService"),
    (r"Speech synthesis failed",         "tts_synthesis_failed"),
    (r"CancellationReason",              "azure_tts_error"),
    (r"WS_OPEN_ERROR",                   "azure_tts_network_error"),
    (r"DNS.*resolution failed",          "dns_failure"),
    (r"get_graph",                       "deprecated_get_graph"),
    (r"get_vertical_line_to_graph",      "deprecated_get_vertical_line_to_graph"),
    (r"ImageMobject",                    "ImageMobject"),
    (r"SVGMobject",                      "SVGMobject"),
    (r"import os",                       "import_os"),
    (r"TypeError",                       "TypeError"),
    (r"AttributeError",                  "AttributeError"),
    (r"NameError",                       "NameError"),
    (r"SyntaxError",                     "SyntaxError"),
    (r"RuntimeError",                    "RuntimeError"),
    (r"timeout",                         "timeout"),
    (r"No video file was generated",     "no_video"),
    (r"truncat",                         "truncated"),
]


def _extract_tags(code: str, error: str) -> list[str]:
    tags: list[str] = []
    combined = (code or "") + "\n" + (error or "")
    for pattern, tag in _TAG_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            tags.append(tag)
    return tags


# ── Core logging function ─────────────────────────────────────────────────────

def log_failure(
    *,
    error: str,
    code: str,
    failure_type: str = "other",
    query: str = "",
    attempt: int = 1,
) -> str:
    """
    Persist a failure bundle to disk.

    Parameters
    ----------
    error        : The error string from the sandbox.
    code         : The Manim Python code that failed.
    failure_type : Classification string from _classify_failure().
    query        : The original user query (optional).
    attempt      : Which attempt number failed (1-indexed).

    Returns
    -------
    Absolute path to the JSON bundle file.
    """
    os.makedirs(FAILURE_LOG_DIR, exist_ok=True)

    # Timestamp (local-aware, but strip tz characters that hurt filenames)
    now = datetime.now().astimezone()
    ts_iso   = now.isoformat(timespec="seconds")
    ts_file  = now.strftime("%Y%m%d_%H%M%S")

    bundle_id   = f"{ts_file}_{failure_type}"
    bundle_path = os.path.join(FAILURE_LOG_DIR, f"{bundle_id}.json")

    tags = _extract_tags(code, error)

    bundle = {
        "id":           bundle_id,
        "timestamp":    ts_iso,
        "failure_type": failure_type,
        "query":        query,
        "attempt":      attempt,
        "error":        error or "",
        "code":         code or "",
        "code_lines":   len((code or "").splitlines()),
        "error_lines":  len((error or "").splitlines()),
        "tags":         tags,
    }

    # Write full bundle
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    # Append to index (one-liner per failure)
    index_entry = {
        "id":           bundle_id,
        "timestamp":    ts_iso,
        "failure_type": failure_type,
        "query":        query,
        "attempt":      attempt,
        "tags":         tags,
        "code_lines":   bundle["code_lines"],
    }
    with open(FAILURE_INDEX, "a", encoding="utf-8") as f:
        f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

    print(f"[FailureLogger] 📂 Saved failure bundle → {bundle_path}")
    return bundle_path


# ── Analysis helpers (importable by the viewer script) ───────────────────────

def load_index() -> list[dict]:
    """Return all index entries as a list of dicts (newest first)."""
    if not os.path.exists(FAILURE_INDEX):
        return []
    entries = []
    with open(FAILURE_INDEX, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return list(reversed(entries))


def load_bundle(bundle_id: str) -> dict | None:
    """Load a full failure bundle by its ID string."""
    path = os.path.join(FAILURE_LOG_DIR, f"{bundle_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag_summary() -> dict[str, int]:
    """Count occurrences of each tag across all failures."""
    counts: dict[str, int] = {}
    for entry in load_index():
        for tag in entry.get("tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def failure_type_summary() -> dict[str, int]:
    """Count failures by failure_type."""
    counts: dict[str, int] = {}
    for entry in load_index():
        ft = entry.get("failure_type", "other")
        counts[ft] = counts.get(ft, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
