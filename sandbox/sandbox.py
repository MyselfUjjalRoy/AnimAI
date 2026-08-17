import subprocess
import os
import shutil
import tempfile
import glob
import logging

log = logging.getLogger("animai.sandbox")

try:
    from agent.failure_logger import log_failure as _log_failure
except ImportError:
    def _log_failure(**kwargs) -> str:
        return ""

MIN_VIDEO_BYTES = 200_000

# In production (Render, Railway, etc.) we run Manim directly — no Docker.
# Set RENDER=true or leave DOCKER_AVAILABLE unset to use direct mode.
USE_DOCKER = os.getenv("RENDER", "false").lower() != "true" and shutil.which("docker") is not None
DOCKER_IMAGE = "manim-voiceover"


def _tail(text: str, max_chars: int = 20000) -> str:
    text = text or ""
    return text[-max_chars:] if len(text) > max_chars else text


def _extract_actual_error(stderr: str) -> str:
    if not stderr:
        return ""
    lines = stderr.splitlines()
    error_lines: list = []
    in_traceback = False
    for line in lines:
        if "Traceback" in line and "most recent call last" in line:
            in_traceback = True
            error_lines = [line]
        elif in_traceback:
            error_lines.append(line)
            stripped = line.strip()
            if stripped and not line.startswith(" ") and not line.startswith("\t") and stripped != line:
                continue
            if any(stripped.startswith(e) for e in (
                "TypeError", "ValueError", "NameError", "AttributeError",
                "RuntimeError", "ImportError", "KeyError", "IndexError",
                "FileNotFoundError", "ModuleNotFoundError", "SyntaxError",
                "ZeroDivisionError", "StopIteration", "RecursionError",
            )):
                break
    if error_lines:
        return "\n".join(error_lines)
    for line in reversed(lines):
        stripped = line.strip()
        if any(stripped.startswith(e) for e in (
            "TypeError", "ValueError", "NameError", "AttributeError",
            "RuntimeError", "ImportError", "KeyError", "IndexError",
            "FileNotFoundError", "ModuleNotFoundError",
        )):
            return stripped
    return _tail(stderr, 3000)


def _pick_newest(paths):
    existing = [p for p in paths if os.path.isfile(p)]
    if not existing:
        return None
    return max(existing, key=os.path.getmtime)


def _find_best_video(output_root: str):
    generated_scene = glob.glob(
        os.path.join(output_root, "videos", "**", "GeneratedScene.mp4"),
        recursive=True,
    )
    best = _pick_newest(generated_scene)
    if best and os.path.getsize(best) >= MIN_VIDEO_BYTES:
        return best, ""

    all_mp4 = glob.glob(os.path.join(output_root, "**", "*.mp4"), recursive=True)
    final_candidates = [
        p for p in all_mp4
        if "partial_movie_files" not in p.replace("\\", "/")
    ]
    best = _pick_newest(final_candidates)
    if best and os.path.getsize(best) >= MIN_VIDEO_BYTES:
        return best, ""

    partials = glob.glob(
        os.path.join(output_root, "**", "partial_movie_files", "**", "*.mp4"),
        recursive=True,
    )
    if partials:
        return None, "Partial segments exist but final merged video is missing"

    if best and os.path.getsize(best) < MIN_VIDEO_BYTES:
        return None, f"Final video exists but is too small ({os.path.getsize(best)} bytes)"

    return None, "No video file was generated"


def _run_direct(code: str, tmp_dir: str, timeout: int) -> subprocess.CompletedProcess:
    """Run Manim directly (no Docker) — used in production."""
    scene_path = os.path.join(tmp_dir, "scene.py")
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    manim_cmd = [
        "manim", "render",
        "--media_dir", output_dir,
        "--verbosity", "WARNING",
        "-ql",
        scene_path,
    ]

    log.info("Running Manim directly: %s", " ".join(manim_cmd))
    return subprocess.run(
        manim_cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=tmp_dir,
    )


def _run_docker(code: str, tmp_dir: str, timeout: int) -> subprocess.CompletedProcess:
    """Run Manim via Docker — used locally."""
    docker_path = tmp_dir.replace("\\", "/")
    if docker_path[1] == ":":
        docker_path = "/" + docker_path[0].lower() + docker_path[2:]

    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_path}:/sandbox",
        "--workdir", "/sandbox",
    ]

    for env_name in (
        "AZURE_SUBSCRIPTION_KEY", "AZURE_SERVICE_REGION",
        "TTS_PROVIDER", "TTS_FALLBACK_PROVIDER",
        "AZURE_TTS_VOICE", "AZURE_TTS_STYLE",
    ):
        if os.getenv(env_name):
            docker_cmd.extend(["-e", env_name])

    docker_cmd.extend([
        DOCKER_IMAGE,
        "manim", "render",
        "--media_dir", "/sandbox/output",
        "--verbosity", "WARNING",
        "-ql",
        "/sandbox/scene.py",
    ])

    log.info("Running Manim via Docker: %s", DOCKER_IMAGE)
    return subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def run_manim_sandbox(code: str, timeout: int = 300, query: str = "") -> dict:
    """
    Takes Manim Python code as a string.
    Runs it directly (production) or via Docker (local).
    Returns success/failure + video path or error message.
    """
    try:
        compile(code, "scene.py", "exec")
    except SyntaxError as e:
        os.makedirs("outputs", exist_ok=True)
        with open(os.path.join("outputs", "last_failed_scene.py"), "w", encoding="utf-8") as f:
            f.write(code)
        syntax_error = (
            f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}\n"
            f"Line: {e.text or ''}"
        )
        with open(os.path.join("outputs", "last_failed_log.txt"), "w", encoding="utf-8") as f:
            f.write(syntax_error)
        _log_failure(error=syntax_error, code=code, failure_type="syntax_error", query=query)
        log.error("Syntax check failed: %s", syntax_error)
        return {"success": False, "error": syntax_error, "stdout": "", "stderr": syntax_error}

    tmp_dir = tempfile.mkdtemp()
    try:
        scene_path = os.path.join(tmp_dir, "scene.py")
        with open(scene_path, "w", encoding="utf-8") as f:
            f.write(code)

        if USE_DOCKER:
            result = _run_docker(code, tmp_dir, timeout)
        else:
            result = _run_direct(code, tmp_dir, timeout)

        if result.returncode == 0:
            video_path, video_error = _find_best_video(os.path.join(tmp_dir, "output"))
            if video_path:
                os.makedirs("outputs", exist_ok=True)
                final_path = os.path.join("outputs", "animation.mp4")
                shutil.copy(video_path, final_path)
                with open(os.path.join("outputs", "last_success_scene.py"), "w", encoding="utf-8") as f:
                    f.write(code)
                log.info("Success! Video: %s", final_path)
                return {"success": True, "video_path": final_path, "stdout": result.stdout, "stderr": result.stderr}
            else:
                log.error("Compiled but video invalid: %s", video_error)
                return {"success": False, "error": video_error, "stdout": result.stdout, "stderr": result.stderr}
        else:
            extracted = _extract_actual_error(result.stderr)
            stderr_tail = _tail(result.stderr)
            stdout_tail = _tail(result.stdout)
            combined_error = (
                f"[extracted error]\n{extracted}\n\n"
                f"[manim stderr tail]\n{stderr_tail}\n\n"
                f"[manim stdout]\n{stdout_tail}"
            )

            os.makedirs("outputs", exist_ok=True)
            with open(os.path.join("outputs", "last_failed_scene.py"), "w", encoding="utf-8") as f:
                f.write(code)
            with open(os.path.join("outputs", "last_failed_log.txt"), "w", encoding="utf-8") as f:
                f.write(combined_error)

            _err_lower = combined_error.lower()
            if "word boundaries are required" in _err_lower or "wait_until_bookmark" in _err_lower:
                _failure_type = "bookmark_error"
            elif "speech synthesis failed" in _err_lower or "cancellationreason.error" in _err_lower or "ws_open_error" in _err_lower or ("dns" in _err_lower and "resolution failed" in _err_lower):
                _failure_type = "tts_synthesis_failed"
            elif "no video file was generated" in _err_lower:
                _failure_type = "no_video"
            elif "syntaxerror" in _err_lower:
                _failure_type = "syntax_error"
            elif "attributeerror" in _err_lower:
                _failure_type = "attribute_error"
            elif "nameerror" in _err_lower:
                _failure_type = "name_error"
            elif "typeerror" in _err_lower:
                _failure_type = "type_error"
            else:
                _failure_type = "runtime_error"

            _log_failure(error=combined_error, code=code, failure_type=_failure_type, query=query)
            log.error("Compilation failed: %s", _tail(combined_error, 500))
            return {"success": False, "error": combined_error, "stdout": result.stdout, "stderr": result.stderr}

    except subprocess.TimeoutExpired:
        log.error("Timed out after %ds", timeout)
        return {"success": False, "error": f"Rendering timed out after {timeout} seconds"}
    except Exception as e:
        log.error("Unexpected error: %s", e)
        return {"success": False, "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
