"""
scripts/app/start_full_app.py
--------------------------------
Start the full FashionSense application with the integrated search stack.

This launcher checks the pieces the merged app needs to work end-to-end:
- Python dependencies import cleanly
- Ollama is reachable (or can be started)
- the expected LLM model exists locally
- the local Qdrant collection already exists

If the prerequisites are satisfied, it starts uvicorn in the foreground.

Usage:
    python scripts/app/start_full_app.py
    python scripts/app/start_full_app.py --reload
    python scripts/app/start_full_app.py --detector-backend yolo_world
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://127.0.0.1:11434"
EXPECTED_OLLAMA_MODEL = "qwen2.5:7b-instruct-q3_K_M"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the full FashionSense app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Run uvicorn with autoreload")
    parser.add_argument(
        "--detector-backend",
        default="yolov8",
        choices=["yolov8", "yolo_world"],
        help="Detector backend to expose through the API",
    )
    parser.add_argument(
        "--model-weights",
        default="",
        help="Optional model weights folder name under models/weights (Cruella/YOLO)",
    )
    parser.add_argument(
        "--edna-weights",
        default="",
        help="Optional model weights folder name under models/weights for Edna (FashionNet). E.g. edna_1.2m",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip the Ollama availability/model checks",
    )
    parser.add_argument(
        "--skip-vector-check",
        action="store_true",
        help="Skip the vector collection check",
    )
    parser.add_argument(
        "--auto-pull-model",
        action="store_true",
        help="Automatically pull the required Ollama model if it is missing",
    )
    return parser.parse_args()


def ensure_imports() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import ultralytics  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Python dependencies are missing or broken.\n"
            "Run: pip install -r requirements.txt\n"
            f"Details: {exc}"
        ) from exc


def ollama_ready() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def find_ollama_exe() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Program Files/Ollama/ollama.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def start_ollama_if_needed() -> None:
    if ollama_ready():
        print("Ollama server is already reachable.")
        return

    ollama_exe = find_ollama_exe()
    if not ollama_exe:
        raise SystemExit(
            "Ollama is not running and the 'ollama' command was not found.\n"
            "Install Ollama and run `ollama serve`, then retry."
        )

    print("Ollama is not reachable. Trying to start `ollama serve`...")

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    subprocess.Popen(  # pragma: no cover - process management is environment-dependent
        [ollama_exe, "serve"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )

    for _ in range(20):
        time.sleep(1)
        if ollama_ready():
            print("Ollama server started successfully.")
            return

    raise SystemExit(
        "Tried to start Ollama, but it is still not reachable.\n"
        "Start it manually with `ollama serve` and retry."
    )


def ensure_ollama_model() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Could not query Ollama models: {exc}") from exc

    models = response.json().get("models", [])
    available = {model.get("name") for model in models}
    if EXPECTED_OLLAMA_MODEL in available:
        print(f"Found Ollama model: {EXPECTED_OLLAMA_MODEL}")
        return True

    return False


def pull_ollama_model() -> None:
    ollama_exe = find_ollama_exe()
    if not ollama_exe:
        raise SystemExit(
            f"Required Ollama model `{EXPECTED_OLLAMA_MODEL}` is missing, and `ollama.exe` could not be found.\n"
            "Add Ollama to PATH or reinstall it, then retry."
        )

    print(f"Pulling Ollama model `{EXPECTED_OLLAMA_MODEL}`...")
    completed = subprocess.run([ollama_exe, "pull", EXPECTED_OLLAMA_MODEL], cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise SystemExit(
            f"Failed to pull `{EXPECTED_OLLAMA_MODEL}`.\n"
            f"You can try manually with:\n\"{ollama_exe}\" pull {EXPECTED_OLLAMA_MODEL}"
        )

    if not ensure_ollama_model():
        raise SystemExit(f"Model pull completed, but `{EXPECTED_OLLAMA_MODEL}` still does not appear in Ollama.")


def ensure_vector_collection() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from LNIAGIA.DB.vector.VectorDBManager import COLLECTION_NAME, _collection_exists, _get_client
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "Could not import the vector DB manager.\n"
            "Make sure the search dependencies are installed.\n"
            f"Details: {exc}"
        ) from exc

    try:
        client = _get_client()
        try:
            exists = _collection_exists(client)
        finally:
            client.close()
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(f"Could not inspect the local vector collection: {exc}") from exc

    if exists:
        print(f"Found local vector collection: {COLLECTION_NAME}")
        return

    raise SystemExit(
        "The integrated search collection has not been built yet.\n"
        "Build it first with the LNIAGIA tooling, then rerun this launcher."
    )


def build_env(args: argparse.Namespace) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["DETECTOR_BACKEND"] = args.detector_backend
    env.setdefault("OLLAMA_MODEL", EXPECTED_OLLAMA_MODEL)
    if args.model_weights:
        env["MODEL_WEIGHTS"] = args.model_weights
    if args.edna_weights:
        env["FASHIONNET_WEIGHTS"] = args.edna_weights
    return env


def run_uvicorn(args: argparse.Namespace, env: dict) -> int:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        command.append("--reload")

    print("\nStarting FashionSense...")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Detector backend: {args.detector_backend}")
    if args.model_weights:
        print(f"  MODEL_WEIGHTS: {args.model_weights}")
    if args.edna_weights:
        print(f"  FASHIONNET_WEIGHTS: {args.edna_weights}")
    print("Press Ctrl+C to stop.\n")

    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
    return completed.returncode


def main() -> int:
    args = parse_args()
    ensure_imports()

    if not args.skip_ollama:
        start_ollama_if_needed()
        if not ensure_ollama_model():
            if args.auto_pull_model:
                pull_ollama_model()
            else:
                ollama_exe = find_ollama_exe()
                if ollama_exe:
                    raise SystemExit(
                        f"Required Ollama model `{EXPECTED_OLLAMA_MODEL}` is not installed.\n"
                        f"Run:\n\"{ollama_exe}\" pull {EXPECTED_OLLAMA_MODEL}\n"
                        "Or rerun this launcher with --auto-pull-model."
                    )
                raise SystemExit(
                    f"Required Ollama model `{EXPECTED_OLLAMA_MODEL}` is not installed.\n"
                    f"Run: ollama pull {EXPECTED_OLLAMA_MODEL}\n"
                    "Or rerun this launcher with --auto-pull-model."
                )
    else:
        print("Skipping Ollama checks.")

    if not args.skip_vector_check:
        ensure_vector_collection()
    else:
        print("Skipping vector collection check.")

    env = build_env(args)
    return run_uvicorn(args, env)


if __name__ == "__main__":
    raise SystemExit(main())
