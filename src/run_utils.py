"""Utilities for isolated, non-overwriting experiment runs."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "runs"


def make_run_id(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{timestamp}"


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "run ID must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return run_id


def prepare_run_dir(
    prefix: str,
    run_id: str | None = None,
    runs_root: str | os.PathLike[str] = DEFAULT_RUNS_ROOT,
    force: bool = False,
) -> Path:
    """Create an isolated run directory and reject accidental reuse."""
    selected = validate_run_id(run_id or make_run_id(prefix))
    run_dir = Path(runs_root).expanduser().resolve() / selected
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Run directory already contains files: {run_dir}. "
            "Choose another --run-id or pass --force to overwrite outputs."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def guard_output(
    path: str | os.PathLike[str], *, force: bool = False, reuse: bool = False
) -> bool:
    """Return True when an output should be written, False when explicitly reused."""
    output = Path(path)
    if output.exists():
        if reuse:
            return False
        if not force:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output}. Pass --force "
                "to replace it or --reuse where supported."
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    return True


def save_torch_state(model: Any, path: str | os.PathLike[str], *, force: bool = False) -> Path:
    """Atomically save a model state dict after enforcing overwrite policy."""
    import torch

    output = Path(path)
    guard_output(output, force=force)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    os.replace(temporary, output)
    return output


def write_json(path: str | os.PathLike[str], value: Any, *, force: bool = False) -> Path:
    output = Path(path)
    guard_output(output, force=force)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def base_manifest(run_id: str, command: list[str], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "command": command,
        "config": config,
    }
