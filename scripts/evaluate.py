#!/usr/bin/env python3
"""Run fixed development evaluations and emit one machine-readable metric."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any


METRIC_PATTERN = re.compile(
    r"^\s*\*\s+(?P<name>[A-Z0-9_]+)\s+Development Accuracy:\s+"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)%\s*$",
    re.IGNORECASE,
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_metric(log_path: Path, metric: str) -> float:
    values: list[float] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_PATTERN.match(line)
            if match and match.group("name").casefold() == metric.casefold():
                values.append(float(match.group("value")))
    if not values:
        raise ValueError(f"No final development metric {metric!r} found in {log_path}")
    value = values[-1]
    if not math.isfinite(value):
        raise ValueError(f"Metric {metric!r} is not finite")
    return value


def run_seed(
    repo: Path,
    train_script: Path,
    log_dir: Path,
    metric: str,
    seed: int,
    epochs: int,
) -> dict[str, Any]:
    iteration = os.environ.get("LOOP_ITERATION", "manual")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"loop-{iteration}-{timestamp}-seed-{seed}"
    log_path = log_dir / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [str(train_script), str(seed), str(epochs), run_id]
    started = dt.datetime.now(dt.timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            sys.stderr.write(line)
            sys.stderr.flush()
        returncode = process.wait()
    duration = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
    if returncode != 0:
        raise RuntimeError(f"Training seed {seed} exited with status {returncode}; see {log_path}")
    return {
        "seed": seed,
        "value": parse_metric(log_path, metric),
        "duration_seconds": round(duration, 3),
        "run_id": run_id,
        "log": str(log_path.relative_to(repo)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="agent/latest_metrics.json")
    parser.add_argument("--metric", default="ensemble_3_k25")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train-script", default="scripts/train.sh")
    parser.add_argument("--log-dir", default="agent/evaluation")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    output = output if output.is_absolute() else repo / output
    train_script = Path(args.train_script)
    train_script = train_script if train_script.is_absolute() else repo / train_script
    log_dir = Path(args.log_dir)
    log_dir = log_dir if log_dir.is_absolute() else repo / log_dir
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "valid": False,
        "metric": args.metric,
        "metric_value": None,
        "direction": "maximize",
        "unit": "percent",
        "aggregation": "mean",
        "epochs": args.epochs,
        "seeds": args.seeds,
        "per_seed": [],
    }
    try:
        if args.epochs <= 0:
            raise ValueError("--epochs must be greater than zero")
        if not args.seeds:
            raise ValueError("At least one seed is required")
        if not train_script.is_file():
            raise FileNotFoundError(f"Training script not found: {train_script}")
        for seed in args.seeds:
            result["per_seed"].append(
                run_seed(repo, train_script, log_dir, args.metric, seed, args.epochs)
            )
        result["metric_value"] = statistics.fmean(
            item["value"] for item in result["per_seed"]
        )
        result["valid"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
        atomic_write_json(output, result)
        print(json.dumps(result, sort_keys=True))
        return 1

    atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
