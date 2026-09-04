"""Summarize repeated complete runs with mean, SD, and a 95% t interval."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.run_utils import write_json


def nested_metric(record, key):
    value = record
    for part in key.split("."):
        value = value[part]
    return float(value)


def summarize(paths, metric):
    values = []
    runs = []
    for path in paths:
        path = Path(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        value = nested_metric(record, metric)
        values.append(value)
        runs.append({"metrics_path": str(path), "value": value})
    if not values:
        raise ValueError("No metrics files were provided")
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1)) if len(array) > 1 else None
    if len(array) > 1:
        half_width = float(t.ppf(0.975, len(array) - 1) * sample_sd / np.sqrt(len(array)))
        interval = [mean - half_width, mean + half_width]
    else:
        interval = None
    return {
        "metric": metric,
        "n": len(values),
        "mean": mean,
        "sample_sd": sample_sd,
        "confidence_interval_95": interval,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="*", help="Explicit metrics.json files")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--metric", default="hybrid.hybrid_test_accuracy")
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = [Path(path) for path in args.metrics]
    if not paths:
        paths = sorted(Path(args.runs_root).glob("*/metrics.json"))
    result = summarize(paths, args.metric)
    print("Individual values:", ", ".join(f"{run['value']:.4f}" for run in result["runs"]))
    print(f"n={result['n']} mean={result['mean']:.4f}")
    if result["sample_sd"] is None:
        print("sample SD and 95% CI require at least two complete runs")
    else:
        low, high = result["confidence_interval_95"]
        print(f"sample_sd={result['sample_sd']:.4f} 95% CI=[{low:.4f}, {high:.4f}]")
    if args.output:
        write_json(args.output, result, force=args.force)


if __name__ == "__main__":
    main()
