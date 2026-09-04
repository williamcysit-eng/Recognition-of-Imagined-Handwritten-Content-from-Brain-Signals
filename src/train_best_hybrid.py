"""One-command training workflow for the OOF/full-refit hybrid."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.evaluate_hybrid_refit_oof import evaluate_hybrid
from src.evaluate_oof_multiarch_ensemble import evaluate_oof
from src.experiment_full_refit import run_full_refit
from src.run_utils import base_manifest, prepare_run_dir, write_json


def training_commands(python, data_path, checkpoint_root, quick=0, cpu=False, force=False):
    common = ["--data-path", str(data_path), "--output-root", str(checkpoint_root)]
    if cpu:
        common.append("--cpu")
    if force:
        common.append("--force")

    def epochs(default):
        return str(quick or default)

    return [
        [python, str(ROOT / "src" / "train_oof_dcn.py"), "--folds", "5", "--epochs", epochs(100), "--seed", "42", *common],
        [python, str(ROOT / "src" / "train_oof_dcn.py"), "--folds", "5", "--epochs", epochs(90), "--seed", "542", *common],
        [python, str(ROOT / "src" / "train_oof_eegnet.py"), "--folds", "5", "--epochs", epochs(80), "--seed", "142", "--kernel", "25", *common],
        [python, str(ROOT / "src" / "train_oof_eegnet.py"), "--folds", "5", "--epochs", epochs(80), "--seed", "342", "--kernel", "15", "--swa", *common],
        [python, str(ROOT / "src" / "train_oof_eegnet.py"), "--folds", "5", "--epochs", epochs(75), "--seed", "642", "--kernel", "15", "--swa", *common],
        [python, str(ROOT / "src" / "train_oof_graph.py"), "--epochs", epochs(60), "--seed", "242", *common],
        [python, str(ROOT / "src" / "train_oof_aligned_dcn.py"), "--epochs", epochs(80), "--seed", "442", *common],
    ]


def checkpoint_inventory(checkpoint_root):
    records = []
    for path in sorted(Path(checkpoint_root).rglob("*.pth")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {
                "path": str(path.relative_to(checkpoint_root)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Run directory name (default: timestamped)")
    parser.add_argument("--runs-root", default=ROOT / "runs")
    parser.add_argument("--data-path", default=ROOT / "data" / "processed" / "eeg_dataset.npz")
    parser.add_argument("--quick", type=int, default=0, help="Use N epochs in every training stage")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing run's outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete command plan without training")
    args = parser.parse_args()
    if args.quick < 0:
        parser.error("--quick cannot be negative")
    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        parser.error(f"dataset not found: {data_path}; run src/extract.py first")

    run_dir = prepare_run_dir("best-hybrid", args.run_id, args.runs_root, args.force)
    checkpoint_root = run_dir / "checkpoints"
    output_dir = run_dir / "outputs"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "oof_multiarch_logits.npz"
    commands = training_commands(
        sys.executable,
        data_path,
        checkpoint_root,
        quick=args.quick,
        cpu=args.cpu,
        force=args.force,
    )
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config = base_manifest(run_dir.name, sys.argv, serializable_args)
    config["stages"] = commands
    write_json(run_dir / "config.json", config, force=args.force)

    print(f"Run directory: {run_dir}")
    if args.dry_run:
        for index, command in enumerate(commands, 1):
            print(f"[{index}/{len(commands)}] {' '.join(command)}")
        return

    for index, command in enumerate(commands, 1):
        print(f"\n{'=' * 78}\nOOF TRAINING STAGE {index}/{len(commands)}\n{'=' * 78}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    print(f"\n{'=' * 78}\nOOF SELECTION AND CACHE\n{'=' * 78}", flush=True)
    oof_result = evaluate_oof(
        checkpoint_root,
        data_path,
        cache_path,
        force=args.force,
        force_cpu=args.cpu,
    )

    print(f"\n{'=' * 78}\nFULL-DEVELOPMENT REFIT\n{'=' * 78}", flush=True)
    refit_result = run_full_refit(
        checkpoint_root,
        data_path,
        force=args.force,
        force_cpu=args.cpu,
        dcn_epochs=args.quick or 40,
        k25_epochs=args.quick or 40,
        k15_epochs=args.quick or 60,
        graph_epochs=args.quick or 40,
        ensemble_weights=oof_result["weights"],
        ensemble_scales=oof_result["scales"],
    )

    print(f"\n{'=' * 78}\nFIXED 50/50 HYBRID\n{'=' * 78}", flush=True)
    hybrid_result = evaluate_hybrid(
        cache_path,
        checkpoint_root,
        data_path,
        force_cpu=args.cpu,
    )

    metrics = {"oof": oof_result, "refit": refit_result, "hybrid": hybrid_result}
    write_json(run_dir / "metrics.json", metrics, force=args.force)
    write_json(
        run_dir / "checkpoints.json",
        {"checkpoints": checkpoint_inventory(checkpoint_root)},
        force=args.force,
    )
    print(f"\nCompleted best-hybrid run: {run_dir}")


if __name__ == "__main__":
    main()
