"""
Runs the fair baseline comparison: every model x every seed, one after another, with
identical settings, then builds the summary tables.

Safe to stop and restart at any time:
  - finished runs (results/runs/<model>_seed<N>/summary.json exists) are skipped
  - an unfinished run continues from its last completed epoch (--resume)

    python scripts/run_baselines.py                    # 3 seeds x 2 models, 30 epochs
    python scripts/run_baselines.py --seeds 0 --epochs 30
"""
import os
import sys
import argparse
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["unet", "car_unet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    for seed in args.seeds:
        for model in args.models:
            run_name = f"{model}_seed{seed}"
            if os.path.exists(os.path.join(BASE_DIR, "results", "runs", run_name, "summary.json")):
                print(f"[skip] {run_name} already finished", flush=True)
                continue
            print(f"\n[run] {run_name}", flush=True)
            cmd = [sys.executable, os.path.join(BASE_DIR, "scripts", "train_fives.py"),
                   "--model", model, "--seed", str(seed), "--epochs", str(args.epochs), "--resume"]
            result = subprocess.run(cmd, cwd=BASE_DIR)
            if result.returncode != 0:
                sys.exit(f"[error] {run_name} failed with exit code {result.returncode}; rerun this script to resume.")

    subprocess.run([sys.executable, os.path.join(BASE_DIR, "scripts", "summarize_results.py")], cwd=BASE_DIR, check=True)


if __name__ == "__main__":
    main()
