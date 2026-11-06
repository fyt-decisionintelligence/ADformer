"""Run one full train/validation/test epoch for each bundled dataset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "XIAN-Taxi": ("XIAN", 135, 2928),
    "NYC-Taxi": ("NYC", 263, 17568),
    "NYC-Bike": ("NYC", 263, 17520),
}


def run_dataset(dataset, gpu, exp_id, output_dir, batch_size, epochs=1):
    city, regions, timesteps = DATASETS[dataset]
    log_dir = ROOT / "log" / "ADFormer" / dataset / exp_id
    console_path = output_dir / f"{dataset}_gpu{gpu}_epochs{epochs}.console.log"
    command = [
        sys.executable, "-u", "main.py", "--dataset_city", city,
        "--dataset_name", dataset, "--num_reg", str(regions),
        "--window", "6", "--horizon", "1", "--epochs", str(epochs),
        "--use_early_stop", "False",
        "--batch_size", str(batch_size), "--device", "0", "--exp_id", exp_id,
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    result = {
        "dataset": dataset, "physical_gpu": gpu, "epochs": epochs,
        "command": command, "environment": {k: env[k] for k in (
            "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        )},
        "log_dir": str(log_dir.relative_to(ROOT)),
        "console_log": str(console_path.relative_to(ROOT)),
        "status": "failed",
    }
    started = time.monotonic()
    print(f"Starting {dataset} on physical GPU {gpu}: {console_path}", flush=True)
    status_path = output_dir / f"{dataset}_gpu{gpu}.status.json"
    status_path.write_text(json.dumps({**result, "status": "running"}, indent=2) + "\n")
    try:
        with console_path.open("w") as console:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=console,
                stderr=subprocess.STDOUT, check=False,
            )
        result["exit_code"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"main.py exited with code {completed.returncode}")
        with (log_dir / "args.json").open() as f:
            saved_args = json.load(f)
        assert saved_args["epochs"] == epochs
        checkpoint = log_dir / "model_best_state.pth"
        assert checkpoint.is_file() and checkpoint.stat().st_size > 0
        with np.load(log_dir / "predictions.npz") as outputs:
            prediction, truth = outputs["prediction"], outputs["truth"]
        expected_samples = round((timesteps - 6) * (1 - 0.7 - 0.1))
        assert prediction.shape == truth.shape == (expected_samples, 1, regions, 2)
        assert np.isfinite(prediction).all() and np.isfinite(truth).all()
        mask = truth > saved_args["metric_mask"]
        mape_mask = truth > 5
        errors = prediction[mask] - truth[mask]
        metrics = {
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MAPE_percent": float(np.mean(np.abs(
                (prediction[mape_mask] - truth[mape_mask]) / truth[mape_mask]
            )) * 100),
        }
        assert all(np.isfinite(value) for value in metrics.values())
        run_log = (log_dir / "run.log").read_text()
        losses = re.search(r"Epoch 0, train loss: ([\d.eE+-]+); val loss: ([\d.eE+-]+)", run_log)
        assert losses is not None, "No finite training/validation loss found"
        assert all(np.isfinite(float(value)) for value in losses.groups())
        epoch_losses = re.findall(r"Epoch (\d+), train loss: ([\d.eE+-]+); val loss: ([\d.eE+-]+)", run_log)
        assert [int(row[0]) for row in epoch_losses] == list(range(epochs))
        assert all(np.isfinite(float(value)) for row in epoch_losses for value in row[1:])
        assert "prediction on 1 steps" in run_log
        result.update({
            "status": "passed", "prediction_shape": list(prediction.shape),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "train_loss": float(losses.group(1)), "val_loss": float(losses.group(2)),
            "completed_epochs": len(epoch_losses),
            "final_train_loss": float(epoch_losses[-1][1]),
            "final_val_loss": float(epoch_losses[-1][2]),
            "metrics": metrics,
        })
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", choices=range(4), default=[0, 1, 2],
                        help="Physical GPU IDs; restricted to the authorized GPUs 0-3")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must not contain duplicates")
    exp_id = datetime.now().strftime("smoke-%Y%m%d-%H%M%S-%f")
    output_dir = ROOT / "log" / "smoke" / exp_id
    output_dir.mkdir(parents=True)
    # Each worker owns one GPU; jobs sharing a GPU run sequentially.
    def run_on_gpu(index, gpu):
        return [run_dataset(dataset, gpu, exp_id, output_dir, args.batch_size)
                for dataset in args.datasets[index::len(args.gpus)]]

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(run_on_gpu, index, gpu) for index, gpu in enumerate(args.gpus)]
        results = [result for future in futures for result in future.result()]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": exp_id, "epochs": 1, "window": 6, "horizon": 1,
        "batch_size": args.batch_size, "results": results,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"Summary: {summary_path}", flush=True)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
