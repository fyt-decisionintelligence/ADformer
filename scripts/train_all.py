"""Train the three datasets for 120 full epochs on physical GPUs 0, 1, and 2."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from smoke_test import DATASETS, ROOT, run_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=3, choices=range(4), default=[0, 1, 2],
                        help="GPUs for XIAN-Taxi, NYC-Taxi, NYC-Bike, in that order")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id", default=datetime.now().strftime("full_%Y%m%d_%H%M%S"))
    parser.add_argument("--detach", action="store_true", help="Launch a durable background supervisor")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the one-epoch check before training")
    args = parser.parse_args()
    if len(set(args.gpus)) != 3:
        parser.error("Use three distinct GPUs")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Epochs and batch size must be positive")
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        parser.error("--run-id must be a directory name")
    output_dir = ROOT / "log" / "full_train" / args.run_id
    if output_dir.exists():
        parser.error(f"Run directory already exists: {output_dir}")
    if args.detach:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        launcher_log = output_dir.parent / f"{args.run_id}.launcher.log"
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--gpus',
                   *map(str, args.gpus), '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size), '--run-id', args.run_id]
        if args.skip_smoke:
            command.append('--skip-smoke')
        env = os.environ.copy()
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with launcher_log.open('x') as console:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {'pid': process.pid, 'run_id': args.run_id, 'launcher_log': str(launcher_log),
                  'run_directory': str(output_dir), 'command': command}
        (output_dir.parent / f'{args.run_id}.launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        print(json.dumps(launch, indent=2), flush=True)
        return 0
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": args.run_id, "started_at": datetime.now().astimezone().isoformat(),
        "supervisor_pid": os.getpid(), "epochs": args.epochs,
        "batch_size": args.batch_size, "window": 6, "horizon": 1,
        "early_stopping": False, "preflight": not args.skip_smoke, "jobs": [],
    }
    for dataset, gpu in zip(DATASETS, args.gpus):
        exp_id = f"{args.run_id}_{dataset}_gpu{gpu}_w6_h1_epochs{args.epochs}_bs{args.batch_size}"
        manifest["jobs"].append({
            "dataset": dataset, "physical_gpu": gpu, "experiment_id": exp_id,
            "artifacts": f"log/ADFormer/{dataset}/{exp_id}",
            "console_log": str((output_dir / f"{dataset}_gpu{gpu}_epochs{args.epochs}.console.log").relative_to(ROOT)),
        })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    preflight_dir = output_dir / 'preflight'
    if not args.skip_smoke:
        preflight_dir.mkdir()

    def train_job(job):
        if not args.skip_smoke:
            preflight = run_dataset(job['dataset'], job['physical_gpu'],
                                    'preflight_' + job['experiment_id'], preflight_dir,
                                    args.batch_size, epochs=1)
            if preflight['status'] != 'passed':
                preflight['error'] = 'Preflight failed; full training was not started. ' + preflight.get('error', '')
                return preflight
        return run_dataset(job['dataset'], job['physical_gpu'], job['experiment_id'],
                           output_dir, args.batch_size, args.epochs)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(train_job, job) for job in manifest['jobs']]
        results = [future.result() for future in futures]
    report = {**manifest, "finished_at": datetime.now().astimezone().isoformat(), "results": results}
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    return 0 if all(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
