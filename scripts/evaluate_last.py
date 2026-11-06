"""Evaluate epoch-120 checkpoints on the original test splits."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['XIAN-Taxi', 'NYC-Taxi', 'NYC-Bike']


def evaluate_dataset(dataset_name, gpu, batch_id):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from utils.ADFormer_dataset import ADFormerDataset
    from utils.ADFormer_trainer import ADFormerTrainer
    from model.ADFormer import ADFormer
    from utils.utils import get_logger

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU evaluation is required')
    results = []
    directories = {}
    for directory in sorted((ROOT / 'log' / 'ADFormer' / dataset_name).glob('full_*')):
        saved = json.loads((directory / 'args.json').read_text())
        if saved['horizon'] in (1, 3, 6):
            if saved['horizon'] in directories:
                raise ValueError(f'Ambiguous runs for {dataset_name}, horizon {saved["horizon"]}')
            directories[saved['horizon']] = directory
    assert set(directories) == {1, 3, 6}
    for horizon, directory in sorted(directories.items()):
        saved = json.loads((directory / 'args.json').read_text())
        summary = json.loads((directory / 'training_summary.json').read_text())
        assert summary['completed_epochs'] == saved['epochs'] == 120
        checkpoint = directory / 'model_last_state.pth'
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        output = directory / 'last_epoch_test' / batch_id
        output.mkdir(parents=True, exist_ok=False)
        (output / 'args.json').write_text(json.dumps(saved, indent=2) + '\n')
        args = argparse.Namespace(**saved)
        args.device = torch.device('cuda:0')
        args.log_dir = str(output)
        logger = get_logger(args.log_dir)
        logger.info(f'Evaluating FINAL epoch 120 checkpoint: {checkpoint}; physical GPU {gpu}')
        dataset = ADFormerDataset(args, logger)
        train, validation, test = dataset.get_data()
        model = ADFormer(args, dataset.get_dataset_feature(), logger)
        trainer = ADFormerTrainer(args, model, logger)
        state = torch.load(checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
        trainer.evaluate(test)
        metrics = json.loads((output / 'metrics.json').read_text())
        with np.load(output / 'predictions.npz') as prediction:
            truth, values = prediction['truth'], prediction['prediction']
            assert truth.shape == values.shape == (len(test.dataset), horizon, saved['num_reg'], 2)
            assert np.isfinite(truth).all() and np.isfinite(values).all()
        assert all(np.isfinite(metrics[key]) for key in ['MAE', 'RMSE', 'MAPE_percent'])
        result = {
            'dataset': dataset_name, 'horizon': horizon,
            'prediction_minutes': horizon * saved['time_interval'] // 60,
            'epoch': 120, 'checkpoint': str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256': checkpoint_hash, 'physical_gpu': gpu,
            'output_directory': str(output.relative_to(ROOT)), **metrics,
        }
        (output / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
        logger.info('Final-checkpoint evaluation and finite-prediction checks passed.')
        del trainer, model, dataset, train, validation, test, truth, values
        torch.cuda.empty_cache()
    report = ROOT / 'log' / 'last_epoch_evaluation' / batch_id / f'{dataset_name}.json'
    report.write_text(json.dumps(results, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--gpu', type=int, choices=range(4))
    parser.add_argument('--batch-id', default=datetime.now().strftime('last_epoch120_%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    if args.dataset:
        if args.gpu is None:
            parser.error('--dataset requires --gpu')
        evaluate_dataset(args.dataset, args.gpu, args.batch_id)
        return
    target = ROOT / 'log' / 'last_epoch_evaluation' / args.batch_id
    target.mkdir(parents=True, exist_ok=False)

    def run(dataset, gpu):
        env = os.environ.copy()
        env.update({'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4',
                    'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                    'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(gpu)})
        command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                   '--dataset', dataset, '--gpu', str(gpu), '--batch-id', args.batch_id]
        console = target / f'{dataset}_gpu{gpu}.console.log'
        print(f'Evaluating {dataset}, GPU {gpu}: {console}', flush=True)
        with console.open('w') as f:
            subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        return json.loads((target / f'{dataset}.json').read_text())

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, dataset, gpu) for gpu, dataset in enumerate(DATASETS)]
        results = [item for future in futures for item in future.result()]
    (target / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    with (target / 'results.csv').open('w', newline='') as f:
        fields = ['dataset', 'prediction_minutes', 'horizon', 'epoch', 'MAE', 'RMSE', 'MAPE_percent']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2), flush=True)
    print(f'Saved results: {target}', flush=True)


if __name__ == '__main__':
    main()
