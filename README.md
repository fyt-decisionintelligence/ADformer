This repository provides the implementation code for our paper: 

*ADFormer: Aggregation Differential Transformer for Passenger Demand Forecasting*.
![The overall framework of ADFormer](./Overall_framework.jpg)
The following is a detailed description of each folder.
## data
`geo/taxi_zones` contains files representing the geographic information of NYC, which can be used for map visualization. 

Similarly, `xian_hexAddr.txt` contains hexagonal grid addresses of Xi'an, which can be mapped to real-world geographic coordinates. 

Files such as `NYC_Taxi_origin.pkl` and `NYC_Taxi_destination.pkl` are stored as 2D arrays. The "origin/destination" refers to the start and end points of a trip, as the data are obtained from travel orders. The first dimension is the number of spatial units and the second is the number of timesteps. They are used to generate input data in `utils/ADFormer_dataset.py`.
## model
`module.py` implements the essential components required by *ADFormer*.

`ADFormer.py` forms the core of our method, implementing attention mechanisms across both spatial and temporal dimensions, as well as the ST-Encoder.
## utils
`ADFormer_config.py` is the configuration file used in our experiments, including settings for the dataset, model, and training process.

`ADFormer_dataset.py` serves several purposes:

(1) It processes raw data (eg. data/xxx_xxx_origin.pkl), such as adding external information and splitting it according to the window and horizon.

(2) It generates the adjacency, distance, and DTW distance matrices based on the geographic and raw data files.

(3) It aggregates spatial units using *dtw_mx* and produces cluster maps.

`ADFormer_trainer.py` defines the components and procedures for training, validation, and evaluation.

`utils.py` includes utility functions used in the experiments.

## Environment

Run the commands below from the repository root. The pinned environment targets
Python 3.10, Linux x86_64, and an NVIDIA GPU supported by FlashAttention 2
(tested hardware: A800). It uses PyTorch 2.4.1 with CUDA 11.8 and FlashAttention
2.6.3. The CUDA runtime is installed inside the virtual environment; no system
CUDA or driver changes are needed on this machine.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1 setuptools==75.8.0 wheel==0.45.1 numpy==1.26.4 Cython==0.29.37
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
python scripts/check_environment.py --gpu 3
```

NumPy and Cython must be installed before building `fastdtw` (a C/C++ compiler
and Python development headers are needed if no cached wheel is available).
The FlashAttention wheel in `requirements.txt` matches Python 3.10 / PyTorch
2.4 / CUDA 11.8 / the PyTorch C++ ABI; do not install a wheel for a different
combination. H3 is pinned to 3.7.7 because the code uses its 3.x API.
Package sources: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/),
[FlashAttention 2.6.3](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.6.3),
[H3 3.7.7](https://pypi.org/project/h3/3.7.7/).
`requirements.lock.txt` records all installed package versions and the official
FlashAttention wheel hash. After the bootstrap command, use that file instead
of `requirements.txt` to recreate this exact environment.

## Verify all three datasets

The repository includes both origin and destination files for every dataset:

| Dataset | City | Regions (`--num_reg`) | Half-hour time steps |
| --- | --- | ---: | ---: |
| XIAN-Taxi | XIAN | 135 | 2928 |
| NYC-Taxi | NYC | 263 | 17568 |
| NYC-Bike | NYC | 263 | 17520 |

```bash
source .venv/bin/activate
python scripts/smoke_test.py --gpus 0 1 2
```

This runs the three datasets concurrently on physical GPUs 0, 1, and 2,
respectively. Each run uses all samples, the default model, a batch size of 64,
`window=6`, `horizon=1`, and one complete training epoch, followed by validation,
checkpoint reload, and evaluation over the entire test split. It checks finite
losses and predictions, prediction shapes, and saved checkpoints, then writes
per-dataset logs and `summary.json` under `log/smoke/<experiment_id>/`.
Model checkpoints, arguments, and predictions are in
`log/ADFormer/<dataset>/<experiment_id>/`.

The verification scripts restrict physical GPUs to 0-3. GPU 3 is used for the
environment check; they do not use or manage GPUs 4-7. To repeat one dataset:

```bash
python scripts/smoke_test.py --datasets NYC-Bike --gpus 2
```

One epoch verifies execution, not convergence or the paper's reported accuracy.
The first run creates geographic, DTW, and split-data caches in `cache/`.
Delete the relevant generated caches before changing split ratios, time
intervals, or external features; those settings are not part of the cache keys.

## Full training on GPUs 0, 1, and 2

```bash
source .venv/bin/activate
python -u scripts/train_all.py --gpus 0 1 2 --epochs 120
```

Add `--detach` to keep the jobs running in the background and save the
supervisor output to `log/full_train/<run_id>.launcher.log`. The command prints
the supervisor PID and all log locations. By default, each GPU first runs a
one-epoch check on its dataset; a passing check starts that dataset's full run
automatically. Check logs are kept in `log/full_train/<run_id>/preflight/`.
Use `--skip-smoke` only if the environment and datasets have already been checked.

This assigns XIAN-Taxi to GPU 0, NYC-Taxi to GPU 1, and NYC-Bike to GPU 2.
It runs 120 epochs with early stopping disabled, then reloads each best
checkpoint and evaluates the entire test set. Other model settings remain at
their defaults (`window=6`, `horizon=1`, `batch_size=64`).

Each run uses an experiment name such as
`full_20261103_210000_NYC-Taxi_gpu1_w6_h1_epochs120_bs64`.
The group directory `log/full_train/<run_id>/` contains `manifest.json`,
`<dataset>_gpu<N>_epochs120.console.log`, per-dataset status JSON files,
and a final `summary.json`. Each dataset's directory under
`log/ADFormer/<dataset>/<experiment_name>/` contains:

| File | Content |
| --- | --- |
| `args.json` | Complete training configuration |
| `run.log` | Timestamped training and evaluation log |
| `training_history.csv` | Losses, learning rate, and timings for every epoch |
| `training_summary.json` | Completed epochs and best validation epoch |
| `model_best_state.pth` | Model with the lowest validation loss |
| `model_last_state.pth` | Model after the final training epoch |
| `predictions.npz` | Best model's test predictions and ground truth |
| `metrics.json` | Test MAE, RMSE, and MAPE |

## Training a single dataset
You can use the following command to train a model:
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python main.py --device 0 --dataset_city XIAN --dataset_name XIAN-Taxi --num_reg 135 --window 6 --horizon 1
```
For NYC-Taxi or NYC-Bike, use `--dataset_city NYC --num_reg 263` and the
corresponding `--dataset_name`. `--device` is the logical GPU index after
`CUDA_VISIBLE_DEVICES` is applied, so it is `0` when only one GPU is visible.
Omitting `--epochs` uses the original 120-epoch training setting.
For other forecast horizons, replace `--horizon 1` with `--horizon 3`
(90 minutes) or `--horizon 6` (180 minutes).
The trained model and results will be saved in the `log` folder.

## Completed experiments and final checkpoints

All nine dataset/horizon combinations completed 120 epochs. The final epoch-120
results, trained weights, predictions, full training logs and per-epoch records
are archived under [results/](results/README.md). See the
[run report](RUN_REPORT.md) and [final results CSV](results/final_epoch120.csv).
`metrics_last.json` reports the final checkpoint; `metrics_best.json` reports
the checkpoint selected by validation loss.

Please note: Since reverse transformation may turn zeros into near-zero values (due to normalization using training data only), MAPE can be distorted. Thus, we compute MAPE only for values greater than 5.
