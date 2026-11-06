# Completed experiments

All nine experiments completed 120 epochs with early stopping disabled,
window=6 and batch_size=64. Each time step is 30 minutes:
`h1` = 30 minutes, `h3` = 90 minutes, `h6` = 180 minutes.

The reported final results use `model_last_state.pth` from epoch 120.
See [final_epoch120.csv](final_epoch120.csv), [final_epoch120.json](final_epoch120.json)
and the [run report](../RUN_REPORT.md).

Each `<dataset>/h<horizon>/` directory contains:

| File | Meaning |
| --- | --- |
| `args.json` | Original training configuration |
| `training_history.csv` | Every epoch's losses, learning rate and timings |
| `training_summary.json` | Completed epochs and best validation epoch |
| `run.log` | Full training log and original best-checkpoint evaluation |
| `model_last_state.pth` | Final epoch-120 state dictionary |
| `metrics_last.json`, `predictions_last.npz` | Final checkpoint's test results |
| `evaluation_last.log`, `evaluation_last.json` | Final evaluation log and checkpoint provenance |
| `model_best_state.pth` | Best validation checkpoint |
| `metrics_best.json`, `predictions_best.npz` | Best checkpoint's original test results |

Prediction archives contain `prediction` and `truth` arrays. MAE/RMSE use
truth >0.5; MAPE uses truth >5. Multi-step metrics aggregate all predicted steps.
Historical paths in configuration/provenance files refer to the original local
`log/` directories; the corresponding archived files are in this directory.

Virtual environments, generated preprocessing caches and Python bytecode are
excluded from Git. Installation instructions and pinned dependencies are in
the repository root. Original local `log/` outputs remain available locally.
