# ADFormer 环境、完整训练与最终测试结果

三个数据集在 30、90、180 分钟预测时长下的 9 组实验均已完成 120 轮训练及测试。
输入窗口为 6，batch size 为 64，关闭提前停止。
以下结果来自第 120 轮最终权重 `model_last_state.pth` 在原测试集上的重新评估。

| 数据集 | 预测时长 | horizon | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| XIAN-Taxi | 30min | 1 | 3.5376 | 5.1770 | 22.6846 |
| XIAN-Taxi | 90min | 3 | 3.7411 | 5.5274 | 22.7557 |
| XIAN-Taxi | 180min | 6 | 5.4568 | 10.0602 | 27.5813 |
| NYC-Taxi | 30min | 1 | 5.5000 | 11.4017 | 18.4514 |
| NYC-Taxi | 90min | 3 | 6.1837 | 13.6684 | 21.1077 |
| NYC-Taxi | 180min | 6 | 6.8399 | 15.1198 | 21.9945 |
| NYC-Bike | 30min | 1 | 3.2821 | 5.0882 | 24.4572 |
| NYC-Bike | 90min | 3 | 3.5476 | 5.6665 | 26.1921 |
| NYC-Bike | 180min | 6 | 4.0496 | 6.7687 | 32.3891 |

MAE、RMSE 统计真实值 >0.5 的位置；MAPE 统计真实值 >5 的位置。
多步结果汇总全部预测步，不只取最后一步。

## 环境验证

项目使用 Python 3.10.12、PyTorch 2.4.1+cu118、FlashAttention 2.6.3。
`pip check`、FlashAttention GPU 前向/反向、H3 3.x 接口、fastdtw 编译扩展及 NYC 地理文件读取均通过。
依赖版本与安装方法见 [README](README.md) 和 [requirements.lock.txt](requirements.lock.txt)。
三个数据集均通过了一轮完整训练、验证、权重重载及完整测试集的运行检查。

## 保存的实验材料

- [最终结果 CSV](results/final_epoch120.csv)
- [最终评估详情及权重 SHA256](results/final_epoch120.json)
- [环境版本记录](results/environment.json)
- [实验归档](results/README.md)：9 组配置、完整训练日志、逐轮 CSV、模型权重、预测及指标。

每个实验位于 `results/<数据集>/h<horizon>/`。`model_last_state.pth` 是最终轮权重，
`metrics_last.json` 和 `predictions_last.npz` 是该权重的测试结果。
验证集最优权重及其原评估另保存为 `model_best_state.pth`、`metrics_best.json` 和 `predictions_best.npz`。

## 修复与复现

- 修复西安数据文件路径大小写，首次运行自动创建缓存目录。
- 地理缓存采用原子替换写入，支持两个纽约任务并行初始化。
- 使用数据集、GPU、参数和时间组成独立运行目录名。
- 保存逐轮 CSV、最终模型、训练汇总及 JSON 评估指标，并检测非有限损失。
- `scripts/train_all.py` 可同时训练三个数据集的 h=1 实验；其他 horizon 的运行命令见 [README](README.md)。
- `scripts/evaluate_last.py` 重新评估本机 `log/ADFormer` 下已完成的九组最终权重，并保存独立结果。

训练及环境检查只使用了授权的 GPU 0–3。
