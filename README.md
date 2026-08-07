# rsna-knee

面向 RSNA knee MRI study-level 多标签分类的可扩展工程骨架，包括：

- 对 570GB、约 82 万 DICOM 文件进行流式、分片、可恢复的 audit；
- 从 DICOM 几何构建 study-level 2.5D window manifest；
- 配置驱动地切换 backbone、selector、aggregator 和 loss；
- 支持固定预算 Top-K、Knee-BCRS coverage loss、AMP 和 `torchrun` DDP；
- 生成可追踪的消融配置矩阵。

详细审计规范见 [RSNA-Knee-Image-Audit-Plan.md](RSNA-Knee-Image-Audit-Plan.md)，总体实验顺序见 [RSNA-Knee-Efficiency-Experiment-Plan.md](RSNA-Knee-Efficiency-Experiment-Plan.md)。

## 1. 安装

```bash
python -m pip install -e ".[train,dicom,viz,dev]"
```

若只跑不含压缩 DICOM 的单元测试，可安装基础依赖。JPEG Lossless/JPEG 2000 数据建议安装 `dicom` extra，并在正式 audit 前用 Smoke Test 验证 decoder。

## 2. Data audit

训练集和测试集使用不同的 audit 输出目录。以下以训练集为例：

```bash
python -m rsna_knee.audit.cli index \
  --dicom-root /data/train_series \
  --output artifacts/audit/train \
  --num-shards 128

python -m rsna_knee.audit.cli headers \
  --audit-root artifacts/audit/train \
  --workers 8

python -m rsna_knee.audit.cli pixels \
  --audit-root artifacts/audit/train \
  --workers 8 \
  --deep-sample-rate 0.10 \
  --hash-pixels

python -m rsna_knee.audit.cli summarize \
  --audit-root artifacts/audit/train \
  --train-csv /data/train.csv \
  --train-series-csv /data/train_series.csv
```

每个阶段按稳定 hash 分为 128 个 shard。已完成的 `part-xxxxx.jsonl` 会自动跳过；可以用 `--shards 0-15` 在不同机器上并行，或用 `--force` 重建指定 shard。

`artifacts/audit/<split>/private/patient_salt.txt` 仅用于 patient hash，不能上传为公开数据集或提交到 Git。

建议先运行少量 shard：

```bash
python -m rsna_knee.audit.cli headers --audit-root artifacts/audit/train --shards 0 --workers 4
python -m rsna_knee.audit.cli pixels --audit-root artifacts/audit/train --shards 0 --workers 4
```

## 3. 构建 2.5D study manifest

```bash
python -m rsna_knee.data.manifest \
  --audit-root artifacts/audit/train \
  --output artifacts/manifests/all_train.jsonl \
  --labels-csv /data/train.csv \
  --series-csv /data/train_series.csv \
  --max-windows-per-series 25 \
  --neighbor-offsets=-1,0,1
```

manifest 只保存相对路径、几何位置、series metadata 和标签，不复制 DICOM 像素。训练/验证 fold 应在 patient/duplicate-safe split 完成后分别输出 manifest；当前命令生成的是全量基础 manifest。

生成 patient-grouped multilabel folds：

```bash
python -m rsna_knee.data.split \
  --manifest artifacts/manifests/all_train.jsonl \
  --output-dir artifacts/manifests/folds \
  --folds 5
```

生成确定性 study montages：

```bash
python -m rsna_knee.audit.montage \
  --manifest artifacts/manifests/all_train.jsonl \
  --dicom-root /data/train_series \
  --output-dir artifacts/audit/train/montages/random \
  --max-studies 20
```

## 4. 配置组合

配置文件从左到右递归合并，后者覆盖前者：

```bash
python -m rsna_knee.train \
  --config configs/train/base.yaml \
  --config configs/backbone/dinov2_small_timm.yaml \
  --config configs/selector/uniform_k15.yaml \
  --config configs/aggregator/per_label_query.yaml \
  --set data.train.dicom_root=/data/train_series \
  --set data.valid.dicom_root=/data/train_series \
  --set optimizer.lr=0.0001
```

可切换组件：

| 组件 | 内置选项 |
|---|---|
| Backbone | `tiny_cnn`、`timm`、`huggingface`、`external` |
| Selector | `uniform`、`central`、`learned_topk`、`recall_safe_topk` |
| Aggregator | `mean_max`、`attention`、`per_label_query` |
| Loss | masked BCE、batch ranking、recall-safe coverage |

`external` adapter 用于 DINOv3、MedNeXt 或用户提供的源码，需求见 [MODEL-SOURCES.md](MODEL-SOURCES.md)。

## 5. 多 GPU 与实验矩阵

```bash
torchrun --nproc_per_node=2 -m rsna_knee.train \
  --config configs/train/base.yaml \
  --config configs/backbone/dinov2_small_timm.yaml

python -m rsna_knee.sweep \
  --base configs/train/base.yaml \
  --matrix configs/sweeps/backbone_selector.yaml \
  --output artifacts/sweeps/backbone_selector
```

Sweep 会输出每个实验的 resolved YAML 和 `queue.json`。每个 run 目录保存 resolved config、逐 epoch 指标、`last.pt` 与最佳 Macro AUC checkpoint。

## 6. 当前边界

- 当前仓库未挂载比赛 DICOM，因此只用合成 DICOM/张量验证框架。
- 正式 fold 生成、报告文本标签抽取和 teacher evidence 生成将在 audit 结果可用后实现。
- DINOv3、MedNeXt 及原始 ESOD/BCRS 代码需在提供固定源码版本后适配。
