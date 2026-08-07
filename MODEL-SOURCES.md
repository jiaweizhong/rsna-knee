# External Model Source Checklist

当前框架已经内置通用 `timm`、Hugging Face 和 `external factory` 适配器。以下模型是否需要你下载源码，取决于我们希望复现到什么程度。

## 现在不需要额外源码

只要 Kaggle 环境中安装并离线挂载相应 wheel/checkpoint，下列模型可直接通过 `timm` adapter 使用：

- DINOv2 Small/Base/Large；
- ConvNeXt Tiny/Small；
- PVTv2-B0/B1；
- ResNet、EfficientNet、Swin 等常规分类 backbone。

正式运行前会用实际安装的 `timm.list_models()` 校验精确 model name，避免版本差异。

## 建议你提供的源码

### 1. Meta DINOv3

需要：

- 官方 repo 的固定 commit 或 release；
- 计划测试的 Small/Base checkpoint；
- 官方 preprocessing/normalization 定义；
- checkpoint license 和 Kaggle redistribution 条件。

原因：不同发行方式的模型构造函数和 state-dict key 可能不同。框架已经预留：

```yaml
model:
  backbone:
    name: external
    params:
      factory: third_party.dinov3_adapter.build_dinov3_vits16
      out_dim: 384
      checkpoint_path: /path/to/checkpoint.pth
```

### 2. MedNeXt-lightweight / MedNeXtV2

需要：

- encoder 或 classification 版本源码；
- 模型构造参数；
- 输出 feature map 的层级和 channel 数；
- 预训练 checkpoint（如果有）；
- 输入 spacing/resampling 要求。

原因：它是 3D 路线，不能仅把 segmentation decoder 删除后假定表征仍合理。需要做显式 encoder adapter 和真实 T4 内存测试。

### 3. 你的 ESOD/BCRS 两项优化实现

最好提供：

- selector/scorer 模块；
- spectral branch；
- coverage/risk loss；
- fixed-budget Top-K 逻辑；
- 参数匹配的 ordinary-conv control；
- 当前使用的配置与 checkpoint（若有）。

原因：框架内置的是面向 knee MRI 的 clean-room `recall_safe_topk` 基线，便于先跑通消融；如果需要忠实验证你的优化，应直接适配原实现，避免实现细节偏差。

## 外部模型的最小接口

源码放在 `third_party/<model_name>/`，再写一个轻量 adapter：

```python
def build_model_variant(**kwargs) -> torch.nn.Module:
    ...
```

模型 forward 可以返回：

- `[B, D]`；
- `[B, tokens, D]`；
- `[B, D, H, W]`；
- `[B, D, Z, H, W]`。

框架会统一池化到 `[B, D]`。同时需要明确提供：

- `out_dim`；
- `spatial_dims`，2D 为 `2`，3D 为 `3`；
- checkpoint key；
- 是否允许 `strict=False`；
- 输入通道数和归一化。

## 每个 repo 下载时请一并记录

- repo URL；
- commit SHA/tag；
- license；
- Python/PyTorch 版本；
- checkpoint 来源和 SHA256；
- 官方输入分辨率、mean/std；
- 是否依赖自定义 CUDA op。

有自定义 CUDA op 的模型不会直接进入 Efficiency Track 首轮候选，除非 Kaggle T4 离线构建和真实 latency 都验证通过。

