# RSNA Knee MRI Image Audit Plan

> 定位：这是正式训练前的 `Phase -1`。目标不是“看几张图确认能打开”，而是建立一个可复现的数据事实层，用它决定 2D/2.5D/3D、序列组织、归一化、自监督策略、切片预算与 T4 推理管线。

## 1. 审计目标

本次 audit 需要回答六类问题：

1. **数据是否可靠**：所有 DICOM 能否解码，UID、切片顺序、几何和像素值是否有效。
2. **MRI 如何组织**：每个 study 有哪些 plane、序列类型和缺失模式，能否形成稳定的 series slots。
3. **模型应看什么**：2D、2.5D 或 3D 哪种输入更符合实际采集几何，应该保留多少切片/window。
4. **如何做域适配**：不同 scanner、协议、强度和空间分辨率的差异有多大，是否值得做 knee MRI 域内自监督。
5. **是否存在伪相关或泄漏**：标签是否和 scanner、协议、序列缺失、患者重复等因素耦合。
6. **数据管线是否满足效率目标**：DICOM header、pixel decode、排序、归一化与 resize 分别消耗多少时间。

### 1.1 非目标

- audit 阶段不训练正式分类模型。
- 不修改或覆盖原始 DICOM。
- 不根据文件名猜测切片顺序。
- 不把放射学报告作为验证或测试时输入。
- 不用少量随机图像替代全量 metadata/integrity 扫描。

## 2. 输入与边界

预期输入由配置文件指定，不能在代码中写死绝对路径：

```yaml
data:
  train_csv: /path/to/train.csv
  train_series_csv: /path/to/train_series.csv
  train_dicom_root: /path/to/train_series
  reports: /path/to/reports_or_null
  test_csv: /path/to/test.csv
  test_series_csv: /path/to/test_series.csv
  test_dicom_root: /path/to/test_series

audit:
  seed: 2026
  output_root: ./artifacts/image_audit
  header_workers: auto
  pixel_workers: auto
  full_decode_check: true
  deep_pixel_sample_per_stratum: 100
  montage_studies_per_bucket: 20
```

测试集相关字段为可选项。若当前环境没有测试图像，先完成训练集 audit；正式提交 notebook 中仍应保留轻量 test-distribution 检查和鲁棒 fallback。

## 3. 最终交付物

```text
artifacts/image_audit/
├── audit_report.md
├── decision_log.md
├── config.snapshot.yaml
├── environment.json
├── tables/
│   ├── study_inventory.parquet
│   ├── series_inventory.parquet
│   ├── dicom_inventory.parquet
│   ├── label_inventory.parquet
│   ├── report_label_agreement.parquet
│   ├── domain_shift.parquet
│   ├── suspected_duplicates.parquet
│   └── runtime_profile.parquet
├── issues/
│   ├── decode_failures.csv
│   ├── geometry_failures.csv
│   ├── ordering_failures.csv
│   ├── metadata_missingness.csv
│   ├── possible_localizers.csv
│   └── quarantine_candidates.csv
├── figures/
│   ├── inventory/
│   ├── geometry/
│   ├── intensity/
│   ├── protocol/
│   ├── labels/
│   ├── leakage/
│   └── runtime/
└── montages/
    ├── random/
    ├── plane_sequence/
    ├── label_positive/
    ├── scanner_protocol/
    └── outliers/
```

所有表均保留稳定的 `StudyInstanceUID`、`SeriesInstanceUID`、`SOPInstanceUID` 关联键；报告和图表必须能反查到原始 study，但不得在公开产物中暴露不必要的患者隐私字段。

## 4. 总体执行流程

```text
A0 20-study smoke test
        ↓
A1 全量 CSV/DICOM header inventory
        ↓
A2 几何、排序、协议和重复数据检查
        ↓
A3 分层 pixel audit + 全量 decodeability 检查
        ↓
A4 自动 montage + 人工影像抽检
        ↓
A5 标签、报告、患者与伪相关审计
        ↓
A6 T4 端到端数据管线 benchmark
        ↓
A7 模型决策表与 Phase 0 配置冻结
```

任何一步发现关键字段定义错误，都应回退并重新生成下游产物，不能只手动修报告。

## 5. A0：Smoke Test

先选择至少 20 个 study，覆盖：

- Sagittal、Coronal、Axial；
- fluid-sensitive 与 non-fluid-sensitive；
- 不同 transfer syntax；
- slice count 的低、中、高分位；
- 至少两个 manufacturer/scanner bucket；
- 已知 gold-positive 和 gold-negative；
- metadata 缺失或异常候选。

Smoke test 验证：

1. CSV 与目录 UID 能正确关联。
2. DICOM decoder 能处理所有出现的 transfer syntax。
3. 几何排序结果和 montage 的解剖连续性一致。
4. 像素归一化不会产生全黑、全白、NaN 或 Inf。
5. 输出表的 schema 在 Windows、本地 Linux/Kaggle 环境均可读取。

通过后冻结 schema，再执行全量扫描。

## 6. A1：全量数据与 DICOM Header Inventory

### 6.1 Study 级字段

每个 study 至少统计：

| 类别 | 字段 |
|---|---|
| 标识 | StudyInstanceUID、PatientID 的不可逆 hash |
| 人口学 | PatientSex、PatientAge（若允许且存在） |
| 规模 | series 数、DICOM 数、总像素估算、磁盘字节数 |
| 完整性 | 缺失目录、空 series、重复 UID、无法关联 CSV |
| 序列覆盖 | plane × fluid-sensitive slot 是否存在 |
| 标签 | gold/derived 标签可用性、正标签数 |

PatientID 仅用于查重和 grouped split；保存前应使用带项目 salt 的不可逆 hash。

### 6.2 Series 级字段

| 类别 | 字段 |
|---|---|
| 标识 | StudyInstanceUID、SeriesInstanceUID |
| 数据集描述 | Fluid_Sensitive、Anatomical_Plane |
| DICOM 描述 | SeriesDescription、ProtocolName、SequenceName |
| 采集类型 | ScanningSequence、SequenceVariant、ScanOptions、MRAcquisitionType |
| 参数 | TR、TE、EchoTrainLength、FlipAngle、MagneticFieldStrength |
| 域信息 | Manufacturer、ManufacturerModelName、SoftwareVersions |
| 几何 | Rows、Columns、PixelSpacing、SliceThickness、SpacingBetweenSlices |
| 规模 | 切片数、估算体素数、磁盘字节数 |

字符串字段先保留原始值，再建立规范化版本；禁止直接用关键词规则覆盖原始 metadata。

### 6.3 DICOM/SOP 级字段

至少读取：

- StudyInstanceUID、SeriesInstanceUID、SOPInstanceUID；
- InstanceNumber；
- ImagePositionPatient；
- ImageOrientationPatient；
- Rows、Columns、PixelSpacing；
- SliceThickness、SpacingBetweenSlices；
- BitsAllocated、BitsStored、HighBit、PixelRepresentation；
- PhotometricInterpretation；
- RescaleSlope、RescaleIntercept；
- WindowCenter、WindowWidth；
- TransferSyntaxUID；
- 文件大小、header 读取时间、相对路径。

### 6.4 必须输出的统计

对 study、series、slice count、Rows、Columns、spacing、thickness、文件大小输出：

- count、missing count、unique count；
- min、P1、P5、P25、P50、P75、P95、P99、max；
- histogram 与 ECDF；
- 按 plane、fluid-sensitive、manufacturer、field strength 分桶的同类统计。

## 7. A2：几何、排序与重复检查

### 7.1 几何排序

对每个 series，从方向余弦得到切片法向量：

\[
\mathbf{n}=\mathbf{r}\times\mathbf{c}
\]

其中 $\mathbf{r}$、$\mathbf{c}$ 来自 `ImageOrientationPatient`。切片标量位置为：

\[
z_i=\mathbf{n}^{\mathsf T}\mathbf{p}_i
\]

$\mathbf{p}_i$ 为 `ImagePositionPatient`。默认按 $z_i$ 排序；`InstanceNumber` 只作为缺失几何时的 fallback，文件名只用于稳定 tie-break，不能决定医学顺序。

### 7.2 几何质量指标

每个 series 计算：

- 相邻 $\Delta z_i$ 的中位数、MAD、min/max；
- spacing coefficient of variation：

\[
CV_z=\frac{\operatorname{std}(|\Delta z|)}{\operatorname{mean}(|\Delta z|)+\epsilon}
\]

- 重复位置、异常大 gap、方向变化；
- metadata plane 与几何推导 plane 的一致性；
- Rows/Columns/PixelSpacing 是否在 series 内变化；
- InstanceNumber 与几何顺序的 Kendall/Spearman 一致性。

### 7.3 重复与近重复

分三层检查：

1. UID 重复：SOP、series、study UID。
2. 像素精确重复：解码后像素内容 hash。
3. 近重复：固定归一化和缩放后的 pHash/series signature。

series signature 使用固定相对位置的若干切片生成，避免只比较中心切片。所有跨 fold、跨 patient hash、train/test 的疑似重复必须单独列出。

## 8. A3：Pixel Audit

### 8.1 两级策略

**Level 1：全量 decodeability pass**

- 逐文件触发真实 pixel decode；
- 不长期保留完整像素数组；
- 记录成功/失败、decoder、transfer syntax、耗时和错误类型；
- 对失败文件使用预注册的第二 decoder/fallback 再试一次；
- 仍失败的文件进入 quarantine candidate，而不是静默跳过。

**Level 2：分层 deep pixel statistics**

分层键至少包含：

```text
plane × fluid_sensitive × manufacturer × field_strength
× slice_count_quantile × spacing_quantile × label_availability
```

每个常见 bucket 固定随机抽样；所有罕见 bucket、decode 异常、geometry 异常和强度 outlier 强制纳入。随机种子和抽样表必须保存。

### 8.2 原始像素统计

在应用合法 modality rescale 后，统计：

- min、max、P0.5、P1、P5、P50、P95、P99、P99.5；
- mean、std、median、MAD、skewness、kurtosis；
- unique value count、zero fraction；
- NaN/Inf、负值、饱和值比例；
- 直方图 entropy；
- 前景 bbox 面积比和背景比例；
- 相邻切片强度相关性。

这些统计是工程 proxy，不应被解释为临床 MRI SNR。

### 8.3 质量 proxy

对 deep sample 计算：

- 高频能量/局部方差：噪声与锐度 proxy；
- 低频强度不均匀：bias-field proxy；
- 边缘密度和梯度方向一致性：motion/ghosting proxy；
- 前景质心和 bbox：膝关节是否居中；
- 相邻切片结构相似度：重复、错序或跳层 proxy；
- 空白、localizer/scout、非诊断图像候选分数。

所有 proxy 只用于筛选人工复核对象，不直接据此删除图像。

### 8.4 归一化候选

在完全相同的 deep sample 上比较：

| ID | 方法 | 主要风险 |
|---|---|---|
| N0 | DICOM rescale 后直接 min-max | 对 outlier 敏感 |
| N1 | per-slice P1–P99 clipping + scale | 可能破坏跨切片强度关系 |
| N2 | per-series P1–P99 clipping + scale | 对全 series 更稳定 |
| N3 | foreground robust z-score | 前景 mask 失败会污染统计 |
| N4 | per-series z-score + fixed clipping | 跨 scanner 较稳健 |

比较标准：

- scanner/protocol 分桶后的分布差异是否缩小；
- 解剖前景对比是否保留；
- 相邻切片的相对强度关系是否保留；
- 是否产生 clipping 过多或低对比度样本；
- CPU 成本和 T4 pipeline 成本。

归一化选择必须在模型实验前冻结；若后续改变，应作为独立消融而不是暗中更改 pipeline。

## 9. A4：自动 Montage 与人工复核

### 9.1 Series montage

每个 series 按几何排序后，从归一化位置选取固定切片：

```text
0%、10%、25%、40%、50%、60%、75%、90%、100%
```

若切片少于所需数量则去重。montage 标题必须显示：

- study/series 的短 hash；
- plane、fluid-sensitive；
- slice count、spacing、Rows × Columns；
- manufacturer/model、field strength；
- 归一化方法；
- geometry/intensity issue flags。

### 9.2 Study montage

按统一 slot 排列一个 study 的全部主要 series：

```text
Sagittal-fluid | Sagittal-nonfluid
Coronal-fluid  | Coronal-nonfluid
Axial-fluid    | Axial-nonfluid
Other / unknown series
```

缺失 slot 显式显示 `MISSING`，不能通过挪动其他 series 隐藏缺失模式。

### 9.3 抽检 bucket

- 完全随机；
- 每种 plane × sequence；
- 每个主要 scanner/protocol；
- 每个标签的 gold-positive；
- gold-negative；
- slice count、spacing、强度、解码延迟的 P1/P99 outlier；
- 所有 critical issue；
- suspected duplicate pairs。

### 9.4 人工复核表

| 项目 | 取值 |
|---|---|
| 解剖对象正确 | yes / no / uncertain |
| plane 标注正确 | yes / no / uncertain |
| 切片顺序连续 | yes / no / uncertain |
| 覆盖完整 | complete / partial / localizer |
| 方向/左右一致 | yes / no / uncertain |
| 明显 motion/ghosting | none / mild / severe |
| fat/fluid-sensitive 观感一致 | yes / no / uncertain |
| 强度归一化可用 | yes / no |
| 是否需要 quarantine | yes / no / review |
| 备注 | free text |

至少对所有关键 outlier 双人复核；若只有一名复核者，则保留 `uncertain`，不强制二元决定。

## 10. A5：标签、报告与伪相关 Audit

> 官方已澄清的标注规则（作为本节所有一致性分析的基准，而非重新假设）：gold 标签由两名肌骨放射科医生独立评图、第三人仲裁分歧产生，独立于报告文本；报告与影像结论冲突时以影像为准；模糊/临界表现一律判为阴性。已知方向性偏差：报告写作阈值更宽松，容易比 gold 标签"过度报告"阳性。社区分析（非官方）显示全部约 58 个 gold study 相对约 4,407 个训练 study 存在约 2×（骨折达 3.1×）的异常富集，因此从 gold 子集估计的患病率/阈值不能代表全量分布。本节的 `report_label_agreement.parquet` 分析应显式验证这一"报告过度报告"的方向是否成立，而不只是报告一致率数字。

### 10.1 标签统计

分别对 gold、规则派生、LLM 派生标签输出：

- 每标签 positive、negative、missing、uncertain 数量；
- prevalence 与 bootstrap 95% CI；
- 每 study 正标签数量；
- 12 × 12 共现矩阵；
- Phi/Jaccard correlation；
- gold 与 derived 标签的一致率、敏感度、特异度；
- 按报告语言和抽取置信度分桶的一致性。

### 10.2 标签与域的关系

每个标签检查其与以下变量的关系：

- manufacturer/model；
- field strength；
- plane/sequence slot 缺失；
- slice count、spacing、分辨率；
- 性别、年龄段（若存在且允许）；
- 报告语言、报告长度和标签来源。

报告分桶 prevalence、odds ratio 和带置信区间的差异。若某标签主要由 scanner 或缺失序列预测，应把该域作为 grouped/stratified validation 维度，并在模型中限制 shortcut。

### 10.3 报告泄漏规则

- 图文预训练、报告软标签和 report teacher 必须在 fold 内生成或训练。
- validation study 的报告不得进入该 fold 的视觉模型训练。
- gold 验证指标只在从未参与训练的 study 上计算。
- 报告中直接出现的 study/patient 标识不能成为文本特征。
- 推理模型不得依赖测试时不存在的报告字段。
- 报告原文不得发送给商业 LLM API（OpenAI、Anthropic、Google 等，包括协作使用的 Claude）用于标签抽取或其他处理；只能使用本地/开源权重模型，推理留在自己的环境内（比赛数据安全条款，官方尚未正面澄清但高风险，应规避）。

### 10.4 Fold 泄漏检查

优先级：

1. 若有 PatientID：按 patient hash 分组。
2. 否则：按重复/近重复 study graph 形成 connected components 分组。
3. 在组级别做 multilabel stratification。

每次生成 folds 后必须断言：

- patient hash 跨 fold 交集为 0；
- exact duplicate component 跨 fold 交集为 0；
- StudyInstanceUID 跨 fold 交集为 0；
- 每个 fold 的 12 类 prevalence 和主要域分布均有报告。

## 11. A6：T4 数据管线 Benchmark

> Kaggle 官方已确认 `RuntimeSeconds`（效率赛道计分依据）是 notebook 从开始执行到结束的完整 wall time，包含包安装、模型加载、DICOM 读取在内的全部耗时，因此本节的所有计时必须覆盖到 notebook 启动阶段，不能只测"数据管线核心步骤"。详见 [Efficiency Experiment Plan 13.2 节](./RSNA-Knee-Efficiency-Experiment-Plan.md)的 T4 测速协议。

### 11.1 测试场景

固定同一批至少覆盖 P50、P95 和最大 study 大小的样本，分别测试：

| ID | 场景 |
|---|---|
| R0 | header only |
| R1 | 全 series 全 slice decode |
| R2 | header 筛选后仅 decode 均匀 Top-K |
| R3 | 低分辨率 selector + 高分辨率 Top-K |
| R4 | decode + normalize + resize + host-to-device |
| R5 | 完整 dataloader，但不运行 backbone |

### 11.2 分项计时

- 包导入/环境初始化；
- 模型 checkpoint 加载；
- 文件发现；
- header 读取；
- 几何排序；
- pixel decode；
- modality rescale/normalization；
- resize 与 2.5D window 构造；
- CPU→GPU transfer；
- dataloader wait；
- selector；
- 缓存命中/未命中。

每项报告 warmup 后的 mean、median、P90、P95、max，并按 transfer syntax、slice count、image size 分桶。

### 11.3 吞吐投影

设每个 study 的端到端数据时间为 $t_s$，对约 1,300 个测试 study 的投影：

\[
T_{data}=\sum_{s=1}^{1300}t_s
\]

同时采用 bootstrap study mix 给出 P50/P90 总耗时区间。内部整体目标仍为平均不超过 16 秒/study，数据管线应只占其中一部分，并为异常长 series 留出余量。

若 GPU 等待数据比例持续较高，后续 backbone FLOPs 对总时间的解释力有限，应先修复 I/O 和 preprocessing。

## 12. Train–Test / Domain Shift Audit

若可访问测试 metadata，比较 train/test 的：

- series 数和 slice count；
- plane × sequence slot 覆盖；
- Rows/Columns、spacing、thickness；
- transfer syntax；
- manufacturer/model、field strength；
- intensity proxy；
- metadata missingness。

连续变量使用 ECDF、KS/Wasserstein 距离；类别变量使用频率差、Jensen–Shannon divergence。这里只做无标签域偏移检查，不读取或推断测试标签。

发现明显偏移时：

- 优先采用 robust normalization；
- 保留 metadata-conditioned adapter/FiLM 候选；
- 验证集增加按域留出测试；
- 不针对少量公开 test 样本手工过拟合预处理。

## 13. Audit 到模型的决策门槛

### 13.1 2D、2.5D 与 3D

| 发现 | 决策 |
|---|---|
| 几何字段缺失多、spacing 不稳定、series 深度差异大 | 2.5D window + geometry-aware position |
| 几何完整且 resample 后体积规模可控 | 允许轻量 3D 进入小规模候选 |
| 相邻切片高度重复但跨层变化重要 | 增大 slice stride，不盲目增加连续切片 |
| 病灶证据集中在少量局部窗口 | Knee-BCRS / learned Top-K |

3D 进入正式筛选前应至少满足：绝大多数主 series 可按物理位置可靠排序、spacing 异常有明确 fallback、实际 T4 内存和 latency 可控。任何一项不满足，3D 只作为探索项。

### 13.2 序列组织

| 发现 | 决策 |
|---|---|
| plane × fluid slot 覆盖稳定 | 固定 slot aggregator |
| slot 缺失和额外 series 很常见 | set encoder + missing/unknown token |
| 同 slot 多个 series 常见 | slot 内 attention/quality selection |
| plane metadata 与几何冲突 | 以几何为主，并加入 unknown/conflict 标记 |

### 13.3 自监督与多模态

| 发现 | 决策 |
|---|---|
| scanner/强度域差异显著 | 优先 image-only domain SSL |
| 无标签 MRI 量大且覆盖多个协议 | 做 masked/DINO-style domain adaptation pilot |
| report-derived 与 gold 一致性高 | 加入 image-report alignment 和 soft-label teacher |
| 报告噪声或语言差异大 | 降低文本损失权重，使用 confidence mask |
| 图文预训练只提升 derived AUC、不提升 gold AUC | 删除多模态链路或仅保留 teacher distillation |

### 13.4 效率

| 发现 | 决策 |
|---|---|
| pixel decode 占主要时间 | decode-before-select 改为 header/低分辨率预选 |
| 小 batch/kernel launch 占主要时间 | 固定 K、稠密 batching |
| 图像分辨率远高于有效解剖细节 | 下调输入分辨率并做同成本消融 |
| 少数超长 series 决定 P95 | 设置有证据的 cap/selector fallback |

## 14. 严重级别与处理原则

### Critical

- 任何未处理的 pixel decode crash；
- fold 中 patient/exact duplicate 泄漏；
- UID 映射错误；
- 排序导致明显解剖跳变；
- 归一化产生 NaN/Inf；
- pipeline 对某种 transfer syntax 无 fallback。

Critical issue 未关闭前不得启动正式训练。

### Warning

- metadata 缺失；
- spacing 不规则；
- plane 冲突；
- 疑似 localizer；
- 极端 slice count；
- 近重复；
- scanner/label prevalence 强相关。

Warning 不一定删除数据，但必须生成 flag，允许模型、fold 或 sampler 显式处理。

### Informational

- 常规分布差异；
- 对模型没有直接影响的非关键 DICOM tag 变化；
- 只影响少量 montage 外观的显示窗口差异。

## 15. 实施任务拆分

| ID | 脚本/Notebook | 输入 | 核心输出 |
|---|---|---|---|
| AUD-00 | `00_smoke_test` | 20 studies | schema、decoder、montage 验证 |
| AUD-01 | `01_header_inventory` | 全量 DICOM headers | 三层 inventory |
| AUD-02 | `02_geometry_and_order` | SOP geometry | 排序和 spacing issues |
| AUD-03 | `03_decode_and_pixel_stats` | pixels | failures、intensity stats |
| AUD-04 | `04_duplicates` | hashes/signatures | duplicate graph |
| AUD-05 | `05_montages` | inventory + pixels | 分桶 montages |
| AUD-06 | `06_labels_reports` | labels + reports | prevalence、agreement |
| AUD-07 | `07_folds` | patient/duplicate graph | grouped folds |
| AUD-08 | `08_runtime_t4` | stratified studies | runtime profile |
| AUD-09 | `09_generate_report` | 所有产物 | audit report + decision log |

实现时优先写为可复用 Python 模块，Notebook 只负责可视化和决策记录；避免把关键 DICOM 逻辑散落在 Notebook cell 中。

## 16. 测试与可复现性

必须包含以下自动测试：

1. 人工构造 DICOM 的几何排序单元测试。
2. reversed order、duplicate position、missing position fallback 测试。
3. 每种已发现 transfer syntax 的 decoder fixture。
4. 像素符号位、rescale slope/intercept 测试。
5. 全黑、常数、极端值、NaN 防护测试。
6. patient/group fold 不相交测试。
7. montage 索引和原始 SOP 对应测试。
8. 相同 seed/config 生成相同抽样表和 issue 表。

环境快照记录：Python、pydicom、pixel decoder、NumPy、PyTorch、CUDA、GPU、OS 与 commit hash。

## 17. 完成标准

只有全部满足才视为 audit 完成：

- [ ] train CSV、series CSV 和 DICOM 目录可完整关联。
- [ ] 全量 header inventory 已生成。
- [ ] 全量 pixel decodeability 已执行，失败项都有明确处理。
- [ ] 几何排序经过自动测试和 montage 人工验证。
- [ ] 主要 plane/sequence/scanner bucket 已可视化。
- [ ] 所有 gold 标签和 rare-label positive 都进入分层抽检。
- [ ] patient、exact duplicate 和 near-duplicate 检查已完成。
- [ ] grouped multilabel folds 已生成且无交集。
- [ ] 归一化方法已冻结并记录。
- [ ] series slot 策略已冻结并记录。
- [ ] T4 数据管线 P50/P95 和 1,300-study 投影已生成。
- [ ] `decision_log.md` 明确回答 2D/2.5D/3D、SSL、多模态和 selector 决策。
- [ ] 所有 Critical issue 已关闭。

## 18. Decision Log 模板

```markdown
# Image Audit Decision Log

## Data snapshot
- audit date:
- dataset version/hash:
- code commit:
- total studies/series/DICOM:

## Decisions
- slice ordering:
- plane source and fallback:
- normalization:
- series slots:
- missing-series behavior:
- 2D/2.5D/3D decision:
- window size/stride candidates:
- default K candidates:
- domain SSL decision:
- report multimodal decision:
- fold grouping:
- quarantine/fallback policy:

## Runtime
- median/P95 data time per study:
- projected 1,300-study data time:
- dominant bottleneck:

## Open warnings
- issue:
- affected fraction:
- mitigation:
- owner:
```

## 19. 与后续实验计划的接口

Audit 完成后向正式实验计划提供以下冻结输入：

1. `fold_id`：patient/duplicate-safe 的 multilabel folds。
2. `series_slot`：规范化 plane/sequence slot 与 missing flags。
3. `slice_position_mm` 和 `relative_position`：2.5D 位置编码。
4. `quality_flags`：localizer、geometry、intensity、decode fallback。
5. `normalization_id`：唯一的默认归一化配置。
6. `runtime_bucket`：P50/P95/long-study 分桶。
7. `domain_bucket`：scanner/protocol 分桶。
8. `ssl_manifest`：允许进入域内自监督的数据与 fold 边界。
9. `quarantine_policy`：训练和推理的确定性 fallback。

后续所有 backbone、Knee-BCRS、聚合头和 loss 消融都必须复用这些输入。除非实验明确研究 audit 中的某个决策，否则不能同时改变数据排序、归一化、fold 或 series slot 定义。
