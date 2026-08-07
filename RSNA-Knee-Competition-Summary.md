# RSNA Knee Abnormality Detection：比赛与数据说明

## 0. 任务性质：这是多标签分类，不是图像分割

尽管输入是膝关节 MRI 图像，本比赛的预测目标不是输出像素级或体素级掩码（mask），而是对**每一次 MRI 检查（study）**输出 12 个临床异常的发生概率。因此，准确的任务定义是：

- **任务类型**：study-level 多标签二分类（multi-label classification）
- **输入单位**：一次膝关节 MRI 检查；每个 study 包含多个 MRI series，每个 series 又包含多张 DICOM 切片
- **输出单位**：每个 study 对应 12 个置信度/概率值
- **训练辅助信息**：原始放射学报告文本，可用于从未完整标注的数据中提取或生成标签
- **不包含的目标**：官方资料没有提供分割掩码，也不要求提交病灶轮廓或解剖结构 mask

## 1. 比赛内容

### 1.1 背景与目标

膝关节 MRI 能显示韧带、软骨、半月板和骨骼等结构，但 ACL/MCL 损伤、半月板撕裂、软骨退变、骨折等异常可能很细微，不同阅片者之间也可能存在判断差异。

参赛者需要建立机器学习模型，从膝关节 MRI 检查中识别 12 类具有临床意义的异常。训练数据将每个影像 study 与其原始放射学报告配对，可利用影像与文本信息进行训练、弱监督标注或标签补全。

最终模型应当对测试集中的每个 study 输出 12 个异常的预测置信度，用于辅助膝关节 MRI 判读。

### 1.2 十二个预测目标

| 提交列名 | 中文含义 | 类型 |
|---|---|---|
| `ACL` | 前交叉韧带损伤 | 0/1 二分类 |
| `MCL` | 内侧副韧带损伤 | 0/1 二分类 |
| `Medial Meniscus` | 内侧半月板撕裂 | 0/1 二分类 |
| `Lateral Meniscus` | 外侧半月板撕裂 | 0/1 二分类 |
| `Medial OA` | 内侧胫股关节骨关节炎 | 0/1 二分类 |
| `Lateral OA` | 外侧胫股关节骨关节炎 | 0/1 二分类 |
| `PF OA` | 髌股关节骨关节炎 | 0/1 二分类 |
| `Effusion` | 关节积液/液体增多 | 0/1 二分类 |
| `Synovitis` | 滑膜炎 | 0/1 二分类 |
| `Baker's` | Baker 囊肿（腘窝囊肿） | 0/1 二分类 |
| `Contusion` | 骨挫伤/骨瘀伤 | 0/1 二分类 |
| `Fracture` | 骨折 | 0/1 二分类 |

### 1.3 提交形式

每行代表一个测试 study，第一列是 `StudyInstanceUID`，其余 12 列为对应异常的预测置信度：

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

比赛要求通过 Kaggle Notebook 提交，输出文件名必须为 `submission.csv`。CPU/GPU Notebook 的运行时间上限均为 9 小时，且运行时不能访问互联网；允许使用免费且公开可用的外部数据和预训练模型。

## 2. 数据集

### 2.1 数据规模与特点

| 项目 | 说明 |
|---|---|
| 数据模态 | 膝关节 MRI DICOM + 训练集放射学报告文本 |
| 文件类型 | `.dcm`、`.csv` |
| 文件数量 | 819,640（页面摘要约记为 820k） |
| 数据大小 | 569.76 GB |
| 测试集规模 | 约 1,300 个 studies |
| 数据来源 | 多个国家/地区的影像站点，覆盖不同扫描仪、协议和人群 |
| 标签情况 | 只有一小部分训练 studies 具有逐异常标签；其余数据提供原始报告，可用于派生标签 |
| 许可 | Subject to Competition Rules |

需要特别注意：训练集、公开榜测试集和最终测试集中的异常患病率不保证一致，可能存在标签分布偏移。

### 2.2 数据层级

```text
Study（一次扫描检查）
└── Series（一次具体 MRI 序列/采集）
    └── SOP Instance（单张 DICOM 切片）
```

训练图像的目录结构为：

```text
train_series/
└── <StudyInstanceUID>/
    └── <SeriesInstanceUID>/
        └── <SOPInstanceUID>.dcm
```

一个 study 包含若干个 series。每个 series 通常包含 20–45 张切片，中位数为 30；少数 series 可达到数百张切片。

### 2.3 文件说明

#### `train.csv`

每行对应一个训练 study：

| 字段 | 含义 |
|---|---|
| `StudyInstanceUID` | study 唯一标识，同时对应 `train_series/` 下的一级目录名 |
| `PatientSex` | 患者性别，取 `Male` 或 `Female`，可能为空 |
| `Report` | 原始自由文本放射学报告；报告可能使用多种语言 |
| 12 个标签列 | 上述 12 个异常的二值标签；数据集中仅一小部分 study 具有逐异常标注 |

#### `train_series.csv`

每行对应一个训练 series：

| 字段 | 含义 |
|---|---|
| `StudyInstanceUID` | 该 series 所属的 study |
| `SeriesInstanceUID` | series 唯一标识，同时对应 study 目录下的二级目录名 |
| `Fluid_Sensitive` | 是否为液体敏感序列；T2、PD、STIR 等记为 1，否则为 0 |
| `Fat_Suppression` | 是否使用脂肪抑制；是为 1，否则为 0 |
| `Anatomical_Plane` | 成像平面：`Sagittal`、`Coronal` 或 `Axial` |

#### 测试与提交文件

| 文件/目录 | 说明 |
|---|---|
| `test.csv` | 示例文件含 3 个公开测试 study ID；正式评分时替换为真实测试数据，真实测试集约 1,300 个 studies |
| `test_series.csv` | 与 `train_series.csv` 相同的 schema；正式评分时替换为真实测试 series 描述 |
| `test_series/` | 与训练图像相同的 DICOM 目录层级；正式评分时替换为真实测试 DICOM |
| `sample_submission.csv` | 合法提交示例，12 个预测列全部填为 `0.5` |

### 2.4 DICOM 数据注意事项

- 不同 series/study 的灰度强度、方向和空间分辨率存在差异。
- 数据混合使用多种 DICOM transfer syntax：未压缩 Explicit VR Little Endian、JPEG Lossless、JPEG 2000、Implicit VR Little Endian。
- 每个 DICOM 文件仅保留允许列表中的 86 个元数据标签。
- 预处理时不能假定固定切片数、固定 spacing、固定方向或固定强度范围。
- 建议根据 DICOM 几何信息而非文件名排序切片，并验证所用解码库能覆盖上述 transfer syntax。

## 3. 评判标准与公式

### 3.1 主赛道：12 类 Macro-averaged ROC AUC

对第 \(i\) 个异常标签，分别计算预测置信度与真实二值标签之间的 ROC AUC，最终分数是 12 个 AUC 的等权平均：

$$
\text{Final Score}
= \frac{1}{12}\sum_{i=0}^{11}\operatorname{AUC}_i
$$

其中：

- \(\operatorname{AUC}_i\) 为第 \(i\) 个异常的 ROC 曲线下面积；
- 12 个标签权重相同，不会因为某一标签样本更多而获得更高权重；
- 主赛道目标是**最大化** Final Score。

ROC 曲线由不同阈值下的真正率和假正率构成：

$$
\operatorname{TPR}=\frac{TP}{TP+FN},
\qquad
\operatorname{FPR}=\frac{FP}{FP+TN}
$$

$$
\operatorname{AUC}=\int_0^1 \operatorname{TPR}(\operatorname{FPR})\,d\operatorname{FPR}
$$

因此评分关注每一类阳性样本相对于阴性样本的排序质量，而不是固定阈值下的 accuracy。类别不平衡时，训练与验证仍应重点观察每类 AUC，不能只看整体 loss 或准确率。

### 3.2 效率赛道：准确率与运行时间联合评分

效率分数为：

$$
\text{Efficiency}
= \frac{\operatorname{AUC}}
{\operatorname{Benchmark}-\max(\operatorname{AUC})}
+ \frac{\operatorname{RuntimeSeconds}}{32400}
$$

| 符号 | 含义 |
|---|---|
| \(\operatorname{AUC}\) | 当前提交在主赛道指标上的得分 |
| \(\operatorname{Benchmark}\) | `sample_submission.csv` 的主赛道得分 |
| \(\max(\operatorname{AUC})\) | Private Leaderboard 所有提交中的最高主赛道得分 |
| \(\operatorname{RuntimeSeconds}\) | 该提交完成评估所需的秒数 |
| \(32400\) | 9 小时对应的秒数，即 Notebook 运行上限 |

效率分数的目标是**最小化**。第一项体现预测性能，第二项按 9 小时上限归一化并惩罚运行时间。参评提交还必须在 Private Leaderboard 上优于 `sample_submission.csv` 基准，并满足比赛规定的提交选择条件。

## 4. 对建模方案的直接含义

1. **按 study 划分训练/验证集**：同一 study 的不同 series 或切片不能跨 fold，否则会产生严重数据泄漏。
2. **进行多序列聚合**：模型需要把不同成像平面、序列类型和可变切片数整合为 study-level 表征。
3. **把报告视为训练监督来源**：测试文件说明中没有提供报告字段，因此报告更适合用于训练期标签抽取、弱监督、蒸馏或表征预训练，而不能默认作为测试时输入。
4. **逐标签监控 AUC**：最终分数是 12 类等权宏平均，弱势类别会与强势类别同等影响总分。
5. **验证分布偏移鲁棒性**：不同站点、设备、协议、语言和异常患病率都可能变化，应采用按 study 的分层/分组验证，并检查各类与各域的表现。
6. **同时考虑推理效率**：若参与效率赛道，应优化 DICOM 解码、切片采样、模型数量和集成规模，并记录端到端运行时间。

## 5. 一句话总结

这是一个以大规模、多中心膝关节 MRI DICOM 为输入、以放射学报告作为训练辅助信息、对每个检查预测 12 类异常概率的多标签分类比赛；主指标是 12 类 ROC AUC 的宏平均，另设同时考察 AUC 与 9 小时内运行时间的效率赛道。
