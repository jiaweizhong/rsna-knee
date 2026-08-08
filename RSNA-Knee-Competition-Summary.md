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

### 1.4 十二个标签的官方判定标准

主办方在比赛论坛公布了完整的标注 rubric（原文附完整病例图示），核心判定标准摘要如下：

| 标签 | 阳性判定标准（rubric 摘要） |
|---|---|
| ACL | 高级别部分或全层撕裂：韧带完全不连续，或 >50% 纤维中断；单纯信号增高/退变/增厚但连续性完整判阴性 |
| MCL | 高级别部分或完全急性撕裂：纤维断裂伴韧带内外水肿；低级别扭伤、慢性/陈旧应力改变判阴性 |
| Medial Meniscus | 异常信号在 ≥2 张图像上确实达到关节面，或半月板形态异常（截断、变小、移位碎片）；未达关节面的实质内退变判阴性 |
| Lateral Meniscus | 同上标准，适用于外侧半月板 |
| Medial OA | 内侧间室中/大面积（约 ≥1cm）高级别软骨丢失（>50% 厚度），伴或不伴软骨下骨髓改变 |
| Lateral OA | 同上标准，适用于外侧间室 |
| PF OA | 同上标准，适用于髌股间室 |
| Effusion | 中量或大量关节积液使关节囊扩张 |
| Synovitis | 滑膜炎症增厚 |
| Baker's | 膝后方典型位置的中量或大量积液 |
| Contusion | 骨髓水肿样信号（撞击所致），无明确骨折线 |
| Fracture | 急性皮质断裂或骨折线 |

两条全局规则（适用于全部 12 类）：

1. **模糊/临界表现一律判为阴性**（favor specificity），不算"疑似阳性"。
2. 每个 study 的标签是"整个检查、单侧膝关节"级别的合并结论，由两名读者独立标注、第三人仲裁分歧得出。

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
| 标签情况 | 官方完整人工标签仅约 **58 个 study**（约占全部约 4,407 个训练 study 的 1.3%）；其余 study 只有原始报告、无逐异常标签，需要派生标签 |
| 许可 | Subject to Competition Rules |

需要特别注意：训练集、公开榜测试集和最终测试集中的异常患病率不保证一致，可能存在标签分布偏移。

社区分析（非官方，来自比赛论坛的统计审计）进一步发现：这 58 个 gold study 相对全部约 4,407 个训练 study 存在约 **2× 的异常富集**（骨折一项达 3.1×，12 类里 11 类方向一致），说明其患病率分布明显偏"病情更重"，不能直接作为患病率先验或分类阈值的校准来源。同时，58 这个样本量对 macro AUC 而言方差很大：两个模型的真实差距小于约 0.02 时，仅凭这 58 个 study 的 OOF 分数很难可靠区分优劣（详见 [Efficiency Experiment Plan 4.3 节](./RSNA-Knee-Efficiency-Experiment-Plan.md)的噪声下限分析）。

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
| `Report` | 原始自由文本放射学报告；报告可能使用多种语言 |
| 12 个标签列 | 上述 12 个异常的二值标签；仅约 58 个 study（约占 4,407 个训练 study 的 1.3%）具有完整逐异常人工标注，其余为空 |

`PatientSex` **不在** `train.csv` 中（主办方已确认：曾计划加入，但因 DICOM header 本身已包含该字段、被判定为冗余而移除，数据页面说明将同步更新）。需要患者性别时应从 DICOM header 直接读取，不能假设 CSV 会提供该字段。

标签标注规则（主办方在论坛澄清）：
- gold 标签由两名肌骨放射科医生独立评图、第三人仲裁分歧产生，**独立于报告文本**；与报告结论冲突时以影像为准。
- 存在已知的系统性方向偏差：报告写作阈值更宽松，容易比 gold 标签"过度报告"（over-call）阳性；**模糊/临界表现在 gold 标注里一律判为阴性**（favor specificity）。
- 因此报告规则/LLM 派生的弱标签，如果不显式复现"模糊即阴性"这条规则，会比 gold 标签更容易误判为阳性——设计弱标签抽取逻辑时必须把这条规则写进去，而不是简单做关键词匹配。
- 双侧膝关节偶尔共用一个 `StudyInstanceUID`；主办方对这类双侧检查/多份报告已人工核对并调整了报告文本/DICOM metadata 以便消歧。

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

Kaggle staff 已在论坛正面澄清两点：**GPU notebook 可以参加效率赛道**（不是 CPU-only 赛道）；`RuntimeSeconds` 是 notebook **从开始执行到结束的完整 wall time**，包含包安装、模型加载、读取测试 DICOM 在内的全部耗时，不是只算核心推理段。因此正式提交应尽量减少不必要的运行时 `pip install` 和重量级 import，把这些开销也计入端到端预算（对应 [Efficiency Experiment Plan 13.2 节](./RSNA-Knee-Efficiency-Experiment-Plan.md)的 T4 测速协议）。

效率分数的目标是**最小化**。第一项体现预测性能，第二项按 9 小时上限归一化并惩罚运行时间。参评提交还必须在 Private Leaderboard 上优于 `sample_submission.csv` 基准，并满足比赛规定的提交选择条件。

## 4. 对建模方案的直接含义

1. **按 study 划分训练/验证集**：同一 study 的不同 series 或切片不能跨 fold，否则会产生严重数据泄漏。
2. **进行多序列聚合**：模型需要把不同成像平面、序列类型和可变切片数整合为 study-level 表征。
3. **把报告视为训练监督来源**：测试文件说明中没有提供报告字段，因此报告更适合用于训练期标签抽取、弱监督、蒸馏或表征预训练，而不能默认作为测试时输入。
4. **逐标签监控 AUC**：最终分数是 12 类等权宏平均，弱势类别会与强势类别同等影响总分。
5. **验证分布偏移鲁棒性**：不同站点、设备、协议、语言和异常患病率都可能变化，应采用按 study 的分层/分组验证，并检查各类与各域的表现。
6. **同时考虑推理效率**：若参与效率赛道，应优化 DICOM 解码、切片采样、模型数量和集成规模，并记录端到端运行时间。

## 5. 合规与开放问题

以下问题来自比赛论坛，部分已由主办方/Kaggle 官方明确，部分仍待官方书面澄清——在有正式回复前不应作为既定策略依赖：

**已确认：**
- GPU notebook 可参加效率赛道；`RuntimeSeconds` 计入完整 wall time（含包安装、模型加载、DICOM 读取）。
- `train.csv` 不含 `PatientSex`，需从 DICOM header 读取。
- gold 标签独立于报告、由影像判定；与报告冲突时以影像为准；模糊表现一律判阴性。

**待官方澄清，暂不应依赖：**
- MRNet、fastMRI+、OAI、SKM-TEA 等需要注册/签署研究协议才能下载的公开膝关节 MRI 数据集，是否满足比赛规则里"公开、免费、平等可获取"的外部数据条件，官方尚未正面答复。

**高风险，建议规避：**
- 将训练集报告文本发送给商业 LLM API（OpenAI、Anthropic、Google 等）用于标签抽取，可能违反比赛规则 4.b 数据安全条款（禁止向未接受规则的第三方提供比赛数据）；多位高分参赛者认为这大概率不允许，官方尚未针对此问题正面回复。本地/开源权重模型（如 Qwen）在自己环境内推理被认为是安全的。**在官方明确澄清前，报告原文不应发送给任何商业 LLM API（包括协作使用的 Claude）。**

## 6. 一句话总结

这是一个以大规模、多中心膝关节 MRI DICOM 为输入、以放射学报告作为训练辅助信息、对每个检查预测 12 类异常概率的多标签分类比赛；主指标是 12 类 ROC AUC 的宏平均，另设同时考察 AUC 与 9 小时内运行时间的效率赛道。
