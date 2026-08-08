# RSNA Knee Abnormality Detection：精度-效率实验计划

> 目标：在 T4 GPU、正式推理时间不超过 8 小时的约束下，建立一个以高质量排序为核心、以可预测计算预算为约束的膝关节 MRI 多标签分类系统，并探索 Efficiency Track 的 Pareto 最优解。

## 1. 任务与工程约束

### 1.1 任务定义

- 输入：一个 knee MRI study，包含若干 sagittal、coronal、axial DICOM series。
- 输出：12 个异常的 study-level 置信度。
- 主指标：12 类 ROC AUC 的宏平均。
- 任务性质：多标签分类，不是像素/体素分割。
- 训练辅助信息：放射学报告和少量完整人工标签；测试时没有报告。

### 1.2 时间与吞吐约束

测试集约 1,300 个 studies。以 8 小时为内部上限：

$$
\frac{1300}{8\times3600}=0.0451\ \text{study/s}
$$

即平均不能超过 22.15 秒/study。为了给 Kaggle 文件系统、DICOM 解码和异常长 series 留出安全余量，内部目标设为：

| 指标 | 目标 | 硬上限 |
|---|---:|---:|
| 1,300 studies 端到端推理 | ≤ 6 小时 | 8 小时 |
| 平均耗时 | ≤ 16 秒/study | 22 秒/study |
| P95 耗时 | ≤ 20 秒/study | 30 秒/study |
| 峰值显存 | ≤ 12 GB | 15 GB |
| 高成本 backbone 输入 | ≤ 15 个/study | 24 个/study |
| 正式提交 | 单模型、无 TTA 优先 | 最多双模型 |

正式效率判断以端到端 T4 实测时间为准，FLOPs 只用于解释，不作为替代指标。

## 2. 总体技术路线

```text
DICOM 与 series metadata
        │
        ├── 几何排序、物理尺度裁剪、laterality 归一化
        │
        ├── 低成本全量扫描分支
        │       └── Knee-BCRS：语义 + 频谱/局部显著性 + 跳过风险
        │                       └── 固定预算选择 Top-K slice windows
        │
        └── 高质量 backbone 仅处理入选 windows
                ├── DINOv2-S/14
                ├── DINOv3 ViT-S/16
                ├── DINOv3 ConvNeXt-Tiny
                ├── PVTv2-B0
                └── MedNeXt-lightweight（探索项）
                         │
                         └── 多尺度 slice/slot 聚合
                                  └── 12 个 per-label logits
```

训练和正式提交分离：训练、交叉验证、伪标签生成和权重选择在独立 notebook 完成；正式提交 notebook 只加载公开挂载的权重并执行测试集推理。

## 3. 从 ESOD/BCRS 迁移到本比赛

### 3.1 BCRS 的直接启发

[BCRS-Proposal](./attention-efficient/BCRS-Proposal.md) 和 [BCRS-Experiment-Plan](./attention-efficient/BCRS-Experiment-Plan.md) 表明：

1. selector 不能只学习普通 objectness，应显式考虑“跳过该候选会造成什么损失”。
2. 固定容量 Top-K 可以让计算量可预测，同时允许候选身份随 study 改变。
3. 频谱证据可能补回语义响应弱的小目标，但这种互补性不是无条件成立。
4. coverage supervision 本身可能比新增频谱分支更划算，必须单独设置 semantic-only + coverage 对照。
5. 必须加入参数/FLOPs 匹配的普通卷积分支，证明收益来自频谱证据而非额外容量。
6. FLOPs 下降不保证 latency 下降；BCRS 已观察到这一反例，因此所有路由实验必须同时报告真实 wall-clock。

### 3.2 Knee-BCRS 的候选单位

第一版不做空间 patch 的动态稀疏执行，而把候选单位定义为：

```text
一个 series slot 中的一个 2.5D slice window
= 中心切片 + 前后相邻切片
```

原因：

- slice/window 级选择能直接减少 DICOM pixel decode 和 backbone forward。
- 选出的 windows 可以重新组成固定尺寸 dense batch，对 T4 更友好。
- 不需要稀疏卷积、gather/scatter 或动态空间 kernel。
- 在没有检测框和分割 mask 的情况下，比空间 patch routing 更容易获得监督。

空间 ROI routing 仅作为后期可选扩展。

### 3.3 跳过风险与 evidence coverage

训练一个高覆盖 teacher，使用较多切片生成每个候选 window 对每个标签的 evidence target：

$$
q_{ij}\in[0,1]
$$

其中 $q_{ij}$ 表示第 $i$ 个 window 对第 $j$ 个异常的证据强度。候选生成方法按优先级测试：

1. teacher per-label attention；
2. leave-one-window-out logit drop；
3. attention × gradient/CAM；
4. 前三者的 rank average。

selector 输出：

$$
u_{ij}=h(s_i,f_i,m_i,e_K)
$$

其中：

- $s_i$：低分辨率语义特征；
- $f_i$：频谱/局部显著性证据；
- $m_i$：plane、fluid-sensitive、fat suppression、spacing 等 metadata；
- $e_K$：可选的预算编码；
- $u_{ij}$：保留 window $i$ 对标签 $j$ 的优先级。

第一版使用固定 $K$，暂不加入预算编码。只有固定预算模型成立后，才训练单模型多预算版本。

### 3.4 Recall-safe 选择策略

简单 Top-K 可能全部服务于同一易识别标签。第一版采用：

1. 每个标签先提出其 Top-1 window；
2. 对候选取并集；
3. 若超过预算，用 cost-sensitive greedy coverage 压缩到 $K$；
4. 剩余预算按 $\max_j u_{ij}$ 填充。

训练时使用 soft gate $g_i$，正标签 $j$ 的覆盖概率定义为：

$$
c_j=1-\prod_i(1-g_iq_{ij})
$$

覆盖损失为：

$$
\mathcal L_{cov}
=-\sum_j w_jy_j\log(c_j+\epsilon)
$$

稀有标签和高漏检代价标签应提高 $w_j$。人工标签 study 的 coverage loss 权重大于报告伪标签 study。

### 3.5 Knee-BCRS 总损失

$$
\mathcal L
=\mathcal L_{cls}
+\lambda_{kd}\mathcal L_{KD}
+\lambda_{cov}\mathcal L_{cov}
+\lambda_{budget}\mathcal L_{budget}
+\lambda_{rank}\mathcal L_{rank}
$$

各项必须逐项消融，不能直接把完整组合与基础模型比较。

## 4. 固定的数据与验证协议

### 4.1 Fold 划分

- 使用 4-fold GroupKFold。
- 以标准化后的报告文本 hash 作为 group，重复报告不能跨 fold。
- 同一 study 的所有 series 和 slices 必须处于同一 fold。
- 保存唯一的 `folds.csv`，后续所有实验复用。

### 4.2 标签来源

| 标签源 | 用途 | 训练权重 |
|---|---|---:|
| 官方完整人工标签（约 58 个 study，占全部约 4,407 个训练 study 的 1.3%） | 主要可信监督，仅用于校准/最终门槛，不作为架构选择的唯一依据 | 3.0–8.0，作为消融变量 |
| 高质量本地 LLM 报告软标签 | 大规模训练监督，也是 Phase 2 backbone/架构排序的主要依据 | 按置信度加权 |
| 多语言规则标签 | fallback/对照 | 按置信度加权 |

标签实验必须先完成，不能在 backbone 实验中同时改变标签源。

**已知的 gold 标签富集偏差**（社区论坛统计审计，非官方）：这 58 个 study 相对全部训练语料存在约 2× 的异常富集（骨折达 3.1×），12 类中 11 类方向一致。由此带来两个后果：(1) 从这 58 个 study 估计的患病率/阈值/类别权重会系统性偏"病情更重"，稀有标签先验（coverage loss 的 $w_j$、BCE pos_weight 等）应以全量约 4,407 study 的弱标签患病率为准，不用 58 个 gold study 的患病率校准；(2) 58 个 study 上测得的 $AUC_{gold}$ 大概率比私榜乐观，私榜分数低于 gold OOF 是预期内的，不代表模型退化。

**报告标签抽取的合规约束**：训练集报告文本不得发送给商业 LLM API（OpenAI、Anthropic、Google 等，包括协作使用的 Claude）——多位参赛者认为这违反比赛数据安全条款，官方尚未正面澄清前应规避。标签抽取只使用本地/开源权重多语言模型（如 Qwen），推理留在自己的环境内。

### 4.3 验证指标

#### 分类质量

- OOF Macro AUC（报告派生标签）。
- OOF Macro AUC（仅人工标签）。
- 12 类 per-label AUC。
- 人工标签 bootstrap 95% CI。
- 保守选择分数：

$$
S_{select}=\min(AUC_{derived},AUC_{gold})
$$

该分数仅用于筛选实验，不视为 leaderboard 估计。

#### 58-study 验证集的噪声下限（重要，决定判断阈值）

社区对 58 个 gold study 做了配对模拟（20,000 次头对头比较），结论必须作为所有基于 $AUC_{gold}$ 判断的前提：

- 两个模型相关度 ρ≈0.9（典型场景）时，配对标准差 σ≈0.0125；真实差距 0.005 只有约 66% 概率选对更优模型，0.01 约 79%，要到 0.02 才有约 94% 把握。
- 可信度强烈依赖相关度：同 backbone 的近似模型比较（换 loss、换聚合头、换 seed，ρ→0.98）σ 降到约 0.006，更可信；跨架构族比较（backbone 之间互相比较，ρ→0）σ 升到约 0.026，同样 0.02 的差距也只有约 79% 把握。
- 方差集中在样本量最小的标签列（如 MCL、Baker's）：这些列贡献了远超其权重占比的 macro AUC 方差，改善它们即使真实存在也更难被这份验证集看见。

据此制定判断规则，全文档统一引用：

- **消融类决策**（同 backbone/同 selector 家族内部的变体比较，如聚合头、loss、selector 变体）：只有当 $\Delta S_{select}$ 达到约 **0.01** 时才视为可信改进；小于这个量级的正向 delta 可以记录、可以保留（如果有工程/理论依据），但不能作为唯一晋级证据。
- **架构类决策**（backbone 家族之间的比较，如 Phase 2）：单次 58-study 比较不可信，必须结合 derived-label（约 4,407 study）OOF 的排序趋势，并至少用 2 个 seed 复核，或要求差距 ≥0.02 才采信。
- 任何声称的改进都应同时报告：delta 数值、该类比较对应的可信阈值、是否达标——不能只报告点估计。

#### Selector 质量

- Teacher evidence coverage@K。
- Positive-label coverage@K。
- 每标签 coverage@K。
- 与 oracle Top-K 的 regret。
- 与均匀采样、中央采样的 AUC 差值。
- selector false-negative case audit。

#### 效率

- 端到端秒/study：mean、median、P95。
- DICOM header、pixel decode、resize、selector、backbone、aggregation 分项时间。
- model-input/s 与 DICOM-slice/s。
- GPU utilization、峰值显存。
- 1,300 studies 的投影总时间。
- 实际 latency 与 FLOPs 的相关性和残差。

## 5. 实验资源等级与晋级规则

### 5.1 资源等级

| 等级 | 用途 | 配置 |
|---|---|---|
| Smoke | 排除实现错误 | 100 studies，1 fold，1 epoch |
| Screen | 比较候选 | 1 fold，冻结/缓存 backbone features |
| Confirm | 验证机制 | 4 folds，固定 seed，轻量 head |
| Final | 最终候选 | 4 folds 或重复 seed，部分微调，T4 全链路测速 |

### 5.2 晋级 Gate

候选进入下一阶段必须满足：

1. 相对当前 baseline，$S_{select}$ 不下降超过消融类噪声下限（约 0.01，见 4.3 节）；或显著减少 runtime 并位于 Pareto frontier。跨架构族比较（如 Phase 2）额外要求 derived-label OOF 排序方向一致，且至少 2 个 seed 复核。
2. 任何单类 AUC 不得下降超过 0.03，除非该类人工样本不足且 CI 大量重叠。
3. 端到端 1,300 studies 投影不超过 8 小时。
4. 无 study/fold 泄漏、无报告在测试时输入、无 submission 缺失。
5. 动态方案必须证明真实 latency 改善；只有 FLOPs 改善不能晋级效率主线。

## 6. Phase 0：数据管线与测量基线

### 目标

建立后续实验不能改变的 DICOM、fold、标签和测速协议。

| ID | 实验 | 变量 | 主要输出 |
|---|---|---|---|
| P0.1 | DICOM 解码兼容性 | transfer syntax | 失败率、速度、fallback |
| P0.2 | 几何排序 | filename / InstanceNumber / geometry | 顺序一致率、耗时 |
| P0.3 | 物理尺度裁剪 | 130 / 150 / 160 mm | 覆盖可视化、resize 成本 |
| P0.4 | Laterality | 无处理 / metadata 归一化 | 4 个侧别相关标签 AUC |
| P0.5 | 流水线性能 | workers、prefetch、batch | T4 各阶段耗时 |

停止条件：如果 DICOM 解码和排序已经占用超过 6 小时投影，先优化 I/O，再进行模型实验。

## 7. Phase 1：标签与高覆盖 Teacher

### 目标

固定标签源并得到用于 selector 监督的高覆盖 teacher。

| ID | 标签/Teacher 配置 | 目的 |
|---|---|---|
| T1.0 | 仅人工标签（约 58 个 study） | 小样本下限；样本量对 12 类 macro AUC 而言极小、方差极大，预期明显劣于 T1.2+，仅作对照 |
| T1.1 | 规则报告标签 | 低成本 baseline |
| T1.2 | 本地/开源权重 LLM 报告软标签（不得使用商业 LLM API，见 4.2 节合规约束） | 主标签候选 |
| T1.3 | LLM 软标签 + gold weight=3 | 权重消融 |
| T1.4 | LLM 软标签 + gold weight=8 | 权重消融 |

Teacher 固定为高覆盖设置：5 个有效 slot、每 slot 8–12 个 window、336 px、DINOv2-S 或当前最稳 backbone。Teacher 的目标是生成稳定 evidence，不参与效率比较。

### Teacher evidence 消融

| ID | Evidence target | 计算成本 | 选择依据 |
|---|---|---:|---|
| TE1 | Per-label attention | 低 | 稳定性和 coverage |
| TE2 | Leave-one-window-out logit drop | 高，仅离线 | 与真实预测影响更一致 |
| TE3 | Attention × gradient | 中 | 局部敏感性 |
| TE4 | TE1/TE2/TE3 rank average | 高，仅离线 | selector OOF AUC 和稳定性 |

## 8. Phase 2：Backbone 筛选

所有 backbone 使用相同的 5 slots、相同 15 个 windows、相同聚合头和标签。

| ID | Backbone | 预训练 | 输入 | 定位 |
|---|---|---|---:|---|
| B2.0 | DINOv2-S/14 | DINOv2 | 224 | 控制组 |
| B2.1 | DINOv3 ViT-S/16 | DINOv3 | 256 | 新版 ViT |
| B2.2 | DINOv3 ConvNeXt-Tiny | DINOv3 | 256 | 效率主候选 |
| B2.3 | PVTv2-B0 | ImageNet | 224/256 | 分层 Transformer |
| B2.4 | MedNeXt-lightweight | 从头/可用权重 | 224 | 探索项 |

**方法论要求**：backbone 之间属于低相关（ρ≈0）比较，58 个 gold study 上的单次 AUC 差异不可信（见 4.3 节噪声下限）。本阶段的主要排序依据是 derived-label（约 4,407 study）OOF trend，至少 2 个 seed；gold OOF 只用于确认没有明显跑偏（例如某 backbone 在 gold 上崩掉但 derived 上正常，需要单独排查原因），不单独作为淘汰依据。

### 训练方式消融

| ID | 方式 |
|---|---|
| FT0 | 完全冻结，训练 MIL head |
| FT1 | 只训练最后 stage/block |
| FT2 | 解冻最后 25% blocks |
| FT3 | 全量微调 |

执行顺序：先全部完成 FT0；只允许 FT0 Pareto frontier 上的前两名进入 FT1/FT2；仅在 FT2 明显受限时测试 FT3。

## 9. Phase 3：分辨率与切片覆盖

固定 Phase 2 最优 backbone。

### 9.1 分辨率消融

| ID | Low-res selector | High-res backbone |
|---|---:|---:|
| R3.0 | 无 | 224 |
| R3.1 | 无 | 256/280 |
| R3.2 | 无 | 336 |
| R3.3 | 126/160 | 280 |
| R3.4 | 126/160 | 336 |

### 9.2 固定采样数量

| ID | 每 slot windows | 最大输入/study |
|---|---:|---:|
| K3.0 | 1 | 5 |
| K3.1 | 2 | 10 |
| K3.2 | 3 | 15 |
| K3.3 | 5 | 25 |
| K3.4 | 8–12 | 40–60，Teacher only |

需要分别绘制 Macro AUC–K、per-label AUC–K 和 runtime–K 曲线。不能只报告平均值，因为不同异常对切片覆盖的敏感度不同。

## 10. Phase 4：Knee-BCRS 针对性消融

固定 Phase 2/3 的 backbone 和高分辨率设置，只改变 selector。

### 10.1 基本选择策略

| ID | Selector | 作用 |
|---|---|---|
| S4.0 | 均匀采样 | 无学习基线 |
| S4.1 | 中央采样 | 公开 baseline 对照 |
| S4.2 | Cheap semantic score | ESOD/objectness 对应项 |
| S4.3 | Semantic + coverage loss | 验证 recall-safe supervision 的独立价值 |
| S4.4 | Spectral-only | 频谱证据独立价值 |
| S4.5 | Semantic + spectral gated fusion | 双证据门控 |
| S4.6 | Semantic + spectral concat fusion | BCRS 中表现更强的融合候选 |
| S4.7 | Semantic + 参数匹配普通卷积分支 | 排除额外容量解释 |
| S4.8 | Oracle teacher Top-K | 性能上界 |

### 10.2 Coverage loss

| ID | $\lambda_{cov}$ | 标签权重 |
|---|---:|---|
| C4.0 | 0 | 无 coverage |
| C4.1 | 0.05 | 均匀 |
| C4.2 | 0.10 | 均匀 |
| C4.3 | 0.10 | inverse prevalence |
| C4.4 | 0.10 | rare-label + gold upweight |

只在 S4.2 上筛选 $\lambda_{cov}$，选定后冻结，不允许在每个 fusion variant 上重新调参。

### 10.3 频谱分支

| ID | 实现 | 说明 |
|---|---|---|
| F4.0 | 无 | semantic-only |
| F4.1 | Sobel/Laplacian depthwise | 最低成本 |
| F4.2 | 高频残差 + 局部方差 | MRI 强度稳健性 |
| F4.3 | FFT/DCT band energy | 频域显式证据 |
| F4.4 | 参数匹配普通 DWConv | 必须的容量对照 |

每个实现报告 selector overhead、GPU kernel 数、median/P95 latency。若频谱分支没有在严格相同 K 下改善 rare-label coverage 或 AUC，则删除。

### 10.4 预算 K

| ID | K | 目的 |
|---|---:|---|
| BK4.0 | 5 | 极限效率 |
| BK4.1 | 10 | 主效率点 |
| BK4.2 | 15 | 默认平衡点 |
| BK4.3 | 25 | 高精度点 |

所有 selector 在完全相同的 K、backbone、分辨率和 batch 下比较。禁止通过多保留候选换取更高 AUC。

### 10.5 固定预算与多预算

| ID | 训练方式 | 进入条件 |
|---|---|---|
| MB4.0 | 每个 K 单独训练 | 固定预算基线 |
| MB4.1 | 单模型随机采样 K | 固定预算 selector 已成立 |
| MB4.2 | 加 budget embedding | MB4.1 在未见 K 上不稳定时 |

第一篇/第一版方案不依赖多预算成功。预算条件化是后续增益，不应阻塞主线。

## 11. Phase 5：聚合头消融

固定最优 backbone、K 和 selector。

| ID | 聚合头 | 额外成本预期 |
|---|---|---:|
| H5.0 | mean + max pooling | 极低 |
| H5.1 | 单一 attention pooling | 低 |
| H5.2 | 12 个 per-label queries | 低 |
| H5.3 | Multi-scale DWConv1D(3/5/7) + H5.2 | 低 |
| H5.4 | H5.3 + channel gate | 低 |
| H5.5 | H5.4 + metadata FiLM | 低 |

Metadata FiLM 输入仅允许使用推理时存在的 metadata。报告文本及由报告直接生成的 embedding 不得进入推理。

### 聚合轴消融

| ID | 结构 |
|---|---|
| A5.0 | 所有 windows 平铺聚合 |
| A5.1 | 先 slice 后 slot |
| A5.2 | slot embedding + per-label query |
| A5.3 | plane-specific head 后再融合 |

如果复杂聚合头相对 H5.2 的 Macro AUC 增益小于消融类噪声下限（约 0.01，见 4.3 节；同 backbone 高相关比较），优先保留 H5.2。

## 12. Phase 6：训练目标与正则化消融

| ID | Loss/策略 | 目的 |
|---|---|---|
| L6.0 | weighted BCE | 基线 |
| L6.1 | BCE + pairwise ranking | 直接优化排序 |
| L6.2 | BCE + teacher distillation | 低覆盖模型保持 teacher 排序 |
| L6.3 | BCE + ranking + KD | 组合 |
| L6.4 | L6.3 + EMA | 稳定伪标签训练 |

增强单独消融：

- intensity/gamma；
- 小角度 affine；
- MRI noise/bias field；
- 不改变 laterality 语义的翻转；
- selector consistency augmentation。

不得使用会重新引入左右膝混乱的增强。

## 13. Phase 7：真实效率与最终 Pareto Frontier

### 13.1 必测配置

| 配置 | Backbone | K | Selector | TTA/ensemble |
|---|---|---:|---|---|
| E7.0 | DINOv2-S | 15 | uniform | 无 |
| E7.1 | 最优 backbone | 15 | uniform | 无 |
| E7.2 | 最优 backbone | 15 | Knee-BCRS | 无 |
| E7.3 | 最优 backbone | 10 | Knee-BCRS | 无 |
| E7.4 | 次优 backbone | 10/15 | Knee-BCRS | 无 |
| E7.5 | E7.2 + 次优模型 | 10/15 | Knee-BCRS | 双模型，仅可选 |

### 13.2 T4 测速协议

Kaggle 官方已确认：GPU notebook 可参加效率赛道；`RuntimeSeconds` 是 notebook 从开始执行到结束的完整 wall time，包含包安装、模型加载、DICOM 读取在内的全部耗时。因此下面的分项计时必须覆盖到 notebook 启动阶段，不能只统计"核心推理"部分。

1. 固定 Kaggle T4、相同容器、相同数据副本。
2. 20 studies warm-up，不计时。
3. 至少 200 studies 稳态计时；最终候选运行完整可用集合。
4. 同时记录冷启动和稳态时间。
5. 分别记录包导入/环境初始化、模型 checkpoint 加载、DICOM I/O、selector、backbone、aggregation、CSV 写入。
6. 报告 median、P95 和 1,300-study 投影，投影结果应以"完整 wall time 口径"汇总，与正式提交的计分方式一致。
7. 动态方案必须与相同 K 的静态 dense batch 比较。

### 13.3 Efficiency Score 敏感性

Private Leaderboard 的 $\max(AUC)$ 未知，因此不使用单一假定值。对：

$$
D=\max(AUC)-Benchmark\in\{0.25,0.30,0.35,0.40,0.45\}
$$

分别计算：

$$
E=-\frac{AUC}{D}+\frac{RuntimeSeconds}{32400}
$$

只有在多个 $D$ 假设下都靠前的配置，才视为稳健效率候选。

## 14. 消融执行顺序与组合控制

禁止对所有维度做笛卡尔积。严格按以下顺序：

1. 固定简单 head，筛 backbone。
2. 固定 backbone，筛分辨率与 K。
3. 固定 backbone/resolution/K，筛 selector 与 coverage。
4. 固定 selector，筛聚合头。
5. 固定架构，筛 loss 与微调深度。
6. 最终 2–3 个候选做完整 T4 efficiency frontier。

每阶段最多保留两个候选。被淘汰配置不因后续模块重新进入，除非有明确交互假设并预注册实验。

## 15. 必须回答的核心问题

| 问题 | 决定性实验 |
|---|---|
| DINOv3 是否优于 DINOv2？ | B2.0 vs B2.1，同输入数量和 head |
| ConvNeXt-Tiny 是否是更好的 T4 backbone？ | B2.1 vs B2.2，AUC–latency |
| PVTv2 多尺度是否帮助小病灶？ | B2.2 vs B2.3，rare-label per-class AUC |
| 更多切片还是更强 backbone 更划算？ | K3.x × B2 finalists 的等 runtime 比较 |
| 学习 selector 是否优于均匀采样？ | S4.0/S4.1 vs S4.2，相同 K |
| Coverage supervision 是否独立有效？ | S4.2 vs S4.3 |
| 频谱证据是否真正互补？ | S4.3/S4.4/S4.5/S4.6/S4.7 |
| BCRS 是否带来真实加速？ | E7.1 vs E7.2/E7.3 的端到端 T4 latency |
| 多尺度 DWConv 聚合是否值得？ | H5.2 vs H5.3 |
| Metadata 条件适配是否有效？ | H5.4 vs H5.5，跨 scanner/protocol 分桶 |
| Ranking loss 是否帮助官方指标？ | L6.0 vs L6.1 |

## 16. 失败条件与降级路径

### Selector 失败

若 Knee-BCRS 在 $K=15$ 时相对均匀采样 AUC 下降超过消融类噪声下限（约 0.01，见 4.3 节；uniform 和 Knee-BCRS 同 backbone，属于高相关比较）：

1. 保留 uniform sampling；
2. 将 selector 退化为训练期 attention regularizer；
3. 或仅让 selector 在中央窗口附近重排，不允许全局删除。

### 频谱分支失败

若频谱分支不优于参数匹配普通 DWConv，删除频谱 claim，保留 semantic + coverage selector。

### FLOPs 降但 latency 不降

若真实时间不改善：

- 不宣称 efficiency 增益；
- 优先使用固定 shape、固定 K、dense batching；
- 将动态操作移到 CPU 预处理或离线阶段；
- 检查 gather/scatter、kernel launch 和小 batch 利用率。

### Gold AUC 不稳定

若人工标签 CI 过宽：

- 不依据单次点估计挑模型；
- 使用 bootstrap 胜率和 per-label 一致方向；
- 优先选择 latency 更低、伪标签 OOF 更稳定的 Pareto 点。

## 17. 实验记录模板

每次实验必须记录：

```yaml
experiment_id:
git_commit:
fold_file:
label_source:
backbone:
pretrained_weights:
input_resolution:
physical_crop_mm:
slots:
windows_per_slot:
selector:
budget_k:
aggregation_head:
losses:
trainable_blocks:
seed:
comparison_type: ablation|architecture
baseline_experiment_id:
delta_vs_baseline_derived:
delta_vs_baseline_gold:
noise_threshold_applied:
oof_macro_auc_derived:
oof_macro_auc_gold:
per_label_auc:
evidence_coverage_at_k:
import_and_load_seconds:
mean_seconds_per_study:
p95_seconds_per_study:
dicom_seconds:
selector_seconds:
backbone_seconds:
peak_vram_gb:
projected_1300_study_hours:
decision:
```

## 18. 第一轮建议执行清单

按优先级执行：

1. 固定 folds、LLM 软标签和 DICOM 几何排序。
2. 复现 DINOv2-S、15 windows、per-label query baseline。
3. 用相同协议完成 DINOv3 ViT-S 和 DINOv3 ConvNeXt-Tiny frozen-feature screen。
4. 选择前两名，比较 224/256/280 和 K=5/10/15。
5. 训练高覆盖 teacher 并生成 window-level evidence cache。
6. 比较 uniform、semantic、semantic+coverage 三个 selector。
7. coverage 成立后才加入 spectral-only、gated、concat 和参数匹配普通卷积对照。
8. 对最优 selector 测试 K=5/10/15/25，并完成真实 T4 break-even 曲线。
9. 最后测试 Multi-scale DWConv1D、channel gate 和 metadata FiLM。
10. 只对 2–3 个最终 Pareto 点做完整 4-fold、部分微调和提交打包。

## 19. 预期最终候选

### 效率主候选

```text
DINOv3 ConvNeXt-Tiny
+ 5 contrast-aware slots
+ Knee-BCRS semantic/coverage selector
+ K=10–15 2.5D windows
+ per-label query aggregation
+ FP16、无 TTA、单模型
```

### 精度平衡候选

```text
DINOv3 ViT-S/16 或 DINOv2-S/14
+ K=15–25
+ Knee-BCRS
+ multi-scale DWConv1D aggregation
+ 部分解冻
```

### 安全 fallback

```text
DINOv2-S/14
+ uniform physical-scale sampling
+ K=15
+ per-label query aggregation
+ 无动态 routing
```

最终选择不依赖模型名称，而由 OOF Macro AUC、per-label 稳定性、evidence coverage 和 T4 端到端 runtime 的 Pareto frontier 决定。

## 20. 参考材料

- [比赛与数据总结](./RSNA-Knee-Competition-Summary.md)
- [BCRS Proposal](./attention-efficient/BCRS-Proposal.md)
- [BCRS Experiment Plan](./attention-efficient/BCRS-Experiment-Plan.md)
- [ESOD](./attention-efficient/ESOD.pdf)
- [SET](./attention-efficient/SET.pdf)
- [DINOv3](./attention-efficient/DINOv3.pdf)
- [EMCAD](./attention-efficient/EMCAD.pdf)
- [DAE-Former](./attention-efficient/DAE-Former.pdf)
- [TransDAE](./attention-efficient/TransDAE.pdf)
- [MedNeXt-lightweight](./attention-efficient/MedNeXt-lightweight.pdf)
- [MedNeXt](./attention-efficient/MedNeXt.pdf)
- [MedNeXtV2](./attention-efficient/MedNeXtV2.pdf)
- [HDNeXt](./attention-efficient/HDNeXt_Hybrid_Dynamic_MedNeXt.pdf)
