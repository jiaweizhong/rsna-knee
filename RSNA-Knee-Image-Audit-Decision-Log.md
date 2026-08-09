# Image Audit Decision Log

> 格式对应 [RSNA-Knee-Image-Audit-Plan.md](./RSNA-Knee-Image-Audit-Plan.md) 第 18 节模板。只记录已经有证据支撑的结论；没有证据的项目显式标注"待定"，不编造决策。

## Data snapshot

- audit date: 2026-08-08
- dataset: RSNA Knee Abnormality Detection（Kaggle 竞赛），train split
- 最近一次 `summarize` 覆盖：504,613 / 约 819,640 文件（约 61.6%），2,714 / 4,407 studies，**58 / 58 gold-labeled studies（100%）**
- audit 代码：`src/rsna_knee/audit/`（index / headers / pixels / coverage / summarize），本次 session 内新增/修复：`coverage.py`（gold 覆盖检查）、`ScanningSequence` 等多值字段的 parquet 崩溃修复、geometry/domain 检查——geometry 三项检查和标签-域相关性均已在 504,613 文件全量数据上跑出真实结果（见下）

## Decisions

- **slice ordering**：以几何法向量（`ImageOrientationPatient` 叉积）+ `ImagePositionPatient` 投影排序为主，`InstanceNumber` 仅作一致性交叉检查，不作为主排序键。**已验证**：504,613 个 header 记录里 `ImageOrientationPatient` 覆盖率 100%（0 个 Unknown plane），几何排序的前提成立。
- **plane source and fallback**：以几何推导的 plane 为准，`train_series.csv` 的 `Anatomical_Plane` 作为对照，冲突写入 `issues/geometry_failures.csv`，不静默采信任一方。**已验证（504,613 文件 / 15,019 series 全量）：`plane_conflict_series = 0`，零冲突。**
- **InstanceNumber 与几何顺序一致性**：**已验证（15,019 series 全量）：`instance_number_disagrees_with_geometry` 命中数 = 0**。原始 spearman 相关系数分布呈双峰（p25≈-1.0，p50/p75/p95/p99≈+1.0），但这是方向约定问题（几何法向量的"正方向"是任意选定的），用 `abs(spearman) < 0.9` 判定后没有一例真正的顺序不一致。`duplicate_position_series` 同样为 0。
- **2D/2.5D/3D 决策**：**2.5D + geometry-aware position 维持为主线默认路线**；**轻量 3D 可以进入 Phase 2/3 的小规模探索候选**（不是主线）。依据：`series_spacing_cv` 中位数 8.9e-7、p99 仅 1.0e-4（切片间距几乎完美均匀），配合几何字段 100% 完整、plane 冲突 0、顺序不一致 0——Image-Audit-Plan 13.1 节"绝大多数主 series 可按物理位置可靠排序、spacing 异常有明确 fallback"这两条前提**已经用全量数据坐实**；但 `series_slice_count` 长尾明显（中位数 30，p99 达 160，max 320），固定尺寸 3D 需要明确的裁剪/重采样策略，且 T4 显存/延迟可控这一条（13.1 第三个门槛）完全没测过，是 3D 转正前必须补的实验（对应 AUD-08，见下）。
- **normalization**：未开始。N0–N4 五种候选方案的对比实验（Image-Audit-Plan 8.4）还没跑，不能编造结论。
- **series slots / missing-series behavior / window size/stride candidates / default K candidates**：不属于 audit 阶段能定的决策，留给 Phase 2/3 建模实验（见 Efficiency Experiment Plan）。
- **domain SSL decision**：**建议做 domain SSL pilot**（Image-Audit-Plan 13.3）。依据是真实 header 统计：厂商分布 Siemens 系约 43%、Philips 系约 31%、GE 系约 21%、Toshiba/Canon 约 4.5%（45 个不同型号），场强明显双峰于 1.5T 和 3.0T——域差异是真实存在的，不是猜测。
- **标签-域 shortcut 检查**：**已跑出真实数字**（58 个 gold study 分布：Siemens 22 / Philips 18 / GE 16 / Canon-Toshiba 2）。大部分标签在 GE/Philips/Siemens 三家之间的差距在小样本噪声量级内。**唯一值得关注的信号：Baker's 囊肿在 GE 上是 0/16（0%），Philips 22.2%，Siemens 31.8%**——16 个 study 零阳性，若真实患病率约 20%，纯属巧合的概率约 2.8%，有一定但不确定的证据强度。Medial/Lateral OA、Lateral Meniscus 上 Siemens 也有弱的系统性偏高模式（约 2 倍于 GE/Philips）。**结论：不算确凿证据（gold n 太小，Canon/Toshiba n=2 完全不可用），但 Baker's 囊肿这一项值得在报告弱标签语料（全量 4,407 study）建好后优先复核，选 backbone/aggregation 时如果 Baker's 类的表现在验证集上异常好或异常差，先检查是不是被 scanner 分布带偏。
- **report multimodal decision（报告弱标签）**：抽取方式定为**本地规则匹配**（用户确认没有本地 LLM 环境，暂不用开源模型路线）。报告原文不得发送给任何商业 LLM API（含 Claude），已写入三个计划文档作为合规约束。规则匹配的具体实现**还没开始写**。
- **fold grouping**：未生成。`patient_hash`（salted SHA256，header 阶段已计算）是生成 patient-grouped fold 的必要材料，已经就绪，但 `data/split.py` 没有针对真实数据跑过，10.4 节要求的"跨 fold 交集为 0"断言没有验证过。
- **quarantine/fallback policy**：3 个文件确认为真实像素数据损坏（"字节数少于预期"，`issues/decode_failures.csv` 里有具体路径），当前只是"跳过 + 记录"，还没有正式的 `quarantine_candidates.csv` 和训练时的显式排除逻辑。

## Runtime

- 目前只有 audit 阶段的单文件耗时：header 解析均值 ~0.97ms/文件，pixel decode 均值 ~0.41ms/文件（504,613 文件全量测得，0 个 header 错误，3 个 pixel 错误）。
- **AUD-08（T4 端到端 runtime benchmark）完全没做**——上面这两个数字不能代表正式推理管线的耗时，因为 selector、backbone、aggregation 都还不存在，也没有测过包导入/模型加载时间（这块在效率赛道正式计入 `RuntimeSeconds`，见 Competition Summary）。
- 1,300-study 投影：无法计算，前提条件（端到端管线）不存在。

## Open warnings

| 问题 | 影响范围 | 处理建议 | 状态 |
|---|---|---|---|
| Transfer syntax 100% 为单一值（Explicit VR Little Endian） | 基于 61.6% 样本 | 可能是选择偏差——剩余 38.4% 未解压数据可能包含 JPEG Lossless/2000/Implicit VR；数据补充后必须复核，现在不能得出"不需要 JPEG decoder"的结论 | 待复核 |
| 3 个文件 pixel 数据真实损坏 | 3/504,613（0.0006%） | 加入正式 quarantine 名单，训练前排除 | 已定位，未正式 quarantine |
| MagneticFieldStrength 约 3.8% 缺失 | ~1.9 万 / 504,613 条记录 | 可用同 series 内其他文件回填，不能假设每条记录都有该字段 | 已知，待处理 |
| Manufacturer 字符串未归一化 | 全部 header 记录 | `_manufacturer_family()` 仅在 label-domain 检查里做了粗粒度分组；`series_inventory.parquet` 等其他产物仍是原始字符串，下游使用时需注意 | 部分处理 |
| Geometry 一致性检查（plane 冲突/重复位置/InstanceNumber 不一致） | 15,019 series 全量 | 三项全部为 0，无需处理 | **已完成，结果干净** |
| Baker's 囊肿 prevalence 在 GE 上为 0/16，Philips/Siemens 为 22–32% | 58 个 gold study 里的 GE 子集（n=16） | 报告弱标签语料建好后在全量 4,407 study 上复核；backbone/selector 筛选时如果 Baker's 这一类指标异常，先排查 scanner 分布 | 已识别，待全量复核 |
| Fold 划分与 patient/duplicate 泄漏断言未执行 | 未知 | `patient_hash` 已就绪，需要跑 `data/split.py` 并验证 10.4 节的不相交断言 | 未开始 |
| 报告弱标签抽取（规则匹配）未实现 | 全部约 4,349 个非 gold study | 用户确认走规则匹配路线（无本地 LLM），需要单独实现，并按主办方澄清的"模糊即阴性"规则设计 | 未开始 |
| 重复检测（UID 级 / 像素哈希级 / 近重复 pHash 级）未实现 | 全部数据 | 目前只有 series 内部位置重复的窄 proxy；fold 划分前如果跳过这步，有 leakage 风险 | 未开始 |
