# Image Audit Decision Log

> 格式对应 [RSNA-Knee-Image-Audit-Plan.md](./RSNA-Knee-Image-Audit-Plan.md) 第 18 节模板。只记录已经有证据支撑的结论；没有证据的项目显式标注"待定"，不编造决策。

## Data snapshot

- audit date: 2026-08-08
- dataset: RSNA Knee Abnormality Detection（Kaggle 竞赛），train split
- 最近一次 `summarize` 覆盖：504,613 / 约 819,640 文件（约 61.6%），2,714 / 4,407 studies，**58 / 58 gold-labeled studies（100%）**
- audit 代码：`src/rsna_knee/audit/`（index / headers / pixels / coverage / summarize），本次 session 内新增/修复：`coverage.py`（gold 覆盖检查）、`ScanningSequence` 等多值字段的 parquet 崩溃修复、geometry/domain 检查（本文档记录的部分结论建立在刚加的代码上，见下方"待验证"标注）

## Decisions

- **slice ordering**：以几何法向量（`ImageOrientationPatient` 叉积）+ `ImagePositionPatient` 投影排序为主，`InstanceNumber` 仅作一致性交叉检查，不作为主排序键。**已验证**：504,613 个 header 记录里 `ImageOrientationPatient` 覆盖率 100%（0 个 Unknown plane），几何排序的前提成立。
- **plane source and fallback**：以几何推导的 plane 为准，`train_series.csv` 的 `Anatomical_Plane` 作为对照，冲突写入 `issues/geometry_failures.csv`，不静默采信任一方。检测代码已写好并通过单元测试（`tests/test_audit_geometry_domain.py`），**但还没有在全量真实数据上跑出实际冲突数字**——需要在数据端重新跑一次 `summarize` 才知道。
- **2D/2.5D/3D 决策**：**2.5D + geometry-aware position 维持为主线默认路线**；**轻量 3D 可以进入 Phase 2/3 的小规模探索候选**（不是主线）。依据：`series_spacing_cv` 中位数 8.9e-7、p99 仅 1.0e-4（切片间距几乎完美均匀），配合 100% 几何字段完整率，满足 Image-Audit-Plan 13.1 节"绝大多数主 series 可按物理位置可靠排序"的前提；但 `series_slice_count` 长尾明显（中位数 30，p99 达 160，max 320），固定尺寸 3D 需要明确的裁剪/重采样策略，且 T4 显存/延迟可控这一条（13.1 第三个门槛）完全没测过，是 3D 转正前必须补的实验（对应 AUD-08，见下）。
- **normalization**：未开始。N0–N4 五种候选方案的对比实验（Image-Audit-Plan 8.4）还没跑，不能编造结论。
- **series slots / missing-series behavior / window size/stride candidates / default K candidates**：不属于 audit 阶段能定的决策，留给 Phase 2/3 建模实验（见 Efficiency Experiment Plan）。
- **domain SSL decision**：**建议做 domain SSL pilot**（Image-Audit-Plan 13.3）。依据是真实 header 统计：厂商分布 Siemens 系约 43%、Philips 系约 31%、GE 系约 21%、Toshiba/Canon 约 4.5%（45 个不同型号），场强明显双峰于 1.5T 和 3.0T——域差异是真实存在的，不是猜测。
- **标签-域 shortcut 检查**：工具已写好（`label_domain_correlation.csv` + `domain_breakdown`），单元测试通过，**但还没在真实 58 个 gold study 上跑出实际数字**。即便跑出来，gold n=58 分散到多个厂商桶后每格样本量很小，只能当粗筛信号，不能当作可靠的 odds ratio——需要报告弱标签语料建好、样本量放大到全量 4,407 study 后才能做严谨版本。
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
| Geometry 一致性检查（plane 冲突/重复位置/InstanceNumber 不一致）尚未在全量数据上跑出结果 | 未知 | 代码已就绪并通过单元测试，需要在 AutoDL 上重新跑一次 `summarize` 才能拿到真实计数 | 工具就绪，待运行 |
| 标签-域相关性检查尚未在真实数据上跑出结果 | 未知 | 同上；跑出来后仍需结合弱标签语料复核（gold n=58 太小） | 工具就绪，待运行 |
| Fold 划分与 patient/duplicate 泄漏断言未执行 | 未知 | `patient_hash` 已就绪，需要跑 `data/split.py` 并验证 10.4 节的不相交断言 | 未开始 |
| 报告弱标签抽取（规则匹配）未实现 | 全部约 4,349 个非 gold study | 用户确认走规则匹配路线（无本地 LLM），需要单独实现，并按主办方澄清的"模糊即阴性"规则设计 | 未开始 |
| 重复检测（UID 级 / 像素哈希级 / 近重复 pHash 级）未实现 | 全部数据 | 目前只有 series 内部位置重复的窄 proxy；fold 划分前如果跳过这步，有 leakage 风险 | 未开始 |
