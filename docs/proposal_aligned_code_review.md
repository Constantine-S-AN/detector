# Proposal-aligned Code Review (Staff-level)

## Scope checked
- Repository structure, pipeline entry points, config, attribution backends, density features, detector training/evaluation, CLI/API, tests, and docs.
- Key paths reviewed:
  - Data/pipeline scripts: `scripts/build_controlled_dataset.py`, `scripts/run_attribution.py`, `scripts/build_features.py`, `scripts/train_detector.py`, `scripts/evaluate_detector.py`, `scripts/demo_end_to_end.sh`
  - Attribution: `ads/attribution/*`
  - Features/detectors/eval: `ads/features/density.py`, `ads/detector/*`, `ads/eval/metrics.py`
  - Product surface: `ads/service.py`, `ads/cli.py`, `ads/api.py`
  - Tests: `tests/*`

---

## 1) 对齐矩阵（proposal requirement -> repo mapping -> gap）

| Proposal requirement | Repo mapping | 现状 | 缺口 | 建议改法 |
|---|---|---|---|---|
| 核心假设：faithful 的 influence 更尖、hallucination 更散 | `ads/features/density.py`, `tests/test_toy_backend.py` | 已有 entropy / top-share / peakiness / gini / effective_k，并有 peaked vs diffuse 测试 | 目前“信号成立”高度依赖 toy backend 的文本触发规则，存在标签泄漏风险；真实研究证据不足 | 用真实或半真实 attribution（优先 DDA）重跑；移除会直接把答案措辞映射到分布形状的 heuristic，改为数据条件触发 |
| H@K（top-K influence entropy） | `ads/features/density.py` | 代码计算的是“对输入分数全量归一化后熵，再除以 log(n)” | 缺少显式 K 参数与“先取 top-K 再归一化”的实现；与 proposal 公式不完全一致 | 新增 `compute_h_at_k(scores, k)`：排序截断 top-K，softmax/非负归一后再算熵；并记录 `k_requested` 与 `k_effective` |
| peakiness ratio（Top1/Top5） | `ads/features/density.py` | 已实现 `top1_share / top5_share` | 与 proposal 中 Top1/Top5 Score 近似但未区分 score 版/概率版，语义未在文档固定 | 同时导出 `peakiness_ratio_score` 与 `peakiness_ratio_prob`，并在报告固定定义 |
| max-influence threshold detector signal | `ads/features/density.py`, `ads/detector/threshold.py` | 有 `max_score` + `max_score_floor` 及 abstain 逻辑 | 未作为独立 detector feature 做系统化阈值扫描与报告 | 在 `scripts/evaluate_detector.py` 增加 max_score-only baseline + threshold sweep |
| 二分类 detector（阈值或 logistic） | `ads/detector/threshold.py`, `ads/detector/logistic.py` | 已有两种 detector | logistic 没有 class imbalance 配置、概率校准步骤 | 增加 `class_weight` 选项、Platt/Isotonic 校准对照，并在报告写明是否启用 |
| 评估 ROC-AUC / PR-AUC / calibration | `ads/eval/metrics.py`, `scripts/evaluate_detector.py` | ROC-AUC/PR-AUC/Brier/ECE/curves 已有 | 缺少统计置信区间和多 seed 报告；calibration 仅 ECE 点估计 | 加 bootstrap CI + multi-seed 聚合脚本；报告均值±方差 |
| Attribution 方法优先 DDA，并对 TRAK/CEA 做 baseline | `ads/attribution/dda_backend.py`, `trak_backend.py`, `cea_backend.py` | 三者都只是 placeholder | proposal 主线尚未落地，无法形成方法对照实验 | PR 优先实现 DDA adapter（最小可运行），再补 TRAK/CEA 最小对照接口 |
| Deliverable: Scanner CLI/API 输出 groundedness + top training snippets | `ads/cli.py`, `ads/api.py`, `ads/service.py` | CLI/API 已输出 groundedness 和 top influential items | 缺少 snippet 级过滤/脱敏策略、解释字段较薄 | 输出 `evidence_summary`（top-k 片段摘要 + score mass）与可选 redaction |
| 可复现实验 pipeline（数据→attribution→metrics→detector→报告） | `scripts/demo_end_to_end.sh`, `scripts/write_run_manifest.py` | 端到端脚本和 run manifest 基本齐全 | 复现真实性弱（toy 数据与 toy attribution）；环境依赖与 lock 不完全闭环 | 增加“research mode” pipeline（真实 attribution）及固定依赖快照 |

---

## 2) Code Review（Correctness / Reproducibility / Research validity / Engineering quality）

## Top issues（P0/P1/P2）

### P0-1: Proposal 主方法（DDA）未实现，TRAK/CEA 也未形成可运行 baseline
- **证据**：`DDABackend.compute` 直接 `NotImplementedError`，`CEABackend` 同样未实现；`TRAKBackend` 仅依赖探测 + `NotImplementedError`。  
- **风险**：无法验证“density 在 DDA 下是否成立”，proposal 核心贡献无法交付。  
- **修复建议**：先落一个可运行 DDA 最小版（固定模型/检查点/缓存协议），并提供统一 attribution schema（id/score/text/meta/source）。  
- **建议测试**：
  - 单测：DDA backend 返回长度=K、按 score 降序、score 非负。
  - 集成：`run_attribution.py --backend dda` 产出可被 `build_features.py` 消费。

### P0-2: Toy attribution 对答案文本触发模式，存在严重标签泄漏/评估污染
- **证据**：`ToyAttributionBackend._resolve_mode` 用 answer 中 `speculative/uncertain/...` 等 token 判定 `diffuse`；而数据集中的 hallucinated 模板显式包含这些 token。  
- **风险**：模型可能只是在“识别模板词”而非真实 groundedness；ROC/PR 高分可能虚高。  
- **修复建议**：
  1. 训练/评估集去除可直接触发模式的词特征；
  2. toy backend 模式改为由样本元数据控制（而非答案文本）；
  3. 报告中将 toy 结果标注为 sanity-only，不作为主结论。
- **建议测试**：
  - 对抗改写测试：不改标签仅改措辞，分数不应系统性翻转。
  - 泄漏探针：仅用答案词袋训练 baseline，验证其性能上限并对比 attribution 模型。

### P1-1: H@K 公式落地不严格（缺少显式 K 截断语义）
- **证据**：`compute_density_features` 使用全部输入 scores 归一化后算熵，未显式提供 `K` 参数执行“top-K后归一化”。  
- **风险**：与 proposal 公式不一致，实验结论可解释性下降；不同 backend 返回长度不一致时不可比。  
- **修复建议**：新增 `k` 参数，统一执行：排序→截断 K→归一化→`H@K=-Σp_i log p_i`（可选 normalized）。
- **建议测试**：
  - 固定分布下 `H@5 <= H@10` 的可解释性测试（含特例）。
  - 当 `K > n` 时 `k_effective=n`，结果稳定。

### P1-2: 训练/评估稳健性不足（单次 split、单 seed、无 CI）
- **证据**：`train_detector.py` 采用一次 `train_test_split`；`evaluate_detector.py` 输出点估计指标。  
- **风险**：结果对随机划分敏感，研究结论不稳健。  
- **修复建议**：增加 multi-seed 重复试验 + bootstrap CI（ROC/PR/ECE/Brier）。
- **建议测试**：
  - 回归测试：固定小数据集，多次运行结果统计量 shape/字段稳定。

### P1-3: 校准评估只有 ECE，缺少校准建模对照
- **证据**：`compute_metrics_bundle` 计算 ECE/Brier 和 calibration points，但无 calibration model（Platt/Isotonic）开关。  
- **风险**：当 score 可分但未校准时，部署阈值和置信解释可能失真。  
- **修复建议**：在 logistic 流程加入可选校准器，并报告 pre/post calibration。
- **建议测试**：
  - 单测：启用校准后输出概率范围正确、曲线点存在。

### P2-1: Detector feature 语义文档不够严格（score版 vs prob版 peakiness）
- **证据**：当前 `peakiness_ratio = top1_share / top5_share`，但 proposal 表述接近“Top1 Score / Top5 Score”。  
- **风险**：论文/实现术语不一致，复现方难以对齐。  
- **修复建议**：在 README + report 明确数学定义，并双轨导出两种版本避免歧义。
- **建议测试**：
  - 单测验证两种 ratio 在统一缩放下的性质（score 版比例不变、prob 版同样不变）。

### P2-2: API explainability payload 可读性可增强
- **证据**：`/scan` 已返回 `top_influential` 原始项，但缺少聚合解释字段。  
- **风险**：下游产品难直接展示“为什么这个分数”。  
- **修复建议**：增加 `evidence_summary`（top-1、top-5 mass、snippet highlights）。
- **建议测试**：
  - API schema test：新增字段存在且与原始 top_influential 一致。

---

## 3) Proposal-aligned PR 拆分计划（可执行）

## PR1 — 修正 density 指标定义（H@K/peakiness/max-threshold）
- **目标**：确保公式实现与 proposal 严格对齐，消除 K 语义歧义。
- **改动文件清单**：
  - `ads/features/density.py`
  - `scripts/build_features.py`
  - `tests/test_density.py`
  - `README.md`（指标定义段）
- **DoD**：
  - 支持 `--h-k` 或显式 `k` 输入，输出 `h_at_k`, `k_requested`, `k_effective`。
  - 新增 `peakiness_ratio_score` 与 `peakiness_ratio_prob`（至少文档明确一版为主）。
  - 现有 pipeline 不破坏（向后兼容 CSV 列名或提供迁移）。
- **最小测试**：
  - unit: H@K 正确性 + 边界条件。
  - integration: `build_features.py` 在旧/新 scores 输入下都可运行。

## PR2 — 去除 toy 泄漏并建立 research-valid 基准
- **目标**：让 toy pipeline 至少不依赖答案模板词触发归因模式。
- **改动文件清单**：
  - `ads/attribution/toy_backend.py`
  - `scripts/build_controlled_dataset.py`
  - `scripts/build_stress_dataset.py`
  - `tests/test_toy_backend.py`
- **DoD**：
  - toy 模式由样本元数据驱动，不再由 answer 关键词驱动。
  - 对抗改写下标签不变时 attribution-density 不应系统漂移。
- **最小测试**：
  - unit: `_resolve_mode` 不读取 answer 文本（或仅在 debug 模式允许）。
  - integration: demo pipeline 指标仍可生成。

## PR3 — DDA 最小可运行实现 + baseline接口统一
- **目标**：补齐 proposal 主方法可执行路径。
- **改动文件清单**：
  - `ads/attribution/dda_backend.py`
  - `ads/attribution/base.py`
  - `scripts/run_attribution.py`
  - `tests/test_optional_backends.py`（或新增 `test_dda_backend.py`）
  - `README.md`（运行说明）
- **DoD**：
  - `--backend dda` 可跑通并产生合法 attribution 输出。
  - 明确依赖、缓存目录和失败报错策略。
- **最小测试**：
  - unit: DDA adapter 输出 schema 校验。
  - integration: attribution→features 串联通过。

## PR4 — 评估稳健性与校准增强（multi-seed + CI + calibration）
- **目标**：提升研究可信度与部署可用性。
- **改动文件清单**：
  - `scripts/train_detector.py`
  - `scripts/evaluate_detector.py`
  - `ads/eval/metrics.py`
  - `ads/eval/plots.py`
  - `tests/test_metrics.py`
- **DoD**：
  - 输出多 seed 聚合表、bootstrap CI。
  - 提供 pre/post calibration 指标对照（至少 ECE/Brier）。
- **最小测试**：
  - unit: CI 计算与字段完整性。
  - integration: 评估脚本产物（json/csv/plots）完整。

## PR5 — Scanner explainability productization（CLI/API）
- **目标**：满足“groundedness + top snippets”的可消费输出。
- **改动文件清单**：
  - `ads/service.py`
  - `ads/api.py`
  - `ads/cli.py`
  - `tests/test_api.py`, `tests/test_service.py`
  - `site/*`（若前端展示要同步）
- **DoD**：
  - API/CLI 增加 `evidence_summary`、top-k mass、可选 redaction。
  - 与当前输出字段向后兼容。
- **最小测试**：
  - API contract test + golden json snapshot。

---

## 4) 结论（给立刻开工的优先级）
1. **先做 PR1 + PR2**：先把“指标定义正确 + 泄漏去除”修到研究可解释基线。
2. **再做 PR3**：尽快把 DDA 跑通，避免 proposal 主线空转。
3. **最后 PR4 + PR5**：把研究可信度和产品可用性补齐到可答辩/可演示。

如果你希望，我下一步可以直接按 **PR1** 输出“精确到函数签名和测试断言”的 patch 计划（可以直接给工程师开始写代码）。
