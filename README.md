# Qwen3.6-27B 网络故障轨迹与 SFT

本仓库用于构造、训练和评测网络故障诊断轨迹数据。当前保留三批日期数据：

- `2026-07-27`：人工策展的多阶段小样本基线；
- `2026-07-31`：首轮 100×10 正确轨迹决策 SFT，也是现有训练方案的默认数据；
- `2026-08-04`：GPT-5.6-Sol accepted-only 轨迹生成的最新决策 SFT 候选集。

训练输出只保留可复核的规划、推理或最终决策，不包含工具调用协议、命令、API 路径和
执行日志。所有 SFT 样本当前均为 `draft`，正式使用前仍需领域审核。

> **Thinking 强制策略**：自 2026-08-04 起，所有新的 Base/LoRA Agent 评测与
> 对比实验必须显式开启 thinking，并在结果中记录 `reasoning_output_tokens`。
> 完整规则见 [`docs/THINKING_POLICY.md`](docs/THINKING_POLICY.md)；未观察到可见
> thinking 输出的历史运行不得与 thinking-on 运行混合为同一能力结论。

## 当前数据

| 目录 | 用途 | 规模与划分 | 状态 |
| --- | --- | --- | --- |
| [`data/2026-07-27/`](data/2026-07-27/) | 多阶段策展基线 | 3 条原始轨迹；7 planning、2 reasoning、3 decision | 保留 |
| [`data/2026-07-31/`](data/2026-07-31/) | 当前 LoRA 训练基线 | 819 decision；训练 759、验证 60 | 已训练、已评测 |
| [`data/2026-08-04/`](data/2026-08-04/) | 最新 accepted-only 候选集 | 814 decision；训练 694、验证 120 | 已归档、待训练 |
| [`data/simulation/`](data/simulation/) | 原始仿真资料 | prompt、JSONL、配置与评测轨迹 | 不可变来源 |

### 2026-07-31 划分

来源为
[`experiments/2026-07-28-ip_codex_train0629_100x10/`](experiments/2026-07-28-ip_codex_train0629_100x10/)。
819 条严格正确轨迹按 `case_id` 整题隔离；六种故障类型各留出一题
（12、24、40、72、86、100），形成 759/60 训练验证划分。完整筛选口径见
[`data/2026-07-31/README.md`](data/2026-07-31/README.md)。

### 2026-08-04 划分

来源为
[`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/`](experiments/2026-08-02-ip_codex_gpt56-sol_100x10/)。
日期归档只统计 1,254 个模型有效 attempt：814 accepted、440 incorrect、0 format error；
基础设施失败与中断不进入日期归档。814 条 accepted 轨迹全部通过二次答案、事件、哈希和
证据清洁检查。

验证集按六种故障类型各留两道完整题。五类只选择成功率 100% 且有 10 条 accepted
轨迹的题；`全局STP未使能` 没有合格题，按显式回退规则选择成功率最高的 q12、q2。
完整候选、回退规则和逐题统计见
[`data/2026-08-04/README.md`](data/2026-08-04/README.md)。

## 数据规则

- `data/simulation/` 是不可变来源，只允许读取或复制，不得编辑、覆盖、移动或删除。
- 新的日期归档只记录模型有效结果：`accepted`、`incorrect` 和 `format_error`。
- 基础设施失败与中断可供 runner 临时控制流程，但不进入日期归档、报表或训练数据。
- 训练/验证必须按 `case_id` 整题隔离，禁止把同题重复轨迹随机分到两侧。
- accepted 样本必须通过参考答案、独立判题、最终事件、文件哈希和证据清洁检查。
- SFT 的 assistant 输出不得包含工具协议、工具名、命令、URL、API 路径或文件路径。

## 常用命令

### 校验现有数据

```powershell
python scripts/validate_sft.py
python scripts/validate_100x10_sft.py
python -B scripts/validate_accepted_only_100x10_sft.py
```

### 重新生成日期数据

```powershell
python scripts/convert_trajectories.py
python scripts/convert_100x10_accepted_to_sft.py
python -B scripts/convert_accepted_only_100x10_to_sft.py
```

已删除的 2026-07-28 历史留一数据仍可从保留的 14×10 来源实验重建：

```powershell
python scripts/convert_codex_run_trajectories.py
python scripts/validate_codex_run_sft.py
```

旧版 SFT 校验兼容 Git 工作区中的 LF/CRLF 换行差异；如果 2026-07-28 数据尚未重建，
Codex-run 校验器会明确要求先运行转换器或通过 `--data-root` 指定数据目录。

### 训练

默认训练方案使用 2026-07-31 的 759/60 划分：

```bash
bash scripts/run_seetacloud_lora_workflow.sh
```

环境、LoRA 参数、早停、部署和恢复流程统一记录在
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)，根 README 不再重复维护服务器路径和
逐步操作说明。

## 当前结果

### LoRA SFT

2026-07-31 的 759/60 数据从基座模型重新训练 2 epochs，最低验证 loss 出现在最终
`checkpoint-760`。确定性生成验证连续执行五次，每次均为 49/60 严格匹配、60/60
格式正确且无工具信息泄漏；合计 245/300（81.67%）。该结果用于训练流程和稳定性检查，
不是独立随机测试集结论。

详细训练参数和逐类错误见
[`docs/2026-07-31_QWEN36_27B_LORA_SFT_759X60_2EPOCH_REPEAT5_RESULT.md`](docs/2026-07-31_QWEN36_27B_LORA_SFT_759X60_2EPOCH_REPEAT5_RESULT.md)。

### 完整 Agent A/B

当前完整 Agent 对比在题 12、24、40、72、86、100 上各运行五次：

| 指标 | Base | LoRA checkpoint-760 +100 |
| --- | ---: | ---: |
| 严格正确 | 7/30（23.33%） | 12/30（40.00%） |
| 平均封顶耗时 | 32.37 分钟 | 24.21 分钟 |
| 超时 | 3 | 4 |

LoRA 严格准确率提高 16.67 个百分点且典型耗时下降，但超时没有改善。逐题结果、运行拓扑
和原始汇总见
[`experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/`](experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/)。

## 评测约定

- 最终答案必须能解析为题目要求的 `<result>...</result>` JSON 列表。
- 严格正确要求预测与一个完整可接受答案精确匹配；漏报、多报和错报均计错。
- 超时和 runner failure 均计为错误，不能把“请求完成”当作“回答正确”。
- Qwen3.6-27B Base/LoRA Agent eval 固定单个 vLLM TP=2 实例、两个 runner、总并发 2。
- 默认单次上限 3,600 秒，最大生成长度 8,000 个新 token。
- validation loss 用于选点，不单独作为能力提升结论；正式对比必须使用相同题目、prompt、
  工具链、并发、超时和判分口径。

基座部署 A/B 与历史全量评测保存在
[`experiments/2026-07-31-qwen36-27b-base-eval/`](experiments/2026-07-31-qwen36-27b-base-eval/)。

## 目录导航

```text
.
├── data/
│   ├── 2026-07-27/
│   ├── 2026-07-31/
│   ├── 2026-08-04/
│   └── simulation/
├── docs/
│   ├── TRAINING_PLAN.md
│   └── 训练与评测结果报告
├── experiments/
│   ├── 2026-07-27-ip_codex_train0629_14x10/
│   ├── 2026-07-28-ip_codex_train0629_10x10/
│   ├── 2026-07-28-ip_codex_train0629_100x10/
│   ├── 2026-07-31-qwen36-27b-base-eval/
│   ├── 2026-08-02-ip_codex_gpt56-sol_100x10/
│   └── 2026-08-02-qwen36-27b-heldout6-agent-ab/
└── scripts/
    ├── 数据转换与校验
    ├── LoRA 训练
    └── Base/LoRA Agent 评测
```

## 归档与清理状态

- `data/2026-07-27/` 保留：它是唯一包含 planning/reasoning 目标的人工策展基线。
- `data/2026-07-28/` 已删除：100 条 decision 样本已被更大数据替代，并可从来源实验重建。
- `experiments/2026-07-31-qwen36-27b-agent-ab/` 已删除：其四道题进入过训练集，不能作为
  泛化结论，已由六题完整 Agent A/B 替代。
- 07-27 14×10、07-28 10×10、两轮 100×10 来源实验继续保留，用于来源审计和复现。
- `2026-07-31-qwen36-27b-base-eval` 继续保留，因为部署决策和后续脚本仍引用该基线。

## 维护规则

每次推送 GitHub 的提交都必须同步更新本 README，确保数据、脚本、实验和当前结论一致。
