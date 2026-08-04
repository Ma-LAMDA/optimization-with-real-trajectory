# 2026-07-28 14×10 正确轨迹决策 SFT 实验

## 实验目的

本目录用于验证：从同一道网络故障题的多次真实 Agent 运行中，按答案正确性和轨迹完整性
严格筛选稳定样本，能否构造成按题隔离的 Qwen3.6-27B 最终决策 SFT 训练集与验证集。

与 2026-07-27 的多阶段人工策展实验不同，本实验扩大到 14 道题、每题 10 条轨迹，
但只训练最终 `decision` 阶段，不训练工具调用、过程规划或中间推理节点。

## 数据来源与筛选

来源为
`experiments/2026-07-27-ip_codex_train0629_14x10/results/runs/fullaccess/`，
共包含 14 道题的 140 条原始轨迹。

轨迹入选需要满足以下条件：

- 运行状态为 `succeeded`；
- 最终答案中的根因项与参考答案按顺序精确一致；
- 最终 Agent 事件与落盘答案一致；
- 最终结论前存在干净证据，不含工具名、API、文件路径或具体执行操作；
- 题目级 10 条轨迹全部满足准入条件，即该题正确率为 100%。

题 25、26、27、28 按人工指定整题排除，共排除 40 条轨迹；其余 10 道题全部达到
100% 正确率，共选入 100 条轨迹。`curation/trajectory_selection.json` 保存逐题质量统计、
每条轨迹的选择结果和排除原因。排除原因可能重叠，不能直接相加作为排除总数。

## 数据划分

| 集合 | 样本数 | 题号 |
| --- | ---: | --- |
| 训练集 | 90 | 13、14、17、18、87、88、91、92、93 |
| 验证集 | 10 | 94 |

划分策略为按 `case_id` 留出整题：题 94 的 10 条轨迹只进入验证集，训练集和验证集
不存在同题泄漏。全部样本当前为 `draft`，正式训练前仍需领域审核。

## 目录结构

```text
2026-07-28/
├── README.md
├── raw/                       # 14 题 × 10 次运行的标准化轨迹
├── curation/
│   └── trajectory_selection.json
└── sft/
    ├── manifest.json
    ├── qwen3_6_27b_reasoning_decision_train.jsonl
    └── qwen3_6_27b_reasoning_decision_validation.jsonl
```

## 重新生成与校验

在仓库根目录执行：

```powershell
python scripts/convert_codex_run_trajectories.py
python scripts/validate_codex_run_sft.py
```

最终样本数、划分、题目正确率、筛选条件和输出哈希以 `sft/manifest.json` 与
`curation/trajectory_selection.json` 为准。
