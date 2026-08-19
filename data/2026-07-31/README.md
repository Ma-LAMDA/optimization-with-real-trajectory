# 2026-07-31 严格正确轨迹 SFT 数据

本目录由 `experiments/2026-07-28-ip_codex_train0629_100x10/` 的完整归档生成。
来源共有 1,313 个 attempt，其中 819 条状态为 accepted，473 条为 rejected，
11 条为 interrupted，10 条为 infrastructure failure。

## 实验目的

本实验把 2026-07-28 的 14×10 决策 SFT 路线扩展到 100 道网络故障题：从多次真实
Codex Agent 排障 attempt 中只保留经独立判题和来源审计确认正确的轨迹，构造
Qwen3.6-27B LoRA 的最终决策 SFT 数据。实验同时按故障类型留出完整题目，用于检验
模型能否把正确轨迹中的诊断决策能力迁移到未参与训练的题目，而不是记住同题答案。

本目录记录的是训练数据构造与验证集划分实验；模型训练参数和训练后 Agent 评测结果
另见仓库 `docs/` 与相应 `experiments/` 目录。

819 条 accepted 轨迹均通过以下准入检查：

- 独立判题状态为 parsed 且 correct；
- 最终答案与不可变来源答案做严格故障集合匹配，支持显式参考答案备选集合；
- 判题、metadata 与最终答案哈希一致；
- 最终 Agent 事件与落盘答案一致；
- 存在前置证据消息，且不含工具名、API、路径或具体执行操作。

## 数据划分

| 集合 | 样本数 | 题号 |
| --- | ---: | --- |
| 训练集 | 759 | 78 个题号 |
| 验证集 | 60 | 题 12、24、40、72、86、100 |

划分键为 `case_id`，训练与验证题号交集为 0。按忽略故障节点后合并得到的 6 种
故障类型分别留出 1 个完整题号；优先选择该类型中严格正确轨迹数最多的题，数量并列
时选择题号最大的题。全部样本当前为 `draft`，正式能力训练前仍需领域审核。

## 目录

```text
2026-07-31/
├── README.md
├── raw/
│   └── qXXXX/run_YY/conversation_trajectory.json
├── curation/
│   ├── accepted_trajectory_selection.json
│   └── FILTER_REPORT.md
└── sft/
    ├── manifest.json
    ├── qwen3_6_27b_reasoning_decision_train.jsonl
    └── qwen3_6_27b_reasoning_decision_validation.jsonl
```

## 重新生成与校验

```powershell
python scripts/convert_100x10_accepted_to_sft.py
python scripts/validate_100x10_sft.py
```

转换器只读取 `data/simulation/train_0629.jsonl`，不会编辑、覆盖、移动或删除该不可变
来源文件。按答案 label 汇总的去重题目数和正确轨迹数、忽略故障节点后的故障类型
合并统计、每类验证题、按成功次数汇总的题目数量，以及逐题 attempt、成功数、
成功率、全部/成功 attempt 平均执行耗时、耗时覆盖、状态分布、SFT 入选数、划分和终态见
`curation/FILTER_REPORT.md`。报告中的耗时不含
随后启动的独立判题；11 条 interrupted attempt 没有耗时记录，也不会以 0 计入均值。
输出哈希与来源审计信息见 `sft/manifest.json`。
