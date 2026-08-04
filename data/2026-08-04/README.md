# 2026-08-04 GPT-5.6-Sol 严格正确轨迹 SFT 数据

本目录由 `experiments/2026-08-02-ip_codex_gpt56-sol_100x10/` 的最终归档生成。
来源实验覆盖 100 个 query，共记账 1,343 个 attempt：814 个 accepted、440 个
incorrect、0 个 format error、55 个 infrastructure failure 和 34 个 interrupted。
实验采用 accepted-only 保留策略，因此只有 814 条 accepted 轨迹具备完整事件、答案、
判题和 metadata；其余 attempt 只保留来源状态计数。

814 条 accepted 候选全部再次通过以下准入检查，没有额外排除：

- metadata 状态为 accepted，模型为 `gpt-5.6-sol`；
- 独立判题为 parsed 且 correct，最终答案与不可变来源答案严格集合匹配；
- 判题、metadata、最终回答和事件文件哈希一致；
- 最终 Agent 事件与落盘答案一致；
- 存在前置证据消息，且证据摘要不含工具名、路径、API 或具体执行操作。

## 数据划分

| 集合 | 样本数 | query 数 |
| --- | ---: | ---: |
| 训练集 | 694 | 72 |
| 验证集 | 120 | 12 |
| 合计 | 814 | 84 |

训练和验证按 `case_id` 整题隔离，query 交集为 0。答案 label 按第一个分号后的
故障类型合并，忽略设备节点；每类只从恰有 10 条入选轨迹的 query 中选择题号最大的
2 个作为验证集：

| 故障类型 | 验证 query | 验证轨迹 |
| --- | --- | ---: |
| `全局STP未使能` | 12、11 | 20 |
| `STP BPDU被过滤` | 24、20 | 20 |
| `存在IP路由环路` | 40、39 | 20 |
| `存在MPLS标签环路` | 72、71 | 20 |
| `VRRP Master角色规划不合理` | 86、85 | 20 |
| `VRRP工作在非抢占模式` | 100、99 | 20 |

所有样本均为 `decision` 类型并标记为 `draft`，正式训练前仍需领域审核。

## 目录

```text
2026-08-04/
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

`raw/` 保存规范化后的 accepted 轨迹与来源哈希；`curation/` 固化逐轨迹准入结果、
确定性划分和完整统计；`sft/manifest.json` 固化来源文件、输出文件、样本数和哈希。

## 重新生成与校验

从仓库根目录执行：

```powershell
python -B scripts/convert_accepted_only_100x10_to_sft.py
python -B scripts/validate_accepted_only_100x10_sft.py
```

转换器只读取 `data/simulation/train_0629.jsonl`，不会编辑、覆盖、移动、重命名或删除
该不可变来源。校验器会独立复核来源哈希、814 条 raw/SFT 映射、每条 judgment 与事件
证据、694/120 数量、每类 2 个满 10 条验证 query、query 隔离及输出哈希。
