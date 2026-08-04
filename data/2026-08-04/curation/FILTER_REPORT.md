# Accepted-only 100×10 轨迹过滤与 SFT 转换报告

- 来源实验：`experiments/2026-08-02-ip_codex_gpt56-sol_100x10`
- 来源 attempt 记账：1343
- accepted 候选：814
- 通过独立复核与证据清洁检查：814
- 候选中排除：0
- 非 accepted attempt：529（来源只保留计数）
- 训练集：694 条，72 个题号
- 验证集：120 条，12 个题号
- 验证题号：11、12、20、24、39、40、71、72、85、86、99、100
- 训练/验证题号交集：0
- accepted 轨迹平均生成耗时：320.801 秒

## 统计口径

- 来源实验采用 accepted-only 保留策略；失败、格式错误、基础设施失败和中断只从 `state.json` 读取计数。
- 每条 accepted 候选重新核对 metadata、独立 judgment、参考答案、最终事件、文件哈希和前置证据清洁性。
- 训练/验证按 `case_id` 整题隔离；每种故障类型只从恰有 10 条入选轨迹的题中，按题号降序确定性选择 2 题。
- 所有样本均标记为 `draft`，正式训练前仍需领域审核。

## 来源 attempt 状态

| 状态 | 数量 |
| --- | ---: |
| accepted | 814 |
| format_error | 0 |
| incorrect | 440 |
| infrastructure_failure | 55 |
| interrupted | 34 |

## 按答案 label 统计

| 答案 label | 题目数量 | 轨迹数量 |
| --- | ---: | ---: |
| `AGG_SW_01;STP BPDU被过滤` | 12 | 103 |
| `Core_SW_01;VRRP Master角色规划不合理` | 13 | 50 |
| `Core_SW_01;VRRP工作在非抢占模式` | 10 | 100 |
| `Core_SW_01;全局STP未使能` | 6 | 56 |
| `Core_SW_02;VRRP Master角色规划不合理` | 13 | 90 |
| `Core_SW_02;VRRP工作在非抢占模式` | 4 | 40 |
| `Core_SW_02;全局STP未使能` | 6 | 55 |
| `PE1;存在IP路由环路` | 16 | 160 |
| `PE1;存在MPLS标签环路` | 16 | 160 |
| `PE2;存在IP路由环路` | 16 | 160 |
| `PE2;存在MPLS标签环路` | 16 | 160 |
| `PE3;存在IP路由环路` | 16 | 160 |
| `PE3;存在MPLS标签环路` | 16 | 160 |

## 按故障类型合并统计

故障类型取答案 label 第一个分号后的部分，忽略设备节点。

| 故障类型 | 设备级 label 数 | 题目数量 | 轨迹数量 |
| --- | ---: | ---: | ---: |
| `STP BPDU被过滤` | 1 | 12 | 103 |
| `VRRP Master角色规划不合理` | 2 | 14 | 140 |
| `VRRP工作在非抢占模式` | 2 | 14 | 140 |
| `全局STP未使能` | 2 | 12 | 111 |
| `存在IP路由环路` | 3 | 16 | 160 |
| `存在MPLS标签环路` | 3 | 16 | 160 |

## 验证集划分

| 故障类型 | 验证题号 | 验证轨迹 |
| --- | --- | ---: |
| `STP BPDU被过滤` | 24, 20 | 20 |
| `VRRP Master角色规划不合理` | 86, 85 | 20 |
| `VRRP工作在非抢占模式` | 100, 99 | 20 |
| `全局STP未使能` | 12, 11 | 20 |
| `存在IP路由环路` | 40, 39 | 20 |
| `存在MPLS标签环路` | 72, 71 | 20 |

## 逐题统计

`Attempt` 不含无法归属到题号的 interrupted 计数；`错误` 为 incorrect + format_error。

| 题号 | Attempt | 成功 | 错误 | 基础设施失败 | SFT | 划分 | 终态 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 26 | 10 | 12 | 4 | 10 | train | completed_with_10_correct |
| 2 | 19 | 10 | 5 | 4 | 10 | train | completed_with_10_correct |
| 3 | 27 | 6 | 18 | 3 | 6 | train | abandoned_after_10_consecutive_wrong |
| 4 | 29 | 10 | 15 | 4 | 10 | train | completed_with_10_correct |
| 5 | 27 | 10 | 13 | 4 | 10 | train | completed_with_10_correct |
| 6 | 25 | 10 | 11 | 4 | 10 | train | completed_with_10_correct |
| 7 | 31 | 5 | 20 | 6 | 5 | train | abandoned_after_20_total_wrong |
| 8 | 26 | 10 | 13 | 3 | 10 | train | completed_with_10_correct |
| 9 | 22 | 10 | 8 | 4 | 10 | train | completed_with_10_correct |
| 10 | 28 | 10 | 14 | 4 | 10 | train | completed_with_10_correct |
| 11 | 23 | 10 | 13 | 0 | 10 | validation | completed_with_10_correct |
| 12 | 12 | 10 | 2 | 0 | 10 | validation | completed_with_10_correct |
| 13 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 14 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 15 | 11 | 10 | 0 | 1 | 10 | train | completed_with_10_correct |
| 16 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 17 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 18 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 19 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 20 | 10 | 10 | 0 | 0 | 10 | validation | completed_with_10_correct |
| 21 | 31 | 9 | 20 | 2 | 9 | train | abandoned_after_20_total_wrong |
| 22 | 18 | 1 | 16 | 1 | 1 | train | abandoned_after_10_consecutive_wrong |
| 23 | 23 | 3 | 18 | 2 | 3 | train | abandoned_after_10_consecutive_wrong |
| 24 | 31 | 10 | 19 | 2 | 10 | validation | completed_with_10_correct |
| 25 | 13 | 10 | 1 | 2 | 10 | train | completed_with_10_correct |
| 26 | 13 | 10 | 1 | 2 | 10 | train | completed_with_10_correct |
| 27 | 14 | 10 | 2 | 2 | 10 | train | completed_with_10_correct |
| 28 | 13 | 10 | 1 | 2 | 10 | train | completed_with_10_correct |
| 29 | 11 | 10 | 0 | 1 | 10 | train | completed_with_10_correct |
| 30 | 24 | 10 | 1 | 13 | 10 | train | completed_with_10_correct |
| 31 | 30 | 10 | 8 | 12 | 10 | train | completed_with_10_correct |
| 32 | 20 | 10 | 9 | 1 | 10 | train | completed_with_10_correct |
| 33 | 13 | 10 | 1 | 2 | 10 | train | completed_with_10_correct |
| 34 | 14 | 10 | 4 | 0 | 10 | train | completed_with_10_correct |
| 35 | 12 | 10 | 1 | 1 | 10 | train | completed_with_10_correct |
| 36 | 17 | 10 | 6 | 1 | 10 | train | completed_with_10_correct |
| 37 | 11 | 10 | 1 | 0 | 10 | train | completed_with_10_correct |
| 38 | 11 | 10 | 0 | 1 | 10 | train | completed_with_10_correct |
| 39 | 12 | 10 | 1 | 1 | 10 | validation | completed_with_10_correct |
| 40 | 11 | 10 | 1 | 0 | 10 | validation | completed_with_10_correct |
| 41 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 42 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 43 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 44 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 45 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 46 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 47 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 48 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 49 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 50 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 51 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 52 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 53 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 54 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 55 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 56 | 10 | 0 | 10 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 57 | 15 | 10 | 5 | 0 | 10 | train | completed_with_10_correct |
| 58 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 59 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 60 | 12 | 10 | 2 | 0 | 10 | train | completed_with_10_correct |
| 61 | 13 | 10 | 3 | 0 | 10 | train | completed_with_10_correct |
| 62 | 12 | 10 | 2 | 0 | 10 | train | completed_with_10_correct |
| 63 | 11 | 10 | 1 | 0 | 10 | train | completed_with_10_correct |
| 64 | 11 | 10 | 1 | 0 | 10 | train | completed_with_10_correct |
| 65 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 66 | 12 | 10 | 2 | 0 | 10 | train | completed_with_10_correct |
| 67 | 12 | 10 | 2 | 0 | 10 | train | completed_with_10_correct |
| 68 | 11 | 10 | 1 | 0 | 10 | train | completed_with_10_correct |
| 69 | 12 | 10 | 2 | 0 | 10 | train | completed_with_10_correct |
| 70 | 11 | 10 | 1 | 0 | 10 | train | completed_with_10_correct |
| 71 | 10 | 10 | 0 | 0 | 10 | validation | completed_with_10_correct |
| 72 | 13 | 10 | 3 | 0 | 10 | validation | completed_with_10_correct |
| 73 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 74 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 75 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 76 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 77 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 78 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 79 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 80 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 81 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 82 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 83 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 84 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 85 | 10 | 10 | 0 | 0 | 10 | validation | completed_with_10_correct |
| 86 | 10 | 10 | 0 | 0 | 10 | validation | completed_with_10_correct |
| 87 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 88 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 89 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 90 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 91 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 92 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 93 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 94 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 95 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 96 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 97 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 98 | 10 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 99 | 10 | 10 | 0 | 0 | 10 | validation | completed_with_10_correct |
| 100 | 10 | 10 | 0 | 0 | 10 | validation | completed_with_10_correct |

## 候选排除原因

无；814 条 accepted 候选全部通过复核。
