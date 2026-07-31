# 100×10 完全正确轨迹过滤与 SFT 转换报告

- 来源实验：`experiments/2026-07-28-ip_codex_train0629_100x10`
- 来源 attempt：1313
- accepted 且独立判题完全正确的候选：819
- 通过 SFT 完整性与证据清洁检查：819
- 候选中排除：0
- 非 accepted attempt 过滤：494
- 训练集：809 条，83 个题号
- 验证集：10 条，题号 100
- 训练/验证题号交集：0
- 全部 attempt 平均耗时：327.551 秒（1302/1313 条有有效耗时）
- 成功 attempt 平均耗时：300.096 秒（819/819 条有有效耗时）

## 统计口径

- `Attempt`：题号目录下存在 `metadata.json` 的独立尝试；全局及逐题数量均与实验 `state.json` 复核一致。
- `成功`：来源 attempt 状态为 `accepted`；`SFT` 列才表示进入候选后继续通过独立判题、参考答案严格集合匹配、最终事件一致性、文件哈希和前置证据清洁检查的最终保留数。
- `平均耗时`：该题所有 `duration_seconds` 非空且非负 attempt 的算术平均，包含 accepted、rejected 和 infrastructure failure；`成功平均耗时` 只统计 accepted attempt。
- `duration_seconds` 由 runner 的单调时钟记录，覆盖 Codex 子进程执行及其输出实时落盘，不包含退出后的事件解析、审计整理和随后启动的独立判题，因此不是完整端到端耗时。
- 缺失耗时的 interrupted attempt 不以 0 计入。
- `有效耗时`：以 `有耗时记录数/Attempt` 展示平均值的实际分母；`SFT` 是最终保留并转换的轨迹数。

## 来源 attempt 状态

| 状态 | 数量 |
| --- | ---: |
| accepted | 819 |
| infrastructure_failure | 10 |
| interrupted | 11 |
| rejected | 473 |

## 按答案 label 统计题目与轨迹

仅统计最终进入 SFT 的严格正确轨迹。`题目数量` 是包含该 label 的去重题号数，`轨迹数量` 是包含该 label 的保留轨迹数。多标签答案会分别计入各 label，因此各行不能直接相加；去重总计为 84 题、819 条轨迹。

| 答案 label | 题目数量 | 轨迹数量 |
| --- | ---: | ---: |
| `AGG_SW_01;STP BPDU被过滤` | 12 | 120 |
| `Core_SW_01;VRRP Master角色规划不合理` | 14 | 68 |
| `Core_SW_01;VRRP工作在非抢占模式` | 10 | 100 |
| `Core_SW_01;全局STP未使能` | 6 | 47 |
| `Core_SW_02;VRRP Master角色规划不合理` | 14 | 72 |
| `Core_SW_02;VRRP工作在非抢占模式` | 4 | 39 |
| `Core_SW_02;全局STP未使能` | 6 | 53 |
| `PE1;存在IP路由环路` | 16 | 160 |
| `PE1;存在MPLS标签环路` | 16 | 160 |
| `PE2;存在IP路由环路` | 16 | 160 |
| `PE2;存在MPLS标签环路` | 16 | 160 |
| `PE3;存在IP路由环路` | 16 | 160 |
| `PE3;存在MPLS标签环路` | 16 | 160 |
| **去重总计** | **84** | **819** |

## 按故障类型合并统计

以答案 label 中第一个分号后的故障原因为合并键，忽略故障节点。同一题或同一轨迹内的多个设备级 label 若属于同一故障类型，只计 1 次。

| 故障类型 | 合并的设备级 label 数 | 题目数量 | 轨迹数量 |
| --- | ---: | ---: | ---: |
| `STP BPDU被过滤` | 1 | 12 | 120 |
| `VRRP Master角色规划不合理` | 2 | 14 | 140 |
| `VRRP工作在非抢占模式` | 2 | 14 | 139 |
| `全局STP未使能` | 2 | 12 | 100 |
| `存在IP路由环路` | 3 | 16 | 160 |
| `存在MPLS标签环路` | 3 | 16 | 160 |
| **去重总计** | **13** | **84** | **819** |

## 按成功次数统计题目数量

| 每题成功次数 | 题目数量 | 成功轨迹小计 |
| ---: | ---: | ---: |
| 0 | 16 | 0 |
| 3 | 1 | 3 |
| 4 | 2 | 8 |
| 9 | 2 | 18 |
| 10 | 79 | 790 |
| **总计** | **100** | **819** |

## 逐题过滤统计

| 题号 | Attempt | 成功 | 成功率 | 平均耗时（秒） | 成功平均耗时（秒） | 有效耗时 | Rejected | Interrupted | Infrastructure failure | SFT | 划分 | 终态 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 27 | 3 | 11.11% | 493.127 | 581.186 | 25/27 | 21 | 2 | 1 | 3 | train | abandoned_after_10_consecutive_wrong |
| 2 | 25 | 10 | 40.00% | 389.614 | 433.691 | 23/25 | 13 | 2 | 0 | 10 | train | completed_with_10_correct |
| 3 | 44 | 10 | 22.73% | 500.502 | 559.627 | 42/44 | 31 | 2 | 1 | 10 | train | completed_with_10_correct |
| 4 | 27 | 10 | 37.04% | 418.638 | 445.609 | 26/27 | 14 | 1 | 2 | 10 | train | completed_with_10_correct |
| 5 | 14 | 4 | 28.57% | 453.542 | 529.049 | 14/14 | 10 | 0 | 0 | 4 | train | abandoned_after_10_consecutive_wrong |
| 6 | 24 | 10 | 41.67% | 394.983 | 415.784 | 24/24 | 14 | 0 | 0 | 10 | train | completed_with_10_correct |
| 7 | 31 | 4 | 12.90% | 450.202 | 407.866 | 31/31 | 27 | 0 | 0 | 4 | train | abandoned_after_10_consecutive_wrong |
| 8 | 20 | 10 | 50.00% | 397.602 | 394.172 | 20/20 | 10 | 0 | 0 | 10 | train | completed_with_10_correct |
| 9 | 40 | 9 | 22.50% | 403.459 | 426.851 | 39/40 | 30 | 1 | 0 | 9 | train | abandoned_after_10_consecutive_wrong |
| 10 | 28 | 10 | 35.71% | 340.665 | 379.365 | 28/28 | 18 | 0 | 0 | 10 | train | completed_with_10_correct |
| 11 | 23 | 10 | 43.48% | 415.792 | 495.396 | 23/23 | 13 | 0 | 0 | 10 | train | completed_with_10_correct |
| 12 | 19 | 10 | 52.63% | 329.861 | 340.513 | 19/19 | 9 | 0 | 0 | 10 | train | completed_with_10_correct |
| 13 | 10 | 10 | 100.00% | 221.155 | 221.155 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 14 | 11 | 10 | 90.91% | 231.671 | 231.671 | 10/11 | 0 | 1 | 0 | 10 | train | completed_with_10_correct |
| 15 | 11 | 10 | 90.91% | 242.311 | 242.311 | 10/11 | 0 | 1 | 0 | 10 | train | completed_with_10_correct |
| 16 | 11 | 10 | 90.91% | 201.969 | 201.969 | 10/11 | 0 | 1 | 0 | 10 | train | completed_with_10_correct |
| 17 | 10 | 10 | 100.00% | 284.957 | 284.957 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 18 | 10 | 10 | 100.00% | 257.611 | 257.611 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 19 | 10 | 10 | 100.00% | 230.862 | 230.862 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 20 | 10 | 10 | 100.00% | 280.620 | 280.620 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 21 | 22 | 10 | 45.45% | 432.233 | 460.656 | 22/22 | 12 | 0 | 0 | 10 | train | completed_with_10_correct |
| 22 | 27 | 10 | 37.04% | 368.260 | 399.463 | 27/27 | 17 | 0 | 0 | 10 | train | completed_with_10_correct |
| 23 | 22 | 10 | 45.45% | 431.585 | 489.521 | 22/22 | 12 | 0 | 0 | 10 | train | completed_with_10_correct |
| 24 | 30 | 10 | 33.33% | 369.658 | 432.410 | 30/30 | 20 | 0 | 0 | 10 | train | completed_with_10_correct |
| 25 | 10 | 10 | 100.00% | 494.875 | 494.875 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 26 | 10 | 10 | 100.00% | 486.163 | 486.163 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 27 | 10 | 10 | 100.00% | 527.906 | 527.906 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 28 | 11 | 10 | 90.91% | 524.907 | 499.361 | 11/11 | 1 | 0 | 0 | 10 | train | completed_with_10_correct |
| 29 | 10 | 10 | 100.00% | 545.522 | 545.522 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 30 | 12 | 10 | 83.33% | 558.926 | 501.977 | 12/12 | 2 | 0 | 0 | 10 | train | completed_with_10_correct |
| 31 | 10 | 10 | 100.00% | 481.236 | 481.236 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 32 | 12 | 10 | 83.33% | 376.585 | 390.942 | 12/12 | 2 | 0 | 0 | 10 | train | completed_with_10_correct |
| 33 | 11 | 10 | 90.91% | 384.067 | 400.009 | 11/11 | 1 | 0 | 0 | 10 | train | completed_with_10_correct |
| 34 | 12 | 10 | 83.33% | 382.726 | 379.403 | 12/12 | 2 | 0 | 0 | 10 | train | completed_with_10_correct |
| 35 | 11 | 10 | 90.91% | 416.269 | 414.160 | 11/11 | 1 | 0 | 0 | 10 | train | completed_with_10_correct |
| 36 | 10 | 10 | 100.00% | 433.112 | 433.112 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 37 | 10 | 10 | 100.00% | 498.442 | 498.442 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 38 | 10 | 10 | 100.00% | 470.498 | 470.498 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 39 | 10 | 10 | 100.00% | 448.354 | 448.354 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 40 | 12 | 10 | 83.33% | 482.374 | 501.193 | 12/12 | 2 | 0 | 0 | 10 | train | completed_with_10_correct |
| 41 | 10 | 0 | 0.00% | 409.869 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 42 | 10 | 0 | 0.00% | 389.802 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 43 | 10 | 0 | 0.00% | 412.149 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 44 | 10 | 0 | 0.00% | 399.935 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 45 | 10 | 0 | 0.00% | 385.568 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 46 | 10 | 0 | 0.00% | 384.062 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 47 | 10 | 0 | 0.00% | 458.030 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 48 | 10 | 0 | 0.00% | 431.580 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 49 | 10 | 0 | 0.00% | 329.218 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 50 | 10 | 0 | 0.00% | 315.781 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 51 | 10 | 0 | 0.00% | 307.518 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 52 | 10 | 0 | 0.00% | 312.038 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 53 | 10 | 0 | 0.00% | 334.262 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 54 | 10 | 0 | 0.00% | 363.367 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 55 | 10 | 0 | 0.00% | 357.598 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 56 | 10 | 0 | 0.00% | 317.144 | — | 10/10 | 10 | 0 | 0 | 0 | excluded | abandoned_after_10_consecutive_wrong |
| 57 | 10 | 10 | 100.00% | 397.060 | 397.060 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 58 | 11 | 10 | 90.91% | 304.526 | 295.774 | 11/11 | 1 | 0 | 0 | 10 | train | completed_with_10_correct |
| 59 | 10 | 10 | 100.00% | 416.046 | 416.046 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 60 | 10 | 10 | 100.00% | 298.818 | 298.818 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 61 | 10 | 10 | 100.00% | 391.577 | 391.577 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 62 | 10 | 10 | 100.00% | 430.895 | 430.895 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 63 | 11 | 10 | 90.91% | 484.099 | 462.346 | 11/11 | 1 | 0 | 0 | 10 | train | completed_with_10_correct |
| 64 | 11 | 10 | 90.91% | 500.583 | 472.032 | 11/11 | 1 | 0 | 0 | 10 | train | completed_with_10_correct |
| 65 | 10 | 10 | 100.00% | 295.448 | 295.448 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 66 | 10 | 10 | 100.00% | 240.289 | 240.289 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 67 | 10 | 10 | 100.00% | 294.849 | 294.849 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 68 | 10 | 10 | 100.00% | 273.488 | 273.488 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 69 | 10 | 10 | 100.00% | 350.253 | 350.253 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 70 | 10 | 10 | 100.00% | 322.403 | 322.403 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 71 | 10 | 10 | 100.00% | 314.893 | 314.893 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 72 | 10 | 10 | 100.00% | 274.552 | 274.552 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 73 | 13 | 10 | 76.92% | 141.792 | 139.600 | 13/13 | 0 | 0 | 3 | 10 | train | completed_with_10_correct |
| 74 | 13 | 10 | 76.92% | 108.311 | 113.577 | 13/13 | 0 | 0 | 3 | 10 | train | completed_with_10_correct |
| 75 | 10 | 10 | 100.00% | 148.628 | 148.628 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 76 | 10 | 10 | 100.00% | 114.862 | 114.862 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 77 | 10 | 10 | 100.00% | 133.119 | 133.119 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 78 | 10 | 10 | 100.00% | 127.238 | 127.238 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 79 | 10 | 10 | 100.00% | 147.422 | 147.422 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 80 | 10 | 10 | 100.00% | 107.247 | 107.247 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 81 | 10 | 10 | 100.00% | 152.613 | 152.613 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 82 | 10 | 10 | 100.00% | 126.311 | 126.311 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 83 | 10 | 10 | 100.00% | 153.417 | 153.417 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 84 | 10 | 10 | 100.00% | 128.249 | 128.249 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 85 | 10 | 10 | 100.00% | 139.633 | 139.633 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 86 | 10 | 10 | 100.00% | 100.945 | 100.945 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 87 | 10 | 10 | 100.00% | 110.849 | 110.849 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 88 | 10 | 10 | 100.00% | 107.751 | 107.751 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 89 | 10 | 10 | 100.00% | 123.403 | 123.403 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 90 | 10 | 10 | 100.00% | 99.762 | 99.762 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 91 | 10 | 10 | 100.00% | 131.650 | 131.650 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 92 | 10 | 10 | 100.00% | 115.014 | 115.014 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 93 | 10 | 10 | 100.00% | 122.886 | 122.886 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 94 | 10 | 10 | 100.00% | 99.581 | 99.581 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 95 | 10 | 10 | 100.00% | 114.979 | 114.979 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 96 | 10 | 10 | 100.00% | 108.802 | 108.802 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 97 | 37 | 9 | 24.32% | 125.043 | 133.931 | 37/37 | 28 | 0 | 0 | 9 | train | abandoned_after_10_consecutive_wrong |
| 98 | 10 | 10 | 100.00% | 111.106 | 111.106 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 99 | 10 | 10 | 100.00% | 129.833 | 129.833 | 10/10 | 0 | 0 | 0 | 10 | train | completed_with_10_correct |
| 100 | 10 | 10 | 100.00% | 114.320 | 114.320 | 10/10 | 0 | 0 | 0 | 10 | validation | completed_with_10_correct |
| **总计** | **1313** | **819** | **62.38%** | **327.551** | **300.096** | **1302/1313** | **473** | **11** | **10** | **819** | — | — |

## 候选排除原因

无；819 条 accepted 候选全部通过独立答案复核、最终事件一致性、判题哈希和证据清洁检查。
