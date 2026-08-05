# 2026-08-04 GPT-5.6-Sol 严格正确轨迹 SFT 数据

本目录由 `experiments/2026-08-02-ip_codex_gpt56-sol_100x10/` 的最终归档生成。
来源实验覆盖 100 个 query；归档按规则只统计 1,254 个模型有效 attempt：814 个
accepted、440 个 incorrect 和 0 个 format error。基础设施失败和中断不进入本目录的
计数、报表或训练数据。实验采用 accepted-only 保留策略，因此只有 814 条 accepted
轨迹具备完整事件、答案、判题和 metadata。

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
故障类型合并，忽略设备节点。成功率按
`accepted / (accepted + incorrect + format_error)` 计算；基础设施失败和中断不计。

## 每个 label 的 100% 成功率候选题

候选题还必须有 10 条入选轨迹。

| 故障类型 | 来源题数 | 满10条题数 | 100%成功率题数 | 合格题号 |
| --- | ---: | ---: | ---: | --- |
| `全局STP未使能` | 12 | 10 | 0 | — |
| `STP BPDU被过滤` | 12 | 9 | 8 | 13–20 |
| `存在IP路由环路` | 32 | 16 | 2 | 29、38 |
| `存在MPLS标签环路` | 16 | 16 | 4 | 58、59、65、71 |
| `VRRP Master角色规划不合理` | 14 | 14 | 14 | 73–86 |
| `VRRP工作在非抢占模式` | 14 | 14 | 14 | 87–100 |

除 `全局STP未使能` 外，每类从合格题中按题号降序选择 2 题。该类没有 100% 候选，
按已确认的显式回退规则，从满 10 条题中依次按成功率、题号降序选择 q12、q2。

| 故障类型 | 验证 query | 成功率 | 验证轨迹 |
| --- | --- | --- | ---: |
| `全局STP未使能` | 12、2 | 83.33%、66.67% | 20 |
| `STP BPDU被过滤` | 20、19 | 100%、100% | 20 |
| `存在IP路由环路` | 38、29 | 100%、100% | 20 |
| `存在MPLS标签环路` | 71、65 | 100%、100% | 20 |
| `VRRP Master角色规划不合理` | 86、85 | 100%、100% | 20 |
| `VRRP工作在非抢占模式` | 100、99 | 100%、100% | 20 |

上述 814 条旧版样本均为 `decision` 类型并标记为 `draft`，正式训练前仍需领域审核。

## 快跑版：每题一条最佳轨迹的原生多轮 SFT

为尽快得到 0804 的首轮训练结果，暂不对同题的 10 条成功轨迹聚类。转换器在每个有
accepted 轨迹的题中确定性选择一条质量最高的轨迹，训练题与验证题都只使用这一条：

| 集合 | query | 最佳轨迹 | SFT 节点样本 |
| --- | ---: | ---: | ---: |
| 训练集 | 72 | 72 | 318 |
| 验证集 | 12 | 12 | 53 |
| 合计 | 84 | 84 | 371 |

训练/验证沿用上面的冻结整题划分，交集为 0。验证集每个 label 保留 2 题；其中五类的
验证题均为 100% 成功率，`全局STP未使能` 因不存在 100% 候选，仍使用已确认的回退题
q12、q2。最佳轨迹的自动评分首先检查最终回答之前是否明确出现根因和故障设备，再检查
策展证据是否对齐；只有在证据充分的候选之间，才以消息数、命令数、重复、失败和耗时
作为效率比较。选择结果及全部候选评分保存在
`curation/best_trajectory_per_case.json`，当前状态为 `not_reviewed_fast_run`。

每条入选轨迹按有价值的可见推理节点拆成多个原生多轮 SFT 样本，而不是整条轨迹只生成
一条 decision。转换时：

- 原始可见 Agent 消息拆为 `<think>` 和阶段结论；它不冒充不可见的原始 CoT；
- 保留对下一步归因有因果价值的真实成功命令及结果摘要，每阶段最多 6 个；
- 已发生的历史消息、命令和结果作为下一节点上下文继承；
- 删除绕路、重复、失败和与最终归因无关的命令，不伪造原轨迹没有执行的调用；
- 4 个取证已经收敛但尚未输出最终答案的节点标为 `decision_ready`，用于监督模型及时停止
  继续取证；最终答案由紧随其后的 `decision` 节点单独监督。

节点类型共计 84 个 `planning`、199 个 `reasoning`、4 个 `decision_ready` 和 84 个
`decision`。loss 使用 ms-swift 的非二值权重：当前 `<think>` 为 0.4，当前阶段结论、
最终答案和当前实际工具调用为 1.0；历史 assistant/tool call 为 0，system、user、
tool response 只提供上下文、不计 loss。训练必须使用
`--loss_scale default --is_binary_loss_scale false`。

该快跑版仍是自动策展 draft。独立静态校验已通过来源哈希、精确命令来源、权重、时间
顺序、最终答案和整题隔离检查；目标 Qwen3.6 tokenizer 的 16K 长度检查必须在有模型与
ms-swift 的训练机上执行，快跑脚本会在训练前强制执行，超长样本会直接阻止训练。

## 目录

```text
2026-08-04/
├── README.md
├── raw/
│   └── qXXXX/run_YY/conversation_trajectory.json
├── curation/
│   ├── accepted_trajectory_selection.json
│   ├── best_trajectory_per_case.json
│   └── FILTER_REPORT.md
└── sft/
    ├── manifest.json
    ├── reasoning_trajectory_best1_manifest.json
    ├── qwen3_6_27b_reasoning_decision_train.jsonl
    ├── qwen3_6_27b_reasoning_decision_validation.jsonl
    ├── qwen3_6_27b_reasoning_trajectory_best1_train.jsonl
    └── qwen3_6_27b_reasoning_trajectory_best1_validation.jsonl
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
证据、694/120 数量、每类 100% 成功率候选数、5 类严格验证题、全局 STP 显式
回退题、query 隔离及输出哈希。

生成并校验每题最佳一条的原生多轮数据：

```powershell
python -B scripts/convert_0804_best_trajectory_reasoning_sft.py
python -B scripts/validate_0804_best_trajectory_reasoning_sft.py
```

在装有目标模型、`ms-swift==4.4.2` 和 GPU 的 Linux 训练机上启动 0804 快跑：

```bash
bash scripts/train_qwen36_0804_best1_quick.sh
```

脚本会重新生成并静态校验数据，使用目标模板逐条检查 16,384 token 上限，然后进行
1 epoch LoRA SFT；任一检查失败都不会进入训练。
