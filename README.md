# Qwen3.6-27B 推理、计划与决策 SFT 数据

本项目按日期保存网络故障分析轨迹及其 SFT 数据。`2026-07-27` 数据由
`q0014`、`q0017`、`q0018` 三段轨迹人工策展为多阶段样本；`2026-07-28`
数据由 14×10 Codex 完整运行转换而来；`2026-07-31` 数据从 100×10 实验的
819 条独立判题严格正确轨迹转换，并按题号执行留一验证划分。训练输出均不包含
工具调用协议。训练目标是让模型学会：

- 根据当前信息判断还缺少哪些事实；
- 说明下一步需要核验什么以及为什么核验；
- 根据新增证据形成阶段判断；
- 证据充分后给出最小故障根因集合。

## 目录

```text
.
├── docs/
│   └── TRAINING_PLAN.md
├── data/
│   ├── 2026-07-27/
│   │   ├── raw/
│   │   │   ├── q0014/conversation_trajectory.json
│   │   │   ├── q0017/conversation_trajectory.json
│   │   │   └── q0018/conversation_trajectory.json
│   │   ├── curation/
│   │   │   └── reasoning_decision_annotations.json
│   │   └── sft/
│   │       ├── manifest.json
│   │       └── qwen3_6_27b_reasoning_decision_sft.jsonl
│   ├── 2026-07-28/
│   │   ├── raw/
│   │   │   └── qXXXX/run_XX/conversation_trajectory.json
│   │   ├── curation/
│   │   │   └── trajectory_selection.json
│   │   └── sft/
│   │       ├── manifest.json
│   │       ├── qwen3_6_27b_reasoning_decision_train.jsonl
│   │       └── qwen3_6_27b_reasoning_decision_validation.jsonl
│   ├── 2026-07-31/
│   │   ├── raw/
│   │   ├── curation/
│   │   └── sft/
│   └── simulation/
│       └── prompts, evaluation trajectories, configs and JSONL data
├── experiments/
│   ├── 2026-07-27-ip_codex_train0629_14x10/
│   │   ├── inputs/
│   │   ├── scripts/
│   │   └── results/
│   ├── 2026-07-28-ip_codex_train0629_10x10/
│   ├── 2026-07-28-ip_codex_train0629_100x10/
│   ├── 2026-08-02-ip_codex_gpt56-sol_100x10/
│   ├── 2026-07-31-qwen36-27b-base-eval/
│   ├── 2026-07-31-qwen36-27b-agent-ab/
│   └── 2026-08-02-qwen36-27b-heldout6-agent-ab/
├── saved_configs_service/
└── scripts/
    ├── convert_codex_run_trajectories.py
    ├── convert_100x10_accepted_to_sft.py
    ├── convert_trajectories.py
    ├── evaluate_sft_validation.py
    ├── finalize_lora_workflow.py
    ├── run_agent_validation.sh
    ├── run_seetacloud_base_agent_eval.sh
    ├── run_seetacloud_agent_checkpoint_eval.sh
    ├── run_seetacloud_lora_workflow.sh
    ├── summarize_agent_validation.py
    ├── summarize_sft_training.py
    ├── train_qwen36_lora_early_stop.sh
    ├── train_qwen36_lora_smoke.sh
    ├── validate_100x10_sft.py
    ├── validate_codex_run_sft.py
    └── validate_sft.py
```

## 保留什么，删除什么

原始轨迹中的 assistant 内容按语义整理，而不是简单按消息角色保留或删除。

保留并净化：

- “先确认源端接入 VLAN 和实际转发路径”；
- “需要比较两条冗余上联的 STP 与 VLAN 配置”；
- “当前最强候选是 BPDU 过滤，但还要排除路由或策略异常”；
- 基于证据形成的最终根因决策。

删除：

- 具体工具名、函数调用结构和调用 ID；
- shell 命令、查询语句、URL、API 路径；
- “使用某工具读取某文件”一类执行细节；
- 工具失败、重试、待办列表和过程日志。

工具返回不会作为 assistant 训练目标。与判断相关的内容会被提炼成自然语言事实，放入后续样本的“当前已知证据”。这样模型学习的是信息需求和决策过程，而不是具体执行工具的偏好。

例如：

```text
原始思考：
I should use Grep to inspect the uplink configuration and compare it with another port.

整理后：
下一步需要比较两条核心上联的 VLAN 与 STP 配置，确认异常是否只存在于实际转发路径。
```

## 多阶段样本

`2026-07-27` 中一条原始轨迹可以生成多个训练样本，共生成 12 条：

| 类型 | 数量 | 训练目标 |
| --- | ---: | --- |
| `planning` | 7 | 根据当前证据决定下一步需要核验的事实 |
| `reasoning` | 2 | 形成阶段判断并指出仍需排除的候选 |
| `decision` | 3 | 输出最终最小根因集合 |

每条 JSONL 样本仍固定使用 `system + user + assistant`：

```json
{
  "id": "q0014_plan_02",
  "messages": [
    {
      "role": "system",
      "content": "你是一名网络故障分析专家……"
    },
    {
      "role": "user",
      "content": "原始题目……\n\n## 当前任务阶段\n\n当前证据不足以形成最终结论……\n\n## 当前已知证据\n\n1. 已获得路径相关设备配置。\n2. 一条核心上联配置了 BPDU 过滤。"
    },
    {
      "role": "assistant",
      "content": "<think>\n当前发现的异常位于实际路径，但还需要比较冗余上联并排除其他候选。\n</think>\n\n下一步：对比两条核心上联的 VLAN 与 STP 配置，并核对沿途路由和安全策略。"
    }
  ],
  "metadata": {
    "dataset_type": "reasoning_decision",
    "target_type": "planning",
    "source_id": "q0014",
    "source_message_index": 19,
    "evidence_message_indices": [13, 16],
    "review_status": "draft"
  }
}
```

`decision` 样本的 assistant 输出则以严格的题目格式结束：

```text
<think>
异常位于实际转发路径，并且是并行链路之间最明确的差异……
</think>

<result>
[
"AGG_SW_01;STP BPDU被过滤"
]
</result>
```

## 策展与来源追踪

`data/2026-07-27/curation/reasoning_decision_annotations.json` 为每个阶段样本记录：

- `source_id`：来源轨迹；
- `source_message_index`：被净化的原始 assistant 消息；
- `evidence_message_indices`：当前阶段已经获得的工具观察；
- `target_type`：`planning`、`reasoning` 或 `decision`；
- `reasoning`：去除执行细节后的思考；
- `response`：当前计划、阶段判断或最终结论；
- `review_status`：`draft` 或 `reviewed`。

转换器会保证证据消息出现在目标消息之前，并校验最终决策与原轨迹答案一致。当前 12 条标注均为 `draft`，正式训练前建议由网络领域专家审核。

## 2026-07-28 Codex 留一数据

`data/2026-07-28/` 来自
`experiments/2026-07-27-ip_codex_train0629_14x10/results/runs/fullaccess/`。
转换器将 140 条运行规范化到 `raw/`，并保留实验事件文件、最终答案、题目记录和
SHA-256 来源信息。

题 25、26、27、28 因 10 次运行的准确率未达到 100% 而整题排除。其余 10 道题
均为 10/10 正确，共形成 100 条 `decision` 样本：

| 集合 | 题号 | 样本数 |
| --- | --- | ---: |
| 训练集 | 13、14、17、18、87、88、91、92、93 | 90 |
| 验证集 | 94 | 10 |
| 排除 | 25、26、27、28 | 40 条原始轨迹，不进入 SFT |

划分策略为按 `case_id` 分组的 `leave_one_case_out`。验证题 94 不会出现在训练集，
因此不存在同题重复运行跨集合泄漏。仅当轨迹最终根因条目与原题标准答案完全一致、
最终事件与落盘答案一致，并且最终证据摘要不含工具操作细节时才允许进入 SFT。
源答案中的 Markdown 代码围栏会规范化为严格的 `<result>` 格式，原文仍保存在
`raw/`。全部样本当前标记为 `draft`，正式训练前仍需领域审核。

## 2026-07-31 严格正确轨迹 SFT 数据

`data/2026-07-31/` 来自
`experiments/2026-07-28-ip_codex_train0629_100x10/`。转换器扫描 1,313 个
attempt，过滤 473 个 rejected、11 个 interrupted 和 10 个 infrastructure failure，
保留 819 条 accepted 轨迹。所有保留轨迹均再次核对独立判题、参考答案精确集合匹配、
最终事件、文件哈希和前置证据清洁性。

| 集合 | 题号 | 样本数 |
| --- | --- | ---: |
| 训练集 | 6 类故障中除验证题外的 78 个题号 | 759 |
| 验证集 | 12、24、40、72、86、100 | 60 |
| 过滤 | 非 accepted attempt | 494 |

训练和验证按 `case_id` 分组，题号交集为 0；每种合并后的故障类型留出 1 个完整题号。
确定性选择规则为：优先留出该类型中严格正确轨迹数最多的题，数量并列时取题号最大
者。819 条样本均为 `decision` 类型并标记为 `draft`。过滤报告按答案 label 汇总
去重题目数与正确轨迹数，并忽略故障节点、按故障类型进一步合并统计；随后列出每类
验证题，按每题成功次数汇总题目数量，再逐题列出 1–100 的 attempt 数、成功数、
成功率、全部/成功 attempt 平均执行耗时、耗时覆盖、状态分布、SFT 入选数、划分和
实验终态；其中 11 条 interrupted attempt 缺失耗时，不以 0 计入平均值。完整统计见
[`data/2026-07-31/curation/FILTER_REPORT.md`](data/2026-07-31/curation/FILTER_REPORT.md)。

## 重新生成与校验

脚本只依赖 Python 标准库：

```powershell
python scripts/convert_trajectories.py
python scripts/validate_sft.py

python scripts/convert_codex_run_trajectories.py
python scripts/validate_codex_run_sft.py

python scripts/convert_100x10_accepted_to_sft.py
python scripts/validate_100x10_sft.py
```

也可以指定其他目录：

```powershell
python scripts/convert_trajectories.py `
  --input-dir D:\path\to\raw `
  --annotation-file D:\path\to\reasoning_decision_annotations.json `
  --output-dir D:\path\to\sft

python scripts/validate_sft.py --sft-dir D:\path\to\sft
```

校验器会检查：

- JSONL 和 `system/user/assistant` 三消息结构；
- planning、reasoning、decision 三类样本格式；
- `<think>` 与最终响应分段；
- 工具协议、具体工具名和执行操作未进入训练输出；
- API 文档已从原始问题中移除；
- 证据来源、样本数量、类型统计和文件哈希一致；
- 0727 数据全部进入训练集，验证集为 0；
- 0728 按题号留一；0731 按故障类型各留一题；两者训练与验证题号交集均为 0；
- 0731 数据只接收独立判题完全正确、最终事件一致且证据清洁的 accepted 轨迹；
- 0731 六种故障类型均恰有一个验证题，且验证题符合确定性选择规则；
- 0731 过滤报告的 100 行逐题统计和总计均与原始 attempt metadata、实验 state
  及最终 SFT 划分一致。

## ms-swift 训练示例

```bash
swift sft \
  --model Qwen/Qwen3.6-27B \
  --dataset data/2026-07-31/sft/qwen3_6_27b_reasoning_decision_train.jsonl \
  --val_dataset data/2026-07-31/sft/qwen3_6_27b_reasoning_decision_validation.jsonl \
  --split_dataset_ratio 0 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --output_dir output/qwen36-27b-reasoning-lora
```

批大小、梯度累积、LoRA 参数、分布式策略和 `max_length` 应按训练硬件及目标
tokenizer 的实际统计调整。Qwen3.6 的线性注意力训练还需要
`flash-linear-attention`；当前环境基线固定为 0.5.1。完整训练门槛、服务器资源
假设、早停参数、逐分钟监控项和验收标准见
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)。

仓库保留两个入口：

- `scripts/train_qwen36_lora_smoke.sh`：单卡单轮链路冒烟；
- `scripts/run_seetacloud_lora_workflow.sh`：SeetaCloud 端到端工作流。

端到端工作流会快进同步 `2026-07-31-sft`、重新生成并校验 759/60 的按故障类型
各留一题划分，
执行单卡 LoRA SFT，每 100 个优化步骤计算一次验证 loss，并在连续 3 次无改进时
早停。脚本始终按最低 `eval_loss` 选择 checkpoint，而不是直接使用最后一次保存；
之后启动单个 TP=2 vLLM 实例，复用原 `qwen36-27b-base-eval` 的 Codex CLI
Agent runner、完整调查 prompt、离线 `saved_configs` 工具和 60 分钟上限，以两个
worker、总并发 2 在 6 个留出题上各运行 5 次。最终能力结论来自完整 Agent 工具
循环的 30 次严格集合匹配，不再来自向模型直接提供既有证据的 60 条 SFT 补全请求。
SFT 验证集仍只用于训练期 `eval_loss`、早停和最佳 checkpoint 选择。

```bash
cd /root/autodl-tmp/optimization-with-real-trajectory
RUN_ID=0731-production \
  bash scripts/run_seetacloud_lora_workflow.sh
```

`VALIDATION_REPEATS` 表示每个留出题执行完整 Agent 调查的次数，默认值为 5。
所有运行严格使用两个 worker、总并发 2；原始事件流保存在 `agent_runs/`，并在
`validation_eval/validation_summary.json` 生成逐次与聚合结果：

```bash
RUN_ID=0731-2epoch-repeat5 \
NUM_TRAIN_EPOCHS=2 \
VALIDATION_REPEATS=5 \
  bash scripts/run_seetacloud_lora_workflow.sh
```

运行产物写入 `output/qwen36-27b-lora-0731-<RUN_ID>/`，其中
`training_summary.json` 记录最低验证 loss 与 checkpoint，
`validation_eval/validation_summary.json` 记录完整 Agent 的严格正确率、超时、
runner 失败、false positive/negative、工具循环、token 和耗时统计，
`workflow_summary.json` 汇总提交、数据、训练和评测溯源。`output/` 默认不提交。

Base 全量评测支持审计式组合为 `100 题 × 5 次 = 500 次`：主体 92 题的 460 次采用
单实例 TP=2、双 runner、总并发 2；历史留出的 8 题可在明确披露来源与并发差异的前提下
复用 37 次非超时结果，并用同一单实例双并发拓扑各补跑题 89、90、99 一次，分别替换旧的
`q89-r3`、`q90-r3`、`q99-r2` 超时槽位。`scripts/run_seetacloud_base_full500_followup.sh`
负责等待主体评测结束、顺序执行三次补跑并调用 `scripts/compose_base_full500_eval.py`；组合器
强制核验 100 个题号每题恰好 5 次、总计 500 次，并在报告中保留每条记录的来源，不能把
组合报告描述成全部 500 次均为双并发运行。

需要在长时间 Base 全量评测中优先验证 LoRA 时，使用
`scripts/run_seetacloud_lora_heldout_then_resume_base.sh`：先冻结 Base 控制器并等待当前两个
runner 自然结束，关闭 Base vLLM，再让推荐的 checkpoint-760 +100 在最新留出题
12、24、40、72、86、100 上各执行 5 次完整 Agent。LoRA 报告完成且服务退出后，脚本使用
原 `RUN_PREFIX` 恢复 Base；已终态样本会跳过，非终态样本才会继续。整个切换期间始终最多一个
TP=2 vLLM 实例和两个 Agent runner，训练期 validation loss 仍只用于早停与 checkpoint 选择。
Base 恢复后不会立刻回到普通题号顺序，而是先让同一组最新留出题
12、24、40、72、86、100 各完成 5 次完整 Agent；已由当前前缀完成的槽位直接跳过。其结果
单独写入 `<RUN_PREFIX>-priority-heldout6-report/`，从而与 LoRA 的 30 次结果形成同题、同次数、
同工具链和同双并发拓扑的端到端 A/B；这 30 个 Base 槽位齐全后才继续其余全量任务。
暂停监督器把已写入 `.runner_exit_code` 但尚未被冻结父进程回收的 zombie runner 视为已经结束，
随后终止并恢复被冻结的控制器以完成回收，避免在安全切换点无限等待。
完整 Agent 严格判分同时支持单一答案列表和数据集中的候选答案列表：单一答案仍要求 JSON 列表
逐项完全相等；候选形式要求预测与其中一个完整候选列表完全相等。false positive/negative 对
差异最小的候选计算，禁止把候选列表直接展平或把任意单个 label 命中误算为整题正确。

若训练已经完成而后处理被中断，可用同一个 `RUN_ID` 并设置
`REUSE_COMPLETED_TRAINING=1`，工作流会核对训练时与当前 manifest 哈希，并重新校验
最低 loss 与 checkpoint 后继续评测；缺少训练时 manifest 哈希或划分不一致时拒绝
复用。vLLM 服务固定设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，绕过
FlashInfer 0.6.13 在 Blackwell sm_120 上错误报告低于 sm75 的采样器 JIT
兼容问题；该设置只切换采样器实现，不改变单实例 TP=2 与双并发评测拓扑。
新运行会在输出目录保存训练源码提交；复用早期未保存该字段的训练时，须一次性设置
`TRAINING_GIT_COMMIT=<训练时提交>`，避免后处理提交被误记成训练提交。

要对一个既有 LoRA checkpoint 单独执行与历史 base-eval 同条件的端到端 A/B，可用：

```bash
CHECKPOINT=/path/to/checkpoint \
RUN_PREFIX=agent-ab-checkpoint-name \
  bash scripts/run_seetacloud_agent_checkpoint_eval.sh
```

该入口默认复用 base-eval 已归档的题 4、5、20、89 各 5 次 TP2 基线，仅运行
LoRA 侧 20 次并生成对比，避免重复消耗 base 推理时间。这里的四题已进入当前训练集，
因此该 A/B 只衡量与历史 Agent 基线的端到端变化；正式泛化验收仍以工作流从 manifest
读取的题 12、24、40、72、86、100 为准。

推荐的 checkpoint-760 +100 已按上述完整 Agent 口径实跑：LoRA 为 15/20
（75.00%），复用的 base 为 3/20（15.00%），提升 60.00 个百分点；两侧均无超时，
LoRA 平均封顶耗时由 26.66 分钟降至 13.95 分钟。分题结果、逐次预测和遥测归档在
[`experiments/2026-07-31-qwen36-27b-agent-ab/`](experiments/2026-07-31-qwen36-27b-agent-ab/)。
该结论只适用于已进入训练集的历史四题兼容 A/B，留出集结论仍须使用上述六题工作流。

要从零重跑历史 base-eval 的其余 92 题（题 1–88、91–94，每题 5 次），使用：

```bash
bash scripts/run_seetacloud_base_agent_eval.sh
```

该入口只启动一个 `Qwen3.6-27B-base` vLLM TP=2 实例，并复用
`scripts/run_agent_validation.sh` 固定的两个 Agent worker；总请求并发严格为 2，单次
上限 60 分钟。运行使用全新前缀，不复用或混入旧全量批次的 8-worker 轨迹；支持以
同一 `RUN_PREFIX` 重启后跳过已结束任务、继续非终态任务。旧 8 并发结果仅保留作历史
审计，不得作为这次双并发重跑的组成部分。

原始基座与多个 LoRA checkpoint 的同口径对比使用
`scripts/run_seetacloud_validation_sweep.sh`。脚本只启动一个 TP=2 vLLM 实例，
所有目标均固定两个 worker、总并发 2；基座默认重复 5 次，LoRA checkpoint
默认各验证 1 次，并生成统一的 `validation_sweep_summary.json`：

```bash
OUTPUT_ROOT="$PWD/output/qwen36-27b-step-sweep-0731" \
LORA_TARGETS="step500=/path/to/checkpoint-500 step600=/path/to/checkpoint-600" \
  bash scripts/run_seetacloud_validation_sweep.sh
```

需要在不恢复已经衰减结束的优化器和学习率调度器时继续训练，可向
`scripts/train_qwen36_lora_early_stop.sh` 传入 `RESUME_FROM_CHECKPOINT`、
`RESUME_ONLY_MODEL=true` 与正整数 `MAX_STEPS`。这种运行只加载 LoRA 权重并从
step 0 建立新的优化器和调度器，同时忽略旧 trainer 的数据游标，因此结果应标记为
“额外训练步数”，不得伪装成原训练曲线的无缝续接。

当前 759/60 分层划分已完成一轮 2-epoch 实跑：step 760 / epoch 2.0 取得最低
验证 loss `0.0045663742`，最终 checkpoint-760 即最佳 checkpoint。随后在同一
单实例 TP=2 服务内执行 5 次验证，每次严格匹配均为 49/60；汇总 300/300 请求
成功、245/300 严格匹配、300/300 格式正确且无泄漏。完整 loss 曲线、逐次结果、
按故障类型统计和 11 条稳定错误分析见
[`docs/2026-07-31_QWEN36_27B_LORA_SFT_759X60_2EPOCH_REPEAT5_RESULT.md`](docs/2026-07-31_QWEN36_27B_LORA_SFT_759X60_2EPOCH_REPEAT5_RESULT.md)。

同一 60 条验证集上的基座 5 次对比汇总为严格正确 22/300（7.33%）、格式正确
55/300（18.33%）；原训练 checkpoint-500/600/700/760 分别为
47/48/49/49。以 `1e-5` 学习率从 checkpoint-760 只加载 LoRA 权重并重建
优化器，额外训练 +100/+200 steps 后均达到 52/60（86.67%），但 +200 只继续
降低验证 loss，没有提高严格正确率，因此当前推荐 +100 checkpoint。完整逐次、
逐题与错误变化见
[`docs/2026-07-31_QWEN36_27B_BASE_AND_STEP_SWEEP_RESULT.md`](docs/2026-07-31_QWEN36_27B_BASE_AND_STEP_SWEEP_RESULT.md)。

历史 809/10 划分的 SeetaCloud 实跑在 step 600（epoch 1.4821）取得最低验证 loss
`0.0057987166`，随后按 patience 继续观察至 step 900 并早停，最终加载
checkpoint-600。该 checkpoint 在题 100 的 10 条验证样本上达到格式、严格集合
匹配和无泄漏均 10/10；该结果不代表当前 759/60 划分，旧 checkpoint 也不得作为
新划分的训练产物复用。完整参数、loss 曲线、运行路径及适用边界见
[`docs/2026-07-31_QWEN36_27B_LORA_SFT_RESULT.md`](docs/2026-07-31_QWEN36_27B_LORA_SFT_RESULT.md)。

## 推理生成约定

后续离线评估、交互推理和服务请求默认将输出上限设为 8,000 个新 token：

```bash
# ms-swift / Transformers
--max_new_tokens 8000
```

```json
{
  "max_tokens": 8000
}
```

8,000 token 是最大允许长度，不会强制模型生成满该长度；模型输出 EOS 时正常提前结束。采样温度等参数由具体任务单独指定。当前原始基座模型在单卡 Transformers 环境、`temperature=0.7`、5,000-token 上限下的单样本实测生成速度约为 21.2 token/s；长上下文下的实际速度可能下降。详细约束与测试口径见 [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)。

## Qwen3.6-27B eval 并发策略

凡调用本地 Qwen3.6-27B 基座或 LoRA adapter 服务进行的 eval，固定使用单实例
双并发：只启动 1 个 vLLM 实例，当前双卡部署采用 `tp2x1`；固定 2 个 eval
runner worker，总请求并发为 2。所有评测样本采用连续补位调度：任一 runner 结束后立即从
队列启动下一个样本，重试也必须复用已有槽位。禁止在 27B eval 中启动 8 个 worker、8 路
请求或自动扩容。该约束不适用于
Codex 轨迹生成及其他数据采集任务；数据采集策略由各实验独立配置。完整约束见
[`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)。

## Codex CLI 验证遥测

使用完整 user prompt、Codex CLI 和本地模型进行多次验证时，应同时保留原始事件流、服务日志和逐次遥测。必须区分 Codex turn、Responses API 调用、Agent 消息段与工具 loop，并记录 TTFT、TPOT、每轮 token、采样参数、上下文峰值、工具耗时、GPU 时间序列、KV cache、prefix cache、错误重试和严格 label 判定。缺失指标统一写为 `null`，不得以 0 或其他计数替代。

训练效果结论必须来自原始基座与 LoRA adapter 的同条件 A/B；两组各运行不少于 5 次，并报告逐次结果、准确率、false positive/negative、均值、中位数、P95、标准差和变异系数。字段定义、计算公式和推荐的 `telemetry.json` schema 见 [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md#9-codex-cli-多次验证遥测规范)。

2026-07-28 使用 epoch-10 LoRA、完整题 94 prompt 和 Codex CLI 串行验证 5 次：runner 与格式均为 5/5 成功，严格 label 匹配为 4/5；其中 Run 1 多报 Core_SW_02。本轮逐次最终输出、label、文件哈希、耗时、token、工具 loop、vLLM 吞吐、训练配置和适用边界见 [`docs/2026-07-28_Q94_EPOCH10_VALIDATION.md`](docs/2026-07-28_Q94_EPOCH10_VALIDATION.md)。由于未执行原始基座同条件对照，该结果不用于量化 LoRA 相对提升。

## 数据说明

- 0727 数据只有 3 条来源轨迹、12 条阶段样本，不单独划分验证集。
- 0728 数据包含 90 条训练样本和 10 条验证样本；按题号整组留一，禁止将同题重复运行拆到两个集合。
- 0731 数据包含 759 条训练样本和 60 条验证样本；默认作为当前 SFT 输入，按 6 种
  合并故障类型分别留出题 12、24、40、72、86、100。
- 题 25、26、27、28 的原始运行只用于审计，不进入训练或验证数据。
- 三条来源轨迹的最终标签相同。正式数据集必须补充不同设备、故障类型、正确配置和难负例，避免模型记忆固定答案。
- 同一轨迹拆出的阶段样本高度相关，训练时应按来源轨迹控制采样权重，避免长轨迹过度影响模型。
- 推送到远端 GitHub 前，应检查配置、地址和内部系统信息是否已经完成脱敏与授权。

### IP 故障分析仿真资料

`data/simulation/` 保存 IP 故障分析仿真的原始提示词、两组 GPT 评测轨迹、配置归档和训练 JSONL。资料包括：

- `ChatGPT system prompt.txt`、`Claude Code system prompt.txt`、`IP user prompt.txt` 和 `IP user prompt by text.txt`；
- `myf-ip评测0725-GPT_轨迹.zip`、`myf-ip评测GPT-0725-2_轨迹.zip` 和 `saved_configs.rar`；
- `train_data_0610.jsonl`（350 条记录）和 `train_0629.jsonl`（100 条记录）；后者与本次 Codex 实验归档的输入副本字节一致。

`Claude Code system prompt.txt` 的来源文件当前为空，仓库按来源状态原样保留。

`data/simulation/` 中的文件均视为不可变原始资料：只允许读取或复制到其他位置，不得编辑、覆盖、重命名、移动或删除。具体协作约束见 [`AGENTS.md`](AGENTS.md)。

### Codex 批量轨迹生成

`experiments/2026-07-27-ip_codex_train0629_14x10/` 集中保存本次实验的输入、生成脚本、完整运行，以及耗时与准确率统计。完整运行直接位于 `results/runs/`，包含题号 13、14、17、18、25、26、27、28、87、88、91、92、93、94 各 10 条成功轨迹，共 140 条。

```powershell
python experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py
```

实验仍从仓库根目录的共享 `saved_configs/` 快照读取离线配置。完整目录结构、轨迹文件说明、重试规则、恢复命令和耗时 CSV 生成方式见 `experiments/2026-07-27-ip_codex_train0629_14x10/README.md`。

`experiments/2026-07-28-ip_codex_train0629_10x10/` 保存使用本地 Codex CLI、`gpt-5.6-sol`
和 `saved_configs_service` 本地 HTTP API 重新生成的 10×10 实验。题号为
13、14、17、18、87、88、91、92、93、94，每题保留 10 条有效成功轨迹，共 100 条；
故障集合精确匹配标准答案后为 96/100 正确，准确率 96%。该目录包含完整事件流、最终回答、
运行/策略审计、逐题准确率 CSV、逐轨迹判分明细和审计工作簿；具体结构与复核命令见
`experiments/2026-07-28-ip_codex_train0629_10x10/README.md`。

`experiments/2026-07-28-ip_codex_train0629_100x10/` 保存覆盖 100 条输入、每题最多
10 条正确轨迹的完整实验。历史运行共保留 819 条 accepted 正确轨迹：79 题完成
10 条正确轨迹，21 题在连续 10 次错误后停止；全部 attempt、事件流、回答、判题结果
和运行审计均原样归档。该实验属于数据采集，历史运行及后续恢复均使用独立采集策略，
不受 Qwen3.6-27B eval 的单实例双并发约束。819 条 accepted 轨迹已经转换为
`data/2026-07-31/` 下的严格正确 SFT 数据。

`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/` 是基于
`IP user prompt by text.txt` 的新一轮本地 Codex CLI + `gpt-5.6-sol` 全量蒸馏。
该目录同时保留原提示词副本和优化提示词；优化版将配置根目录明确为
`saved_configs/`，说明 `<项目>/<节点>/<命令回显>.txt` 的三级目录与文件名转换规则，
并要求生成器直接列目录、搜索和读取本地文件；HTTP/API 读取被禁止，标准答案仍由安全
输入边界隔离。实验覆盖全部 100 题，每题只收录
独立严格判题正确的 10 条轨迹；连续错误达到 10 次或累计错误达到 20 次时停止该题，
基础设施失败不计入这两个阈值。运行状态、accepted 唯一映射和恢复方法见该实验的
[`README.md`](experiments/2026-08-02-ip_codex_gpt56-sol_100x10/README.md)。2026-08-03 账号切换
检查点曾归档 accepted 18 / 1,000；随后确认实际 user prompt 存在问题，旧 `results/`
已整体作废并删除。实际 prompt 已在 2026-08-03 完成本地文件读取版优化，之后从 q0001
attempt 1 全新启动，不恢复旧断点。实验已于 2026-08-04 完成，100 道题全部到达终态，
共保留 814 条 accepted 轨迹：79 题收齐 10 条正确轨迹，19 题因连续 10 次错误停止，
2 题因累计 20 次错误停止；最终完整性审计通过。新运行强制采用 accepted-only 保留策略：
**失败或中断结果一律不保留**；错误、格式错误、基础设施失败、超时和中断只保留必要的
状态计数，不归档、不提交、不长期保留其事件流、回答、日志或 attempt 目录。重置状态、
固定的 Standard 速度/初始及最大并发 10 配置及启动清单见
[`HANDOFF.md`](experiments/2026-08-02-ip_codex_gpt56-sol_100x10/HANDOFF.md)。

三个 Codex 轨迹实验已采用统一的紧凑归档：`prompt.txt` 和
`source_record.json` 按“实验 + 题号”各保留一份；100×10 实验只保留
`events.jsonl` 作为 Codex 原始标准输出流，并将共享 hooks 配置集中到
`config/hooks.json`。迁移删除 7,412 个重复文件、新建 249 个规范文件，净减少
7,163 个文件项和 971,494,909 字节（926.49 MiB），不修改事件、答案或判题证据。
逐实验统计见
[`experiments/ARCHIVE_COMPACTION_REPORT.json`](experiments/ARCHIVE_COMPACTION_REPORT.json)；
可用 `python scripts/compact_experiment_archives.py` 只读复核。

2026-07-30 至 2026-07-31 的 Qwen3.6-27B 基座部署 A/B 和全量评测已作为独立实验
归档到 [`experiments/2026-07-31-qwen36-27b-base-eval/`](experiments/2026-07-31-qwen36-27b-base-eval/)。
目录将部署对比与全量结果分开保存，并提供总体、逐题和逐次明细。

checkpoint-760 +100 与历史 base 的完整 Agent A/B 已归档到
[`experiments/2026-07-31-qwen36-27b-agent-ab/`](experiments/2026-07-31-qwen36-27b-agent-ab/)。
该实验复用 base 的 20 次 TP2 结果，只实跑 LoRA 侧；所有运行固定单实例 TP2、双并发
和 60 分钟上限。

checkpoint-760 +100 与 Base 在最新留出题 12、24、40、72、86、100 上的同条件完整 Agent
A/B 已归档到 [`experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/`](experiments/2026-08-02-qwen36-27b-heldout6-agent-ab/)：
LoRA 严格正确 12/30（40.00%），Base 为 7/30（23.33%），提升 16.67 个百分点；两侧分别有
4/3 次超时，均无非超时 runner failure。LoRA 平均/中位封顶耗时为 24.21/13.74 分钟，Base
为 32.37/26.81 分钟。两侧题目、prompt、工具链、次数、3600 秒上限与单实例双并发拓扑一致，
该结果作为当前最新留出划分的正式端到端泛化 A/B 结论。

## 提交维护规则

每次创建并推送 GitHub 提交时，必须在同一个提交中同步更新本 README，记录该次变更对项目内容、数据、脚本或使用方式的影响。

### 更新记录

- 2026-08-03：清理 0802 实验中未被引用的 smoke 输入、Python 字节码、空目录和
  约 337 MiB 的可重建 Codex CLI 副本，并合并 `.gitignore` 中已被 `/runtime/` 覆盖的
  重复规则；正式运行会自动重建所需 runtime 和输入索引。
- 2026-08-03：进一步简化 0802 实验 prompt，删除对其他读取方式和只读快照属性的
  重复强调，保留 `saved_configs/` 路径、三级目录解析、文件名转换和必要调查步骤；
  运行侧的文件访问边界保持不变。
- 2026-08-03：重写 0802 GPT-5.6-Sol 100×10 实验 prompt，将配置访问从本地 API
  改为直接只读 `saved_configs/` 文件，补充项目、节点、命令回显文件的目录解析规则，
  并同步切换输入边界、运行 hook、最终审计和交接文档；尚未启动新一轮采集。
- 2026-08-01：增加 27B base 全量 Agent eval 的单实例 TP2、双并发重跑入口；重新
  覆盖历史相同的 92 题×5 次范围，明确与旧 8-worker 轨迹隔离，并保留断点恢复能力。
- 2026-08-01：完成 checkpoint-760 +100 与历史 base-eval 的同条件完整 Agent A/B；
  LoRA 严格正确率 15/20（75%），较复用 base 的 3/20（15%）提升 60 个百分点，
  平均耗时由 26.66 分钟降至 13.95 分钟，20 次均无超时；归档逐次结果并明确该四题
  已进入训练集，正式泛化验收仍使用当前六题留出集。
- 2026-07-31：将训练工作流的最终验证改为复用 base-eval 的完整 Codex Agent
  runner、调查 prompt、离线工具和严格判分；最新 6 个留出题默认各跑 5 次，固定
  单实例 TP2、双并发、单次 60 分钟，并增加复用历史 base 20 次结果的 checkpoint
  端到端 A/B 入口。训练期 SFT validation loss 仍仅用于早停和 checkpoint 选择。
- 2026-07-31：在当前 60 条验证集上完成原始 27B 基座的 5 次双并发验证，
  扫描原训练 step500/600/700/760，并从 checkpoint-760 独立续训
  +100/+200 steps；严格正确率从基座均值 7.33% 提升至 86.67%，推荐 +100。
- 2026-07-31：训练脚本支持只加载既有 LoRA 权重并以独立优化器执行指定
  `MAX_STEPS` 的额外训练，用于在原调度器已经衰减结束后复现实验性 step 扩展。
- 2026-07-31：增加原始 27B 基座与多个 LoRA checkpoint 的统一验证扫描脚本；
  同一单实例 TP=2 服务中，基座默认按双并发重复 5 次，checkpoint 默认各验证
  1 次，并汇总严格正确率及相对基座变化。
- 2026-07-31：按最新 759/60 分层划分完成 2-epoch LoRA SFT，最低验证 loss
  位于 checkpoint-760；在同一单实例 TP=2 服务中完成 5 次双并发验证，每次严格
  匹配 49/60，并归档按题号、故障类型和稳定错误模式的分析。
- 2026-07-31：压缩三个 Codex 轨迹实验归档，删除与 `events.jsonl` 完全相同的
  1,313 份 `stdout.log`，并按题号集中 prompt/source record、集中共享 hooks；
  同步更新 metadata、转换器、校验器和后续 runner，净释放 926.49 MiB。
- 2026-07-31：将 0731 的 819 条严格正确轨迹重新按故障类型分层并以题号整组划分，
  每种故障类型确定性留出 1 题，生成 759 条训练样本和 60 条验证样本；同步固化
  六类覆盖、题号隔离、选择规则和报告校验。
- 2026-07-31：完成 Qwen3.6-27B LoRA SFT 实跑，选定最低验证 loss 的
  checkpoint-600，并在单实例 TP=2、双并发验证中取得严格匹配 10/10；固化实际
  参数、结果、远端路径和训练/后处理提交分离的溯源规则。
- 2026-07-31：固化 SeetaCloud LoRA SFT 端到端工作流，增加按 `eval_loss`
  早停与最佳 checkpoint 校验，并以单实例双并发在固定验证集上执行格式、严格集合
  匹配及泄漏评测；支持安全复用已经完成的训练状态继续后处理。
- 2026-07-31：从 100×10 实验的 1,313 个 attempt 中过滤 819 条独立判题完全正确轨迹，生成 809 条训练和 10 条题 100 验证 SFT 数据，并增加可复现转换与独立校验脚本。
- 2026-07-31：合并 `taowen` 的 `saved_configs_service`、10×10 与 100×10 实验，保留完整轨迹和审计产物，并保持数据采集策略独立配置。
- 2026-07-31：将 Qwen3.6-27B eval 策略固定为单个 vLLM 实例、2 个 eval runner worker、总请求并发 2；该约束不适用于轨迹生成等数据采集任务。
- 2026-07-31：建立独立的 Qwen3.6-27B 基座评测实验目录，分开归档部署 A/B 与终止时的 381 个全量已结束样本，并提供 JSON、CSV、Markdown 统计。
- 2026-07-28：将题 94 epoch-10 LoRA 的 5 次实测结果同步到 Training Plan 和实验 README，补充运行命令、严格 4/5 判定、耗时/token/工具 loop、外部产物路径及基座 A/B 缺口。
- 2026-07-28：归档 Qwen3.6-27B epoch-10 LoRA 在题 94 上的 5 次 Codex CLI 验证报告，记录原始 label/输出、严格 4/5 结果、耗时、token、工具 loop、vLLM 指标、哈希与评测局限。
- 2026-07-28：增加 Codex CLI 多次验证遥测规范，统一 turn、API 调用、Agent 消息和工具 loop 口径，并规定 TTFT、TPOT、token、缓存、GPU、质量判定及基座/LoRA A/B 的记录要求。
- 2026-07-28：新增 `experiments/2026-07-28-ip_codex_train0629_10x10/`，归档通过本地 API 仿真环境重新生成的 100 条有效 Codex 轨迹、运行审计及 96% 准确率统计。
- 2026-07-28：将实验运行压缩为 `results/runs/fullaccess/q<题号>_r<轮次>/attempt_<序号>/`，合并重复的 case/run 层级，同时保留额度重试所需的 attempt 记录。
- 2026-07-28：将实验目录按“日期-实验名”合并命名为 `experiments/2026-07-27-ip_codex_train0629_14x10/`，移除 `results/runs/` 下的日期层，并同步适配生成、统计和 SFT 转换脚本。
- 2026-07-28：将最新 140 条 Codex 运行规范化到 `data/2026-07-28/`；排除准确率未达 100% 的题 25、26、27、28，并按题号留出题 94，生成 90 条训练和 10 条验证样本。
- 2026-07-28：将本次 Codex 实验使用的 `train_0629.jsonl` 原样复制到 `data/simulation/`，并增加仿真原始资料只读、只允许复制的保护规则。
- 2026-07-28：为 `raw`、`curation` 和 `sft` 增加统一的 `data/2026-07-27/` 日期层；`data/simulation/` 作为原始仿真资料保持原位不变。
- 2026-07-28：将 Train 0629 Codex 轨迹实验的输入、脚本、140 条完整轨迹、runner 日志和统计 CSV 统一整理到 `experiments/2026-07-27-ip_codex_train0629_14x10/`。
- 2026-07-27：新增指定 IP 题目的 Codex 批量执行脚本，每题执行 10 个成功轮次，共生成 140 条完整 JSONL 轨迹；额度不足时保留失败尝试，并每隔 30 分钟无限重试同一槽位。
- 2026-07-27：新增根目录 `saved_configs` 离线组网配置快照和 `IP user prompt with saved configs skills.txt`，用于按项目、节点与命令文件查询故障证据。
- 2026-07-27：向 `data/simulation/` 补充 IP 故障分析提示词、GPT 评测轨迹、配置归档及两份训练 JSONL，并记录文件清单与可解析记录数。
- 2026-07-27：将后续推理默认输出上限统一为 8,000 个新 token，并记录 ms-swift、vLLM 参数写法及原始基座模型的单样本速度基线。
- 2026-07-27：根据首次冒烟执行结果补充 `flash-linear-attention==0.5.1` 环境要求，并在训练脚本中增加启动前依赖检查。
- 2026-07-27：新增 Qwen3.6-27B LoRA 训练方案和单卡冒烟训练脚本，补充数据准入、逐分钟 loss 监控与验收要求，并将 ms-swift 4.x 参数更正为 `--tuner_type lora`。
- 2026-07-27：将一轨迹一决策样本扩展为多阶段样本；保留抽象的下一步核验计划，删除具体工具与执行方式，共整理 7 条 planning、2 条 reasoning 和 3 条 decision 样本。
- 2026-07-27：将生成目标升级为单一 `reasoning_decision` SFT；新增显式策展证据和推理标注，移除工具调用训练格式。
- 2026-07-27：建立 README 同步维护规则，并增加仓库级协作说明。
