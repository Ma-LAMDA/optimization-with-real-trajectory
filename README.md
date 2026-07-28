# Qwen3.6-27B 推理、计划与决策 SFT 数据

本项目按日期保存网络故障分析轨迹及其 SFT 数据。`2026-07-27` 数据由
`q0014`、`q0017`、`q0018` 三段轨迹人工策展为多阶段样本；`2026-07-28`
数据由最新一次 14×10 Codex 完整运行转换而来，并按题号执行留一验证划分。
两批训练输出均不包含工具调用协议。训练目标是让模型学会：

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
│   └── simulation/
│       └── prompts, evaluation trajectories, configs and JSONL data
├── experiments/
│   └── 2026-07-27-ip_codex_train0629_14x10/
│       ├── inputs/
│       ├── scripts/
│       └── results/
└── scripts/
    ├── convert_codex_run_trajectories.py
    ├── convert_trajectories.py
    ├── train_qwen36_lora_smoke.sh
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

## 重新生成与校验

脚本只依赖 Python 标准库：

```powershell
python scripts/convert_trajectories.py
python scripts/validate_sft.py

python scripts/convert_codex_run_trajectories.py
python scripts/validate_codex_run_sft.py
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
- 0728 数据按题号留一，训练集与验证集题号交集为 0。

## ms-swift 训练示例

```bash
swift sft \
  --model Qwen/Qwen3.6-27B \
  --dataset data/2026-07-28/sft/qwen3_6_27b_reasoning_decision_train.jsonl \
  --val_dataset data/2026-07-28/sft/qwen3_6_27b_reasoning_decision_validation.jsonl \
  --split_dataset_ratio 0 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --output_dir output/qwen36-27b-reasoning-lora
```

批大小、梯度累积、LoRA 参数、分布式策略和 `max_length` 应按训练硬件及目标 tokenizer 的实际统计调整。Qwen3.6 的线性注意力训练还需要 `flash-linear-attention`；当前环境基线固定为 0.5.1。完整训练门槛、服务器资源假设、冒烟训练参数、逐分钟监控项和验收标准见 [`docs/TRAINING_PLAN.md`](docs/TRAINING_PLAN.md)。仓库提供的 `scripts/train_qwen36_lora_smoke.sh` 会预检该依赖，并只执行单卡一轮 LoRA 链路验证，不代表正式能力训练。

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

## 数据说明

- 0727 数据只有 3 条来源轨迹、12 条阶段样本，不单独划分验证集。
- 0728 数据包含 90 条训练样本和 10 条验证样本；按题号整组留一，禁止将同题重复运行拆到两个集合。
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

## 提交维护规则

每次创建并推送 GitHub 提交时，必须在同一个提交中同步更新本 README，记录该次变更对项目内容、数据、脚本或使用方式的影响。

### 更新记录

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
