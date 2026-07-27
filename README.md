# Qwen3.6-27B 推理、计划与决策 SFT 数据

本项目保存 `q0014`、`q0017`、`q0018` 三段原始网络故障分析轨迹，并将它们整理为不包含工具调用协议的多阶段 SFT 数据。训练目标是让模型学会：

- 根据当前信息判断还缺少哪些事实；
- 说明下一步需要核验什么以及为什么核验；
- 根据新增证据形成阶段判断；
- 证据充分后给出最小故障根因集合。

## 目录

```text
.
├── data/
│   ├── raw/
│   │   ├── q0014/conversation_trajectory.json
│   │   ├── q0017/conversation_trajectory.json
│   │   └── q0018/conversation_trajectory.json
│   ├── curation/
│   │   └── reasoning_decision_annotations.json
│   └── sft/
│       ├── manifest.json
│       └── qwen3_6_27b_reasoning_decision_sft.jsonl
└── scripts/
    ├── convert_trajectories.py
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

一条原始轨迹可以生成多个训练样本。当前共生成 12 条：

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

`data/curation/reasoning_decision_annotations.json` 为每个阶段样本记录：

- `source_id`：来源轨迹；
- `source_message_index`：被净化的原始 assistant 消息；
- `evidence_message_indices`：当前阶段已经获得的工具观察；
- `target_type`：`planning`、`reasoning` 或 `decision`；
- `reasoning`：去除执行细节后的思考；
- `response`：当前计划、阶段判断或最终结论；
- `review_status`：`draft` 或 `reviewed`。

转换器会保证证据消息出现在目标消息之前，并校验最终决策与原轨迹答案一致。当前 12 条标注均为 `draft`，正式训练前建议由网络领域专家审核。

## 重新生成与校验

脚本只依赖 Python 标准库：

```powershell
python scripts/convert_trajectories.py
python scripts/validate_sft.py
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
- 当前数据全部进入训练集，验证集为 0。

## ms-swift 训练示例

```bash
swift sft \
  --model Qwen/Qwen3.6-27B \
  --dataset data/sft/qwen3_6_27b_reasoning_decision_sft.jsonl \
  --train_type lora \
  --torch_dtype bfloat16 \
  --output_dir output/qwen36-27b-reasoning-lora
```

批大小、梯度累积、LoRA 参数、分布式策略和 `max_length` 应按训练硬件及目标 tokenizer 的实际统计调整。

## 数据说明

- 当前只有 3 条来源轨迹、12 条阶段样本，暂不划分验证集；待轨迹增加后应按 `source_id` 分组划分，避免同一轨迹进入训练集和验证集。
- 三条来源轨迹的最终标签相同。正式数据集必须补充不同设备、故障类型、正确配置和难负例，避免模型记忆固定答案。
- 同一轨迹拆出的阶段样本高度相关，训练时应按来源轨迹控制采样权重，避免长轨迹过度影响模型。
- 推送到远端 GitHub 前，应检查配置、地址和内部系统信息是否已经完成脱敏与授权。

## 提交维护规则

每次创建并推送 GitHub 提交时，必须在同一个提交中同步更新本 README，记录该次变更对项目内容、数据、脚本或使用方式的影响。

### 更新记录

- 2026-07-27：将一轨迹一决策样本扩展为多阶段样本；保留抽象的下一步核验计划，删除具体工具与执行方式，共整理 7 条 planning、2 条 reasoning 和 3 条 decision 样本。
- 2026-07-27：将生成目标升级为单一 `reasoning_decision` SFT；新增显式策展证据和推理标注，移除工具调用训练格式。
- 2026-07-27：建立 README 同步维护规则，并增加仓库级协作说明。
