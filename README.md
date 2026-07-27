# Qwen3.6-27B 轨迹 SFT 数据

本项目保存 `q0014`、`q0017`、`q0018` 三段原始 agent 轨迹，并将其转换为 Qwen3.6-27B 可用的监督微调数据。

## 目录

```text
.
├── data/
│   ├── raw/
│   │   ├── q0014/conversation_trajectory.json
│   │   ├── q0017/conversation_trajectory.json
│   │   └── q0018/conversation_trajectory.json
│   └── sft/
│       ├── manifest.json
│       ├── qwen3_6_27b_native_tool_sft.jsonl
│       ├── qwen3_6_27b_ms_swift_agent_sft.jsonl
│       └── qwen3_6_27b_final_answer_sft.jsonl
└── scripts/
    ├── convert_trajectories.py
    └── validate_sft.py
```

## 该用哪个文件

- `qwen3_6_27b_ms_swift_agent_sft.jsonl`：推荐用于 ms-swift。保留多轮思考、并行工具调用、工具返回和最终答案。
- `qwen3_6_27b_native_tool_sft.jsonl`：Qwen3.6 官方聊天模板的原生消息结构，适合 Transformers/TRL 或自定义训练管线。
- `qwen3_6_27b_final_answer_sft.jsonl`：只保留 `system + user + 最终 assistant`，适合不训练工具调用的普通对话 SFT。

原始文件逐字节保存在 `data/raw`，生成文件可随时重建。

## 字段映射

| 原轨迹 | Qwen 原生格式 | ms-swift 格式 |
| --- | --- | --- |
| `content[].type=text` | `content` | `content` |
| `content[].type=thinking` | `reasoning_content` | assistant 内容中的 `<think>...</think>` |
| `content[].type=tool_use` | `assistant.tool_calls[]` | `role=tool_call` |
| `content[].type=tool_result` | `role=tool` | `role=tool_response` |

相邻的 assistant 文本、思考和工具调用会合并为同一个生成回合；连续工具调用保留为并行调用。工具定义由三段轨迹中实际出现的参数自动推导，未凭空补写业务语义。

## 重新生成与校验

项目脚本只依赖 Python 标准库：

```powershell
python scripts/convert_trajectories.py
python scripts/validate_sft.py
```

也可指定其他输入和输出目录：

```powershell
python scripts/convert_trajectories.py --input-dir D:\path\to\raw --output-dir D:\path\to\sft
python scripts/validate_sft.py --sft-dir D:\path\to\sft
```

## ms-swift SFT 示例

先安装支持 Qwen3.6 的新版 ms-swift，再运行：

```bash
swift sft \
  --model Qwen/Qwen3.6-27B \
  --dataset data/sft/qwen3_6_27b_ms_swift_agent_sft.jsonl \
  --agent_template qwen3_5 \
  --train_type lora \
  --torch_dtype bfloat16 \
  --output_dir output/qwen36-27b-agent-lora
```

批大小、梯度累积、LoRA 参数、分布式策略和 `max_length` 应按训练硬件调整。三段完整轨迹都很长，正式训练前务必用目标模型 tokenizer 统计 token 长度；不要依赖框架默认长度，以免轨迹尾部被截断。

## 数据说明

- 当前只有 3 个样本，不建议再机械切分验证集。
- 工具返回不参与生成时，应在训练框架中保持默认的非 assistant loss mask。
- `final_answer` 版本会丢失工具使用能力，只适合作为兼容或对照数据。
- 数据可能包含内部系统提示、工具输出、网络地址或其他敏感信息；推送到远端 GitHub 前请先完成脱敏和授权检查。

## 提交维护规则

每次创建并推送 GitHub 提交时，必须在同一个提交中同步更新本 README，记录该次变更对项目内容、数据、脚本或使用方式的影响。

### 更新记录

- 2026-07-27：在 `data/simulation/` 中新增 `IP user prompt by text.txt`，保存 IP 故障分析仿真的用户提示词文本。
- 2026-07-27：建立 README 同步维护规则，并增加仓库级协作说明。

