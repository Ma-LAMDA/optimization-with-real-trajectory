# 2026-07-27 多阶段推理与决策 SFT 小样本实验

## 实验目的

本目录用于验证一条小规模数据构造路线：从真实网络故障排查轨迹中人工选择关键节点，
分别制作“下一步规划（planning）”“阶段推理（reasoning）”和“最终决策
（decision）”样本，观察这些不同阶段能否统一转换为 Qwen3.6-27B 可使用的
三轮对话 SFT 格式。

这是一组数据格式和策展方法的基线实验，不是完整训练集，也不包含工具调用模仿。
训练目标是让模型根据题目与已知证据给出简洁、可复核的思考及当前阶段输出。

## 数据来源与处理

- 原始数据为题 14、17、18 的 3 条网络故障排查对话轨迹。
- `curation/reasoning_decision_annotations.json` 人工标注轨迹中的关键消息、证据来源、
  推理摘要和期望回答。
- 转换时去除工具协议和环境接口细节，只保留 `system`、`user`、`assistant`
  三轮训练结构。
- 所有标注当前均为 `draft`，正式训练前仍需领域审核。

## 产出

| 项目 | 数量 |
| --- | ---: |
| 原始轨迹 | 3 |
| planning 样本 | 7 |
| reasoning 样本 | 2 |
| decision 样本 | 3 |
| SFT 样本合计 | 12 |

12 条样本全部放入训练集，没有单独的验证集。因此，这批数据适合检查格式、训练流程和
多阶段目标设计，不适合单独用于评估泛化能力。

## 目录结构

```text
2026-07-27/
├── README.md
├── raw/                       # 3 条原始对话轨迹
├── curation/
│   └── reasoning_decision_annotations.json
└── sft/
    ├── manifest.json
    └── qwen3_6_27b_reasoning_decision_sft.jsonl
```

## 重新生成与校验

在仓库根目录执行：

```powershell
python scripts/convert_trajectories.py
python scripts/validate_sft.py
```

样本数量、类型分布、输出哈希和标注文件哈希以 `sft/manifest.json` 为准。
