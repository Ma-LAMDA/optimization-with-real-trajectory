# 2026-08-07 observation-bound evidence-gated SFT

0807 v7 从 0805 的 840 条严格成功轨迹独立重建，复用 0804 的 72/12 冻结题目划分；不修改
0805、0804 或 0731。v7 延续 q73–q86 的 inclusive-OR 与严格 VLAN/MST 闭环，并根据
`2026-08-07_1703_0807_sft_reaudit_v5.md` 修复 LLDP/LDP 家族边界、混合事实/计划绕过和
训练阶段步数后置条件。当前状态：

```text
rule_and_target_tokenizer_validated_release_candidate
```

规则校验、无共享生成语义的独立校验、目标 Qwen tokenizer 16K 和逐 token loss-mask 预检均已
通过。该状态表示数据可进入训练准备；它不等价于模型效果已经通过 Agent 评测。

## 来源与划分

- 不可变来源：`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/`。
- 归档：84 题 × 10 条 accepted = 840；incorrect 502；format_error 0。
- q41–q56 没有正确轨迹，不进入 0807。
- 基础设施失败和人为中断不记录、不统计、不训练。
- 训练 72 题；验证 12 题：q2、q12、q19、q20、q29、q38、q65、q71、q85、q86、q99、
  q100。
- 训练/验证按 `case_id` 整题隔离；该验证集不是 topology-heldout。

## q73–q86 标准答案修订

q73–q86 的 evaluator 接受三个精确集合：仅 `Core_SW_01`、仅 `Core_SW_02`、或两者同时出现；
14 题的源答案、140 个 raw 和 curation 均已同步为三个选项。现有 140 条成功轨迹全部仍为
单设备答案，因此题目划分、成功轨迹数和 accepted/incorrect 统计不变。

判分可接受范围与 SFT 规范目标分离：SFT 优先监督证据最强的单设备答案。只有两个设备分别
满足“源主机地址 → 源 VLAN → 同 VLAN 的 VRRP Master → 显式 MST VLAN-instance 映射 →
同实例 Alternate/Discarding”完整闭环时，才允许双设备 target。旧版 15 个角色错位 endpoint
只有 4 个满足该合同；v5 为 q73–q86 各保留 1 个，14/14 均完整闭环，旧版不安全终点为 0。

## v7 生成与训练规则

### 动作与路径

动作、轨迹质量和代表路径选择不读取 verified label。每个动作必须指向题目指定的单一快照；
snapshot、device、filename 三层分别检测 `* ? [`，任何路径 glob 都不获得正 loss。

动作不再按固定 6 条截断。生成器从原始 conclusion 提取下一动作意图，以贪心 set cover 保留
最小动作集合；每条记录分别归档源意图和最终监督意图的 full/partial/zero/unscoped 覆盖，以及
每个保留 action 的显式 binding。294 个有动作节点的源意图覆盖为 117/20/15/142；最终监督意图
为 191/0/0/103。原始多步计划只有在全部设备/协议意图均被当前动作覆盖时才保留，否则按实际
动作重建；两套验证器要求最终监督的 partial/zero 均为 0。

3,505 个原始可见 checkpoint 形成 3,016 个因果前缀 checkpoint 和 753 个精确路径簇；104 条
路径通过直接证据门控，84/84 题至少一条。

### 普通 reasoning 的原子证据合同

普通事实原句不再因设备名或协议词出现在命令路径就直接参与 loss。原句仅用于从更早结果中选择
相关证据，监督文本只包含 exact observation atoms；每个 atom 绑定 action ID 和非空、非表头
span。empty output、纯表头、帮助文本、截断标记和 `unselected lines omitted` 均不构成证据，
也不能支持“未配置/不存在”。当前保留 78 条 observation-bound 事实归纳和 51 条纯未来计划，
删除 697 个不受支持事实子句。含事实与计划的混合句先切分；事实必须绑定更早 observation，
只有不含既成事实断言的未来子句才可按 `procedural_plan` 监督。报告中的已知坏句均进入固定回归。

LLDP 是独立的邻接拓扑 family；`ldp`、`lsp`、`mpls` 使用字母数字 token 边界匹配，禁止用
`lldp` 内部的 `ldp` 子串满足 MPLS 意图。当前 10 个 LLDP 动作误标为 MPLS 的数量为 0。

### 排除错误候选

原始排除语句先拆为原子事实。每个进入监督的事实都必须绑定：

1. 更早的 action ID；
2. 非空、非帮助、非表头的原始 observation span；
3. span 中逐项出现的显式 anchor。

无法绑定的分句删除；监督文本只复述精确 span，并且只有 span 能映射到明确反证范围（CRC、
持续 down、带宽拥塞、MTU 或与已显示出接口冲突的绕行）时才生成排除节点；未覆盖候选保持
未决。最终 12 条通过：训练 11、验证 1。旧错误行
`q0004_path_05_success_08_step_03`、`q0008_path_04_success_06_step_03`、
`q0032_path_01_success_06_step_03` 均不再生成，并进入固定回归测试。

### evidence-gated endpoint

门控只接受题目快照中可见的直接事实：STP Disabled、BPDU filter、VRRP 非抢占、VRRP
Master 与 STP Alternate/Discarding 角色错位、IP 静态路由三设备环或 MPLS 静态 LSP
三设备标签环。IP/MPLS next-hop 所属设备必须由同一可见历史中的接口地址事实证明，不再使用
硬编码 PE 地址映射。

104 条路径的门控分布为：STP Disabled 14、BPDU filter 15、VRRP 非抢占 29、角色错位 14、
IP 环 16、MPLS 环 16。多设备 recovery 只使用同题、同快照多条成功轨迹的最小真实只读证据，
保留 donor 与哈希。

每条路径仍监督 evidence summary、停止判断和最终答案，但三段放在一个 `endpoint_bundle` 中；
同一长上下文只序列化和前向计算一次。停止判断只声称“当前证据支持候选并可进入最终决策”，
不声称已经证明所有其他故障不存在，也不在监督文本中暴露内部 gate ID。

## 数据规模与采样

| 集合 | planning | reasoning | source stop | elimination | endpoint bundle | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 训练语义池 | 89 | 155 | 0 | 11 | 91 | 346 |
| 验证 | 13 | 25 | 0 | 1 | 13 | 52 |

每个 bundle 内含 summary、stop、decision 三类正监督，因此逻辑 endpoint 组件仍各有 104 条。

正式五轮每轮：

- core：72 题 × 2 = 144 行；
- endpoint：72 题 × 1 bundle = 72 行；
- 合计：每题 3 行，每轮 216 行。

五轮覆盖全部 255 条训练 core 候选和 91 条训练 endpoint 路径。q89 只有一个高置信 core，
因此每轮对该来源做两次明确 replay；不为满足数量合成无证据节点。语义池和候选池仅用于审计与
排程，不应直接作为单轮训练输入。

## loss 与工具协议

| 来源 | loss scale |
| --- | ---: |
| 安全程序性原始 thinking | 1.00 |
| observation-bound reasoning reconstruction | 0.60 |
| claim-bound 排除 thinking / 结论 | 0.60 / 0.60 |
| 自动阶段重建结论 | 0.30 |
| evidence summary thinking / 结论 | 0.05 / 0.05 |
| 停止判断 thinking / 结论 | 0.10 / 0.10 |
| 固定桥接、final bridge thinking | 0 |
| 严格最终答案 | 1.00 |
| 当前工具调用 | 0.02 |
| system/user/历史消息/tool response | 0 |

必须使用 `--loss_scale default --is_binary_loss_scale false`。五轮启发式加权范围：thinking
42.07%–43.89%，工具 0.81%–0.90%，自动 summary+stop 22.89%–23.72%。

监督工具协议为 `exec_command` + `cmd`、Linux 只读命令和仓库相对
`saved_configs/...` 路径。语义池当前动作 369 个，配置类 10（2.71%）、LLDP 10；cross-snapshot、
snapshot glob、device glob、filename glob、训练目标 Windows 路径均为 0。

## 目标 tokenizer 放行

训练服务器环境：Qwen3.6-27B；Python 3.12.3；ms-swift 4.4.2；Transformers 5.12.1；
train template；非二值 `loss_scale=default`。

| 数据集 | 行数 | p99 token | 最大 token |
| --- | ---: | ---: | ---: |
| core pool | 255 | 3,642 | 4,044 |
| endpoint pool | 91 | 4,006 | 4,146 |
| validation | 52 | 3,383 | 3,578 |
| 全部 | 398 | 4,001 | 4,146 |

超过 16,384 的样本为 0；`labels` 与逐 token `loss_scale` 不一致为 0。模型 tokenizer、chat
template 和 config 的 SHA256 及各 loss scale token 数见
`sft/TARGET_TOKENIZER_PREFLIGHT.json`。

## 复现与验证

```powershell
python -B scripts/update_q73_q86_inclusive_or.py
python -B scripts/convert_accepted_only_100x10_to_sft.py `
  --output-root data/2026-08-07 --archive-only
python -B scripts/sync_0807_q73_q86_inclusive_or.py
python -B scripts/convert_0807_evidence_gated_reasoning_sft.py
python -B scripts/validate_0807_evidence_gated_reasoning_sft.py
python -B scripts/independent_validate_0807_sft.py
python -B scripts/audit_0807_evidence_gated_sft.py
```

服务器 tokenizer 复现：

```bash
/root/autodl-tmp/envs/qwen36-sft/bin/python \
  scripts/check_0807_target_tokenizer_preflight.py \
  --model /root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B \
  --max-length 16384 \
  --output data/2026-08-07/sft/TARGET_TOKENIZER_PREFLIGHT.json
```

训练准备和 dry-run：

```bash
RUN_MODE=prepare bash scripts/train_qwen36_0807_evidence_gated_5epoch.sh
RUN_MODE=dry-run bash scripts/train_qwen36_0807_evidence_gated_5epoch.sh
```

正式入口与 0805 一致使用 GPU 0、1 双进程 DDP、每卡 batch 1、梯度累积 4，故全局有效
batch 仍为 8；同时使用 constant scheduler、零 warmup、五档固定 LR 和完整 checkpoint
resume。每阶段固定 216 行，必须新增 27 个 optimizer step；五阶段 global-step
边界固定为 `0/27/54/81/108/135`，checkpoint 必须精确为 `27/54/81/108/135`。callback
在所有 DDP rank 强制 LR，但仅 rank 0 写审计；阶段结束后校验完整连续审计链。五轮 eval loss
只作诊断，不运行 Agent checkpoint selection；与 0805 一样固定使用 epoch 3 的
`epoch_03/checkpoint-81` 执行12道验证题×5次最终 Agent 验证。

完整输入哈希、输出哈希、算法、环境和强制变更流程见 `REPRODUCIBILITY.md`；本轮问题与修复证据
见 `AUDIT_REPORT.md`。
