# 2026-08-07 0807 SFT v7 修复与放行报告

复审日期：2026-08-07  
输入意见：`trajectory-analysis/2026-08-07_q0073-q0086_inclusive_or_impact.md`、
`trajectory-analysis/2026-08-07_1703_0807_sft_reaudit_v5.md`  
范围：`data/2026-08-07/`、0807 生成/验证/tokenizer/训练脚本  
结论：

```text
rule_and_target_tokenizer_validated_release_candidate
```

q73–q86 inclusive-OR 标准答案、0807 raw/curation 和 SFT 终点已经同步。官方验证器与不导入
0807 生成器的独立验证器同时通过 v7 规则与训练合同检查；目标 Qwen3.6 tokenizer 16K 与逐
token loss mask 已对 398 条语义样本重跑并通过。下文 v6/v5/v4/v3 内容作为历史审计保留；
当前放行结论以新增的“v7 对齐 0805 正式训练方式”一节为准。

## v7 对齐 0805 正式训练方式

根据用户复核，v6 的核心 LoRA 参数虽与 0805 相同，但硬件拓扑和 checkpoint 选择不同。v7
只修正训练执行合同，不改变来源、划分、语义节点、loss、采样表或 tokenizer 结果：

- 单卡 batch 1、梯度累积 8 改为 GPU 0、1 双进程 DDP、每卡 batch 1、梯度累积 4；全局有效
  batch 保持 8；
- 216 行每阶段仍为 27 optimizer steps，五阶段边界仍为 `0/27/54/81/108/135`；
- callback 在两个 rank 都强制并核验固定 LR，只有 rank 0 写审计；
- 删除6题×2次 Agent checkpoint selection，五轮 eval loss 只作诊断；
- 固定使用 epoch 3 / `epoch_03/checkpoint-81` 做全部12题×5次最终 Agent 验证，与 0805 的
  fixed-epoch-3 策略一致。

官方与独立 validator 同时检查 DDP/world-size/CUDA/梯度累积/有效 batch、rank-0 审计、固定
epoch 3 和禁止 Agent 选点。manifest schema 升至 v7，audit metrics 升至 v6。

## v6 对 17:03 独立复审的响应

### P0：LLDP/LDP family 已修复

- 新增独立 `lldp` 邻接拓扑 family；`ldp/lsp/mpls` 改为字母数字 token 边界匹配；
- `display_lldp_*` 不能再因 `ldp in lldp` 被标为 MPLS；当前 LLDP 动作 10，误标 0；
- 官方与独立 validator 均从 command/filename 自行推导 family，再与 metadata 和正 loss 文本
  对比；固定 LLDP 负例及报告列出的受影响行已纳入回归。

### P0：mixed procedural 绕过已修复

- 原始 thinking 先按子句切分；既成事实必须绑定更早 action ID 与 exact observation span，
  无法绑定即删除；
- 只有纯未来动作可标为 `procedural_plan`；裸 `再/随后/然后` 不再足以放行事实描述，只有后接
  明确检查动词时才视为计划；
- q0023 step05、q0066 step02 和审计中列出的高风险固定文本均不再进入正 loss；当前已知
  unsafe mixed procedural rows 为 0。

### P1：最终动作意图与训练阶段合同已收紧

- 原始多步 conclusion 必须由当前动作完整覆盖全部设备/协议意图，否则重建为当前动作精确计划；
  最终监督 intent coverage 为 `full=191, unscoped=103, partial=0, zero=0`；
- 每阶段固定 216 行、梯度累积 8、27 optimizer steps；global-step 边界固定
  `0/27/54/81/108/135`，checkpoint 固定 `27/54/81/108/135`；训练后逐阶段核对完整连续
  `step_begin` 序列、开始/结束 step 与 checkpoint 后缀；
- 未执行高成本真实 GPU resume smoke test；该项仍是正式长训练前的运行时验收，而不是把
  dry-run 当作已验证 resume。

### 排除节点回归防护

逐子句清洗初次实现曾使排除节点从 12 条降到 1 条：排除标记与冒号后的反证被拆开，绑定器只
看到前半句。v6 改为普通 reasoning 使用切分子句，但排除候选回看原始完整句后再逐事实绑定。
最终恢复 12 条 elimination、13 个绑定原子事实；旧 q0004/q0008/q0032 强排除仍为 0。

### v6 最终结果

| 指标 | v5 | v6 |
| --- | ---: | ---: |
| retained paths | 103 | 104 |
| train / validation semantic | 344 / 53 | 346 / 52 |
| core / endpoint pool | 254 / 90 | 255 / 91 |
| 动作 / config / LLDP | 372 / 11 / 误标13 | 369 / 10 / 10（误标0） |
| observation-bound / procedural | 78 / 14 | 78 / 51 |
| supervised partial / zero | 14 / 0 | 0 / 0 |
| elimination | 12 | 12 |
| 每阶段训练行 / optimizer step | 216 / 未强校验 | 216 / 27 |
| tokenizer rows / max | 397 / 4,867 | 398 / 4,146 |

五轮自动 summary+stop 启发式最大权重占比为 23.72%，低于 25% gate；工具为
0.81%–0.90%。真实 tokenizer 检查 p99 4,001、max 4,146/16,384、overlong 0、loss-mask
failure 0。manifest 状态为 `rule_and_target_tokenizer_validated_release_candidate`。

## v5 inclusive-OR 与严格 VLAN/实例闭环

### 标准答案与来源同步

- q73–q86 每题恰好三个 evaluator 选项：仅 A、仅 B、A+B；不把 `[A,B]` 错写成唯一 AND；
- `data/simulation/train_0629.jsonl`、实验输入副本、140 个 0807 raw 和 curation 已同步；
- 140 条现有 accepted 轨迹全部仍为 singleton，新增命中 0、失效 0；840 条来源、84 题、
  72/12 划分均未变化；
- SFT 优先 evidence-strongest singleton；双设备 target 必须为两个设备分别建立完整闭环。

### 旧终点问题与修复

旧 gate 只要求答案设备上出现任意 VRRP Master 与任意 Alternate/Discarding，不能证明它们属于
题目源流量。逐题重审发现旧 15 个 q73–q86 endpoint 中只有 q75–q78 的 4 个完整匹配；其余
11 个使用了错误 VLAN、MST instance 0 或与源 VLAN 无关的实例。最严重的 q86 用
`Core_SW_01 / Vlanif30 / instance 2` 支撑 VLAN120 的题目。

v5 强制闭环：源主机 `/24` 地址 → 源 VLAN → 同 VLAN Vlanif 的 VRRP Master → 同设备显式
MST VLAN-instance mapping → 同实例 Alternate/Discarding。当前 q73–q86 各 1 个 endpoint，
14/14 完整闭环；q86 改为 `Core_SW_02 / Vlanif120 / instance 3`。独立 validator 内置旧 q86
错 VLAN 和“正确 VLAN 但错误 instance 0”两个负例，均必须被拒绝。

### 数量影响

| 范围 | v4 | v5 | 变化 |
| --- | ---: | ---: | ---: |
| q73–q86 semantic rows | 48 | 46 | -2 |
| q73–q86 endpoints | 15 | 14 | -1 |
| retained paths | 104 | 103 | -1 |
| train / validation semantic rows | 346 / 53 | 344 / 53 | -2 / 0 |
| train core / endpoint pool | 255 / 91 | 254 / 90 | -1 / -1 |
| 每轮训练 | 216 | 216 | 0 |

减少的两行来自 q82 的一个未通过新闭环的冗余路径及其关联 reasoning，不是删除题目或成功
轨迹。训练仍为每题每轮 2 core + 1 endpoint bundle，query 等权合同不变。

### 最终静态与 tokenizer 结果

- 正式 validator：passed；独立 validator：passed；
- 角色错位严格闭环 14/14；旧任意 VLAN/实例回归命中 0；
- 语义池动作 372，配置类 11（2.96%），跨快照和三层 glob 均为 0；
- tokenizer：397 行，p99 4,001，最大 4,867/16,384，overlong 0，loss-mask failure 0。

## v4 复审响应

### 普通 reasoning P0

- 事实原句不再直接参与 loss，只在 metadata 中保留用于审计；
- 实际 thinking 改写为此前工具结果中的 exact observation atoms，每个 atom 绑定 action ID 与
  非空、非表头 span；
- 命令路径中的设备名不能为事实背书；empty output、纯表头、帮助、截断提示和
  `unselected lines omitted` 均不能成为正证据；
- 独立 validator 不导入 converter，逐条复核普通 reasoning bindings，并固定检查报告列出的
  21 条越证据 row ID；旧关键词 grounding 类型必须为 0。

当前普通 reasoning 为 78 条 observation-bound 事实归纳、15 条安全程序性原句，删除 220 条
不受支持句。两套 validator 均确认原始越证据句不在正 loss target 中。

### 错误候选排除

排除样本继续从原始可见推理句寻找候选，但最终只监督 earlier exact span，并且 span 必须能映射
到明确反证范围：CRC 物理错误、当前持续 down、接口拥塞、MTU 假设或与已显示出接口冲突的
转发绕行。泛化的“与回显矛盾的子候选”不生成。最终保留 12 条（训练 11、验证 1）；q0004、
q0008、q0032 的三个旧错误 row ID 均为 0。

### 动作覆盖

不再用 `claim_coverage_after == claim_coverage_before` 单独宣称语义安全。311 个有动作节点分别
公布：

| 覆盖范围 | full | partial | zero | unscoped |
| --- | ---: | ---: | ---: | ---: |
| 原始 conclusion 的下一动作意图 | 120 | 18 | 15 | 142 |
| 最终参与监督的下一动作意图 | 182 | 13 | 0 | 100 |

15 个源 0→0 被如实记录为 zero；它们的监督 conclusion 已按实际保留动作重建，因此最终监督层
没有 zero coverage。metadata 同时归档 selected action IDs 与 claim units。

### v4 结构快照

| 指标 | v3 | v4 |
| --- | ---: | ---: |
| retained paths | 110 | 104 |
| train / validation semantic rows | 369 / 63 | 346 / 53 |
| core / endpoint pool | 274 / 95 | 255 / 91 |
| elimination | 8 | 12 |
| current actions | 514 | 372 |
| config actions | 20（3.89%） | 10（2.69%） |
| 每轮训练 | 216 | 216 |

104 条 endpoint 仍覆盖 84/84 题，分布为 STP Disabled 14、BPDU filter 15、VRRP 非抢占 28、
VRRP/STP 角色错位 15、IP cycle 16、MPLS cycle 16。每轮保持 72 ×（2 core + 1 bundle）；
q89 仅一个高置信 core，使用显式 replay，不合成无证据节点。目标 tokenizer 实测 p99 4,001、
最大 4,867/16,384、overlong 0、loss-mask failure 0。

## 1. 最终快照

| 指标 | v2 | v3 |
| --- | ---: | ---: |
| 严格成功来源 | 840 | 840 |
| 题目划分 | 72/12 | 72/12 |
| raw visible checkpoints | 3,505 | 3,505 |
| causal prefix checkpoints | 3,098 | 3,086 |
| path clusters | 733 | 760 |
| retained paths | 173 | 110 |
| train semantic rows | 904 | 369 |
| validation rows | 145 | 63 |
| core pool | 457 | 274 |
| endpoint pool | 447（三节点） | 95（单行 bundle） |
| claim-bound elimination | 51（规则假阳性） | 8 |
| 每轮训练行 | 360 | 216 |
| 每题每轮行 | 5 | 3 |
| 当前动作 | 3,016 | 514 |
| config 动作 | 233（7.73%） | 20（3.89%） |
| snapshot/device/filename glob | 漏报为 0 | 0/0/0，独立扫描一致 |
| 目标 tokenizer 最大长度 | 未归档 | 6,306/16,384 |
| loss-mask 错误 | 未归档 | 0 |

最终 110 条 endpoint gate：STP Disabled 13、BPDU filter 16、VRRP non-preempt 32、
VRRP/STP 角色错位 17、IP cycle 16、MPLS cycle 16；84/84 题均有 endpoint。45 条多设备
路径使用同题同快照的真实跨成功轨迹 recovery。

## 2. P0-1：无证据排除结论

### 原问题

v2 只要求排除整句与 earlier output 有少量术语重合。q0004、q0008、q0032 的结果只有空输出
或协议表头，却监督利用率、丢包、下一跳、出接口、MTU 和 HRP 等强结论。

### v3 修复

1. 原始排除句按事实标点拆分；推断前缀和程序性文本不直接监督。
2. 每个保留事实必须绑定更早 action ID 和 exact observation span。
3. 空结果、帮助、Legend、Flags、协议表头、截断标记均不得作为事实 span。
4. claim anchors 必须逐项在 span 中出现；负向 claim 还要求显式 0/no/none/not/disable。
5. 最终监督只复述 `action ID + exact span`，结论限定为“削弱与该回显直接矛盾的子候选”；
   未覆盖候选保持未决。

### 回归结果

旧错误行：

```text
q0004_path_05_success_08_step_03
q0008_path_04_success_06_step_03
q0032_path_01_success_06_step_03
```

最终产物中均为 0。独立验证器另用三组“表头/空结果 + 强 claim”固定反例确认不能通过。最终
只有 8 条排除样本（训练 7、验证 1），8/8 具有完整原子 binding。覆盖率明显降低，但不再用
高权重训练不可见事实。

## 3. P0-2：device/filename glob 漏检

### 原问题

v2 只检查 snapshot 字段，漏掉：

```text
saved_configs/CampusNetwork_07/*/display_current-configuration.txt
saved_configs/CampusNetwork_07/PE*/*.txt
saved_configs/CampusNetwork_05/PE1/*route*.txt
```

### v3 修复与结果

路径解析分别输出 `snapshot_glob`、`device_glob`、`filename_glob`，三者 OR 为 `has_path_glob`。
任何一项为真都不能进入正 loss。最终独立扫描：

| 范围 | cross snapshot | snapshot glob | device glob | filename glob |
| --- | ---: | ---: | ---: | ---: |
| train + validation semantic pool | 0 | 0 | 0 | 0 |

独立回归命令 `CampusNetwork_07/PE*/*route*.txt` 同时识别 device 与 filename glob，修复了 v2
检测器的结构性盲点。

## 4. P1-1：固定动作上限改为 claim coverage

v2 的 450 个调查节点中 403 个正好顶到 6。v3 不设数值上限：从当前源 turn 提取 device、
family 和 device/family pair，选择覆盖这些主张的最小动作集合。

结果：原始动作 10,870，过滤后合格 5,250，最终保留 514。312 个有动作节点全部满足
`claim_coverage_after == claim_coverage_before`。保留动作数为 1–6 的自然分布；只有 1 行为
6，不再形成统一顶格。每行 metadata 都记录 before/after 覆盖和 original/eligible/kept 数量。

## 5. P1-2：三节点 endpoint 合并

v2 每个 path 把同一长历史序列化三次。v3 的一条 `endpoint_bundle` 依次包含：

```text
summary thinking + summary conclusion
stop thinking + stop conclusion
final result
```

五条 assistant 消息分别带 loss，因果顺序不变，但上下文只前向一次。训练 endpoint schedule 从
216 行/轮降为 72 行/轮，总训练从 360 降为 216 行/轮。每题权重仍严格相同：2 core + 1
bundle。五轮覆盖 274/274 core rows 和 95/95 endpoint paths。

## 6. P1-3：模板与答案后验语气

监督文本不再出现内部 gate ID、“覆盖 N 个待验证设备的全部原子命题”或“继续调用工具不会
改变最小集合”。新 stop 明确区分：

- 当前直接事实足以支持候选并进入最终决策；
- 找到支持证据不等于已经排尽所有其他故障。

非 result 监督出现 exact `device;reason` 的行数为 0，`derived_from_verified_final_answer` 行数为
0。summary loss 降为 0.05，stop 保持 0.10；自动 summary+stop 的启发式加权占比为
23.20%–24.81%，未超过 25%，同时 72/72 训练 query 每轮仍有停止监督。

## 7. P1-4：IP/MPLS 环路可见证明

删除硬编码 `PE_CYCLES`。生成器从可见 `ip address` 行抽取 `address -> device` ownership，
再验证三条 forwarding facts 的 next-hop owner 构成闭环；MPLS 还要求每跳 out-label 等于下一
设备 in-label。独立验证器用另一套实现重建 ownership 和 cycle，未复用 converter 函数。

## 8. LR 可观测性

callback 现在每个 `on_step_begin` 都强制并写入 optimizer LR，不再只在 global step 0 写首条。
每阶段结束写 `train_end`，训练脚本验证审计链：

```text
train_begin -> step_begin × N -> train_end
```

任一 optimizer group 与当轮目标 LR 不一致都会中止。

## 9. 目标 tokenizer 与逐 token loss mask

服务器实际环境：Python 3.12.3、ms-swift 4.4.2、Transformers 5.12.1、Qwen3.6-27B，train
template，`loss_scale=default`，`is_binary_loss_scale=false`。

| dataset | rows | p99 | max | >16384 | mask failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| core | 274 | 4,076 | 6,306 | 0 | 0 |
| endpoint | 95 | 4,164 | 5,609 | 0 | 0 |
| validation | 63 | 3,693 | 4,034 | 0 | 0 |

报告逐 token 检查 `input_ids/labels/loss_scale` 等长，且 `label != -100` 当且仅当 scale > 0；
每条消息声明的 scale 均出现在实际 mask。报告哈希与 tokenizer/chat-template/config 哈希进入
manifest。

## 10. 最终验证

必须同时通过：

```text
python -B scripts/validate_0807_evidence_gated_reasoning_sft.py
python -B scripts/independent_validate_0807_sft.py
python -B scripts/audit_0807_evidence_gated_sft.py
```

独立验证器不导入 0807 converter，独立检查输出哈希、路径/glob、排除 binding、endpoint
bundle、IP/MPLS ownership、采样和固定反例。

## 11. 剩余限制

1. 验证仍是 case-isolated，不是 topology-heldout；按本轮要求不作为阻断项。
2. 排除错误候选只剩 8 条，高精度但覆盖偏低；不应放宽 span 合同来增加数量。
3. 45 条 cycle endpoint 是同题同快照跨成功轨迹 recovery，不是单条自然连续运行。
4. 同拓扑/同故障直接事实会在不同 query 重复；通过每题每轮 1 bundle 和低 summary loss 控制。
5. 尚未训练 0807 LoRA，最终收益需用固定 Agent 评测验证；同 5 epoch 与 0805 不是等 optimizer
   steps，横向实验应同时报告固定 epoch 与等 step 对照。

## 12. 结论

独立复审的训练阻断项已由“规则自洽”升级为“最终 JSONL 独立扫描 + 固定反例 + 目标 tokenizer
逐 token 检查”。0807 v3 推荐作为当前高置信正式训练候选；0805 可继续保留为更大路径多样性
基线，但不应覆盖或替代 v3。
