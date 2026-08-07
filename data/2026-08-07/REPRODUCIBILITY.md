# 2026-08-07 SFT 生成与复现规范

本文是 0807 v7 的权威复现记录。任何来源、筛选、划分、聚类、节点、消息、system prompt、
工具协议、loss、采样、tokenizer、训练入口、验证器或生成物修改，都必须在同一变更中更新
根 README、本目录 README、本文和审计报告，提升 schema，重生成全部派生文件，并通过官方与
独立验证器。

## 1. 版本与放行状态

- manifest：`qwen36-0807-evidence-gated-case-balanced-sft.v7`；
- selection：`0807-evidence-gated-cluster-selection.v6`；
- curation：`codex-ip-accepted-trajectory-curation.v5`；
- audit metrics：`0807-evidence-gated-audit-metrics.v6`；
- tokenizer report：`0807-target-tokenizer-loss-mask-preflight.v1`；
- 状态：`rule_and_target_tokenizer_validated_release_candidate`。

“release candidate”表示静态数据合同和目标 tokenizer 合同通过，不表示训练后 Agent 效果通过。

## 2. 不可变输入

### 2.1 轨迹与状态口径

来源实验：`experiments/2026-08-02-ip_codex_gpt56-sol_100x10/`。0807 `raw/` 保留相同的 840 条
事件与模型输出；q73–q86 的 140 个 raw 仅同步了新 reference options 与修订 provenance，因此
不再与 0805 对应 JSON 逐字节相同。每个 case 恰好 10 条 `accepted`；状态统计仅包含模型有效结果：

| status | count |
| --- | ---: |
| accepted | 840 |
| incorrect | 502 |
| format_error | 0 |

基础设施失败和中断不归档、不计数、不聚类、不训练。q41–q56 没有 accepted，不进入 0807。

关键输入 SHA256（LF 规范化）：

| 输入 | SHA256 |
| --- | --- |
| `data/simulation/train_0629.jsonl` | `2c9faac68eb9e366eaedd2ad732e599d220c9a38a07bfac8ce5597cb0315febf` |
| `data/2026-08-07/curation/accepted_trajectory_selection.json` | `3884796a05badd5dce9ea02b187a1eb1dcf53500d770afc1f97c255a73e295bb` |
| `data/2026-08-04/curation/accepted_trajectory_selection.json` | `ccea8437f7a7663fa1dd0fec232da7c825e1d3a4d6485db5cc83ac4ff4870070` |
| `config/codex_qwen_model_catalog.json` | `de0ebe3b59453783657b846d2957002c81b14ad287c0d3fe9dafd830dcba3d31` |

### 2.2 冻结划分

分组键为 `case_id`。训练 72 题，验证 12 题：2、12、19、20、29、38、65、71、85、86、
99、100；交集为 0。划分与 0804 完全相同，适用于 checkpoint 横向比较，但不是
topology-heldout。

### 2.3 system prompt 与工具协议

每行 system prompt 逐字来自 `config/codex_qwen_model_catalog.json` 中
`Qwen3.6-27B-trained.base_instructions`，manifest 记录内容哈希。

唯一监督工具为：

```text
name: exec_command
arguments: {"cmd": "..."}
OS: Linux
path: repository-relative saved_configs/...
```

只接纳等价的只读 `cat/grep/find/head/tail/test`。源 PowerShell、规范化 PowerShell、Linux
命令、翻译类型、源/目标哈希保存在 metadata。任何无法可靠转换的命令导致生成失败。

### 2.4 q73–q86 inclusive-OR 修订

输入影响报告：`trajectory-analysis/2026-08-07_q0073-q0086_inclusive_or_impact.md`，LF 规范化
SHA256 为 `8c3e5231ffd1ed2516a9ae6871a208b3921a7b4f02b29407d2056c83985d1164`。
q73–q86 的 `answer` 必须恰好包含三个互异集合：`[A]`、`[B]`、`[A,B]`。源答案更新后运行
`sync_0807_q73_q86_inclusive_or.py`，同步 140 个 raw、curation options、修订字段和哈希。
现有 140 条轨迹均为单设备且仍被接受，故成功数和 72/12 划分不变。

SFT 不直接把 evaluator 的宽松范围当作 target 分布：优先选择证据最强的 singleton；双设备
target 只有在两个设备各自满足源 VLAN/VRRP/MST instance/STP role 完整闭环时才可进入监督。

## 3. v7 生成算法（语义转换沿用 v6）

### 3.1 可见事件与 label 隔离

解析可见 `agent_message`、成功工具命令、exit code 和 output。隐藏 chain-of-thought 不可见，
不声称恢复。verified label 只用于严格最终答案和已抽取直接事实的 post-hoc endpoint gate；
不参与动作评分、轨迹质量、路径聚类或代表选择。

事实性 thinking 原句只用于选择相关的更早证据，不再直接参与 loss。生成器从此前工具结果中
提取非空、非帮助、非表头 exact observation atoms，逐 atom 绑定 action ID 与 span，并把
原句保存在 metadata 审计。empty output、表头、帮助、截断标记及 `unselected lines omitted`
不能支持正向或 absence 断言。安全程序性原句可保留；未来或不受支持句删除。当前为 78 条
observation-bound 事实归纳、14 条安全程序性原句、219 条删除句；旧关键词 factual grounding
为 0。独立 validator 固定检查复审报告列出的 21 个越证据 row ID。

### 3.2 三层 glob 检测

对规范化 `saved_configs/<snapshot>/<device>/<filename>` 分别检测 `* ? [`：

- `has_snapshot_glob`；
- `has_device_glob`；
- `has_filename_glob`；
- `has_path_glob` 为三者 OR。

任一 glob、跨题目快照、discovery 或不支持的命令均不得成为正 loss 动作。当前四项结果：
cross-snapshot 0、snapshot glob 0、device glob 0、filename glob 0。独立验证器内置
`CampusNetwork_07/PE*/*route*.txt` 回归样本，必须同时命中 device 和 filename glob。

### 3.3 claim coverage 动作选择

取消固定动作数上限。每个源 conclusion 提取 `device:*`、`family:*` 和 `pair:device/family`
下一动作意图，先
过滤为同快照、无 glob、非 discovery 的候选，再按 label-independent 优先级执行贪心最小
set cover。无可提取 claim 的 planning turn 只保留一个最高信息增益动作。

每行归档原始/合格/保留数量、源意图覆盖、最终监督意图覆盖和 selected-action bindings。
294 个有动作节点的源覆盖 full/partial/zero/unscoped 为 117/20/15/142；最终监督覆盖为
191/0/0/103。原始 conclusion 只有全部 claim units 都被当前动作覆盖才可保留；否则按当前动作
重建。validator 要求最终监督只能是 full/unscoped。全语义池最终保留动作 369；配置类 10
（2.71%），LLDP 10，LLDP→MPLS 误标 0。

### 3.4 精确路径聚类

路径 key 包含快照、设备、文件/查询、family、接口、前缀和方向；trajectory threshold 0.46，
node duplicate threshold 0.76，每题最多保留 4 个互异候选路径，但最终只有通过 endpoint gate
的路径保留。结果：3,505 raw visible checkpoints、3,016 causal prefix checkpoints、753 path
clusters、104 retained paths。

### 3.5 claim-bound hypothesis elimination

每题最多选择一个源排除 turn。处理顺序：

1. 按 `：,，、；。！？` 拆为原子事实，去掉推断前缀和程序性分句；
2. 从更早保留动作的可见输出中寻找 exact span；
3. 空结果、帮助、Legend、Flags、协议表头和截断标记禁止作为正证据；
4. 每个 anchor 必须在绑定 span 中逐项出现；负向 claim 还要求 span 明确包含 0/no/none/not/
   disable 等负证据；
5. 监督文本仅引用 `action ID + exact span`，并把结论限制为“削弱与该回显直接矛盾的子候选”；
   未覆盖候选保持未决。

只有 exact span 能映射到明确反证范围（CRC、持续 down、带宽拥塞、MTU 或与已显示出接口冲突
的绕行）才生成节点；泛化的“与回显矛盾的子候选”不进入训练。结果为 12 行（训练 11、验证
1）。以下旧错误行必须为 0：

```text
q0004_path_05_success_08_step_03
q0008_path_04_success_06_step_03
q0032_path_01_success_06_step_03
```

### 3.6 endpoint gate 与可见拓扑证明

支持六类 gate：STP Disabled、BPDU filter、VRRP 非抢占、VRRP Master/STP Alternate 角色错位、
三设备 IP 静态路由环、三设备 MPLS 静态 LSP 标签环。前四类引用目标设备的直接行；环路必须
同时具备三条 forwarding facts 和三条可见 `ip address` ownership facts，并从这些事实构建
`device -> next-hop owner` 三节点闭环。生成器不含硬编码 PE next-hop 常量。

角色错位 gate 额外要求：从题目源主机的 `/24` 地址推导 VLAN；VRRP Master 必须属于同 VLAN
Vlanif；同设备回显必须显式给出包含该 VLAN 的 MST instance mapping；Alternate/Discarding
必须落在这个实例。双设备答案分别闭环，禁止任意 Vlanif 与任意实例拼接。

104 条 endpoint：14/15/29/14/16/16。跨轨迹 recovery 只使用同题、同快照不同成功轨迹；
每个 donor action 与 output 哈希均归档。summary/stop 不含 exact `device;reason`，内部 gate ID
不进入监督文本，`derived_from_verified_final_answer=false`。

### 3.7 单行多目标 endpoint

每个 path 的 summary、stop、decision 在一个 `endpoint_bundle` 中共同计算 loss：summary 两条
assistant 消息、stop 两条 assistant 消息、最终 result 一条 assistant 消息。三段因果顺序不变，
但同一工具历史只序列化一次。停止文本只表示“当前证据支持候选并达到作答强度”，不声称所有
其他故障已被证明不存在。

## 4. 语义池与五轮采样

| split | planning | reasoning | source stop | elimination | endpoint bundle | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 89 | 155 | 0 | 11 | 91 | 346 |
| validation | 13 | 25 | 0 | 1 | 13 | 52 |

训练 core pool 255；endpoint pool 91。每轮确定性 round-robin：72 × 2 core = 144；72 × 1
bundle = 72；合计 216 行、每题 3 行。五轮覆盖全部 255 个 core source rows 和 91 个 endpoint
paths。q89 只有一个高置信 core，因此每轮对其做显式 replay，不合成无证据节点。候选池不能
直接用于正式训练。

## 5. loss 合同

| source | scale |
| --- | ---: |
| `pruned_original_visible_agent_message` thinking | 1.00 |
| `observation_bound_reasoning_reconstruction` thinking | 0.60 |
| `claim_bound_hypothesis_elimination` thinking | 0.60 |
| fixed/final bridge thinking | 0 |
| original visible conclusion | 1.00 |
| evidence-aligned reconstruction | 0.30 |
| source-grounded elimination conclusion | 0.60 |
| evidence summary thinking/conclusion | 0.05 / 0.05 |
| stop thinking/conclusion | 0.10 / 0.10 |
| verified final result | 1.00 |
| current tool call | 0.02 |
| system/user/history/tool response | 0 |

训练必须传 `--loss_scale default --is_binary_loss_scale false`。每轮启发式区间：thinking
42.07%–43.89%，tool 0.81%–0.90%，自动 summary+stop 22.89%–23.72%。该启发式仅用于
分布审核，不替代第 7 节真实 tokenizer 报告。

## 6. 生成物

以下哈希是数据语义稳定后的 JSONL SHA256（LF 规范化）；文档或脚本改变会使 manifest 自身和
selection 文档哈希变化，最终值以 manifest 为准。

| output | rows | bytes | SHA256 |
| --- | ---: | ---: | --- |
| train semantic | 346 | 5,085,830 | `085cdb1bf6dd5ce936c9d31628dc7f162c3de2cc79832bc3fe1564af338335eb` |
| train core pool | 255 | 3,781,353 | `bcc5e85e96cf8e6d7c90c1f9c28309fe49135de32fa1cd22ad80bba79dd31881` |
| train endpoint pool | 91 | 1,303,457 | `11021af45863202f85ab9aaee0bd9db072877456b575393673f73c6b710176a5` |
| validation | 52 | 793,250 | `9bcd99c73eaaa1a8b26a24c4b8c2eb8096460e0bacf47f70094086f7f9927961` |
| core epoch 01 | 144 | 2,151,461 | `1ce44401cef7dea1c54801f5ccc48e76c8a099de41b3a0fe6d4a71c1b3a4984c` |
| core epoch 02 | 144 | 2,158,886 | `6567dd50fff5bd39b2a6281cd3f4a33416d0c1a053200eea49a3dc957b5f59db` |
| core epoch 03 | 144 | 2,089,715 | `2bcf039af4507440548e00c862d595600d671cc2bfff3601db0d17d2738bf7a8` |
| core epoch 04 | 144 | 2,152,492 | `0ee25fe352ee857c20752b0a342dbb290c52437a95cf1b05a6603c744c93eafe` |
| core epoch 05 | 144 | 2,150,748 | `92f06cf98f8b43c88cbc699b888f9a275d76d19fb5ac699eaafccb44565b25ca` |
| endpoint epoch 01 | 72 | 1,051,449 | `5c1f3b288182bcbde61e60d994177db3c81f283d50df8da9f35a4b90d031411f` |
| endpoint epoch 02 | 72 | 1,052,307 | `6aeecf03f50c31bd85e306504cb87ce0a6e6cb1c7c6b5bc0c50b3359865ac82c` |
| endpoint epoch 03 | 72 | 1,057,101 | `ab444e512188284fdf6802927cc330f1116a3f26f0c1b61dd8d3c64e79a3169a` |
| endpoint epoch 04 | 72 | 1,050,539 | `434ad7edd15ebcdf2714579070520225d4488313d9fa45180299f41964c3ab41` |
| endpoint epoch 05 | 72 | 1,054,225 | `123f6474c5ddfb1e136a1a9caff0695d623da84dadb207ab1459c678df5fdc64` |

## 7. 目标 tokenizer 与 loss mask

服务器：Python 3.12.3、ms-swift 4.4.2、Transformers 5.12.1、Qwen3.6-27B tokenizer，
`template.set_mode('train')`、`loss_scale=default`、`is_binary_loss_scale=false`、release max
length 16,384。

报告：`data/2026-08-07/sft/TARGET_TOKENIZER_PREFLIGHT.json`，SHA256
`360dbb0b730878df7e14e47c8542088198c665d34bc76a61315d84e681d93dfc`。

| dataset | rows | p99 | max | overlong | mask failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| core pool | 255 | 3,642 | 4,044 | 0 | 0 |
| endpoint pool | 91 | 4,006 | 4,146 | 0 | 0 |
| validation | 52 | 3,383 | 3,578 | 0 | 0 |
| all | 398 | 4,001 | 4,146 | 0 | 0 |

检查要求：`input_ids/labels/loss_scale` 等长；`label != -100` 当且仅当 token loss scale > 0；
每条消息声明的正 loss scale 必须实际出现在 token mask；所有行长度 ≤ 16,384。报告还记录
tokenizer.json、tokenizer_config.json、chat_template.jinja、config.json 的原始 SHA256。

## 8. 验证器合同

官方验证器 `validate_0807_evidence_gated_reasoning_sft.py` 检查 manifest、来源、元数据、prompt、
工具翻译、grounding、endpoint、采样和训练合同。独立验证器
`independent_validate_0807_sft.py` 不导入 0807 converter，独立解析路径/glob、ordinary
reasoning action/span bindings、21 条已知越证据 reasoning、elimination bindings、cycle
ownership、bundle loss、动作覆盖和 schedule，并运行 empty/header/truncated-config 固定反例。
v6 还独立校验 q73–q86 的三个 reference options、140 个 raw/curation 哈希、错误 VLAN/
instance、LLDP/LDP token 边界、混合事实/计划固定反例、最终监督 partial/zero 为 0，以及
五阶段 27-step/global-step/checkpoint 后置条件。

两者必须同时输出 passed。`audit_0807_evidence_gated_sft.py` 生成
`curation/AUDIT_METRICS.json`，统计四类 path 风险、动作分布、重复率、冻结验证重合、loss 和
长度。任何单个验证失败都禁止正式训练。

## 9. 精确复现命令

本地归档与生成：

```powershell
python -B scripts/update_q73_q86_inclusive_or.py
python -B scripts/convert_accepted_only_100x10_to_sft.py `
  --output-root data/2026-08-07 --archive-only
python -B scripts/sync_0807_q73_q86_inclusive_or.py
python -B scripts/convert_0807_evidence_gated_reasoning_sft.py
python -B scripts/validate_0807_evidence_gated_reasoning_sft.py
python -B scripts/independent_validate_0807_sft.py
python -B scripts/audit_0807_evidence_gated_sft.py
python -m py_compile scripts/convert_0807_evidence_gated_reasoning_sft.py `
  scripts/validate_0807_evidence_gated_reasoning_sft.py `
  scripts/independent_validate_0807_sft.py `
  scripts/check_0807_target_tokenizer_preflight.py
git diff --check
```

目标 tokenizer：

```bash
/root/autodl-tmp/envs/qwen36-sft/bin/python \
  scripts/check_0807_target_tokenizer_preflight.py \
  --model /root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B \
  --max-length 16384 \
  --output data/2026-08-07/sft/TARGET_TOKENIZER_PREFLIGHT.json
```

训练入口：

```bash
RUN_MODE=prepare bash scripts/train_qwen36_0807_evidence_gated_5epoch.sh
RUN_MODE=dry-run bash scripts/train_qwen36_0807_evidence_gated_5epoch.sh
RUN_MODE=train bash scripts/train_qwen36_0807_evidence_gated_5epoch.sh
```

正式参数：Qwen3.6-27B LoRA，rank 8，alpha 32，dropout 0.05，max_length 16,384；GPU 0、1
双进程 DDP，每卡 batch 1、gradient accumulation 4、全局有效 batch 8；constant scheduler、
warmup 0，五轮固定 LR
`2e-5/1.5e-5/1e-5/6e-6/3e-6`。第 2–5 阶段恢复模型、optimizer、scheduler 和 Trainer
状态；callback 在每个 step 强制并记录 LR，训练后验证 `train_begin -> step_begin* -> train_end`。
每阶段输入必须恰为 216 行，因此新增 optimizer step 必须恰为 27；五阶段开始/结束 global step
固定为 `0→27→54→81→108→135`，训练脚本逐阶段要求连续的 27 个 `step_begin`，并只接受
精确命名的 `checkpoint-27/54/81/108/135`。本轮只完成静态合同、服务器 tokenizer 预检与
脚本语法/结构审查；未冒充执行高成本的真实两阶段 GPU resume smoke test。
五轮 eval loss 仅作诊断，不参与 checkpoint 选择；不运行6题×2次 Agent 选点，固定使用
epoch 3 的 `epoch_03/checkpoint-81` 对全部12道验证题各运行5次。LR callback 在两个 rank
都强制和核验学习率，但只有 rank 0 写 `learning_rate_audit.jsonl`。

## 10. 已知限制

1. 冻结验证是 case-isolated，不是 topology-heldout；工具调用和结果仍与训练有表面重合。
2. 12 条 elimination 是高精度、低覆盖版本；未能映射到明确反证范围的排错句被有意舍弃。
3. 部分环路 endpoint 使用同题同快照跨成功轨迹恢复，虽然有完整 provenance，但不是单条执行
   内的自然连续历史。
4. endpoint 直接事实会在同拓扑/同故障模板的不同 query 间重复；当前用每题每轮一次 bundle、
   summary 0.05 和 stop 0.10 控制权重。
5. 尚未进行 0807 LoRA 训练或 Agent 端到端评测；release candidate 仅是数据放行状态。
6. 当前 q73–q86 的 140 条来源均为 singleton；dual target 的双闭环分支已有规则和验证，但尚无
   真实双设备训练样本覆盖。

## 11. v7 变更记录

相对 0807 v6：

1. 正式训练从单卡、梯度累积 8 改为与 0805 相同的 GPU 0、1 双进程 DDP、每卡 batch 1、
   梯度累积 4；全局有效 batch 保持 8，216 行仍严格对应每阶段 27 optimizer steps；
2. LR callback 改为所有 rank 强制和校验学习率、仅 rank 0 写审计文件，避免并发重复记录；
3. checkpoint 策略改为与 0805 相同的固定 epoch：五轮 eval loss 只作诊断，禁用 Agent
   checkpoint selection，固定以 epoch 3 / `checkpoint-81` 执行12题×5次最终验证；
4. manifest 升级为 v7、audit metrics 升级为 v6；selection 和全部语义 JSONL 保持 v6 内容，
   tokenizer/loss-mask 结果不变。

## 12. v6 变更记录

相对 0807 v5：

1. 新增独立 `lldp` 邻接拓扑 family；纯字母数字 family pattern 使用 token 边界，修复
   `ldp in lldp` 导致的 LLDP→MPLS 误标；官方与独立 validator 从 command/filename 重新推导
   family，并固定检查审计报告涉及的 LLDP 行；
2. mixed thinking 先切为事实/计划子句：事实只能改写为更早 exact observation，纯未来计划才可
   原样监督；`再/随后/然后` 只有连接明确检查动作时才可作为计划前缀；
3. 原始多步 conclusion 只有在设备/协议意图全部被当前动作覆盖时才保留，否则重建为当前动作
   的精确计划；最终监督 intent coverage 固定为 full/unscoped，partial/zero 一律阻断生成；
4. 排除候选仍回看未切碎的原始完整句，再对其中事实逐项绑定，避免 mixed-sentence 清洗使高精度
   elimination 从 12 条意外退化；当前仍为 12 条、13 个绑定原子事实；
5. 每阶段训练合同固定为 216 行、梯度累积 8、27 optimizer steps，global-step 边界为
   `0/27/54/81/108/135`，且 checkpoint 名必须精确匹配 `27/54/81/108/135`；
6. schema 升级为 manifest/selection v6、audit metrics v5；重建后 train/validation 为
   346/52、retained paths 104、每轮仍为 216；真实 tokenizer 398 行最大 4,146/16,384，
   overlong 与 loss-mask failure 均为 0。

## 13. v5 变更记录

相对 0807 v4：

1. q73–q86 源答案规范为三个 exact-set 选项 `[A]`、`[B]`、`[A,B]`，同步 140 个 raw 与
   curation；现有成功数、题目划分和源 target 不变；
2. evaluator 的 inclusive-OR 接受范围与 SFT target 分离，SFT 选 evidence-strongest singleton；
3. VRRP/STP 角色错位 gate 增加源主机 VLAN、同 VLAN Vlanif Master、显式 MST 映射和同实例
   Alternate/Discarding 四段闭环，双设备 target 需分别闭环；
4. 旧 15 个角色终点中仅 4 个满足新合同；重建后 q73–q86 为 14/14 严格终点，不安全终点为 0；
5. q73–q86 语义行 48→46；全量 retained paths 104→103、训练语义 346→344，验证仍为 53，
   每轮仍为 216；
6. 官方与独立 validator 新增 reference 同步、严格闭环和旧 q86 错 VLAN/实例反例；schema 升级
   为 manifest/selection/curation v5、audit metrics v4，并重跑真实 Qwen tokenizer/loss mask。

## 14. v4 变更记录

相对 0807 v3：

1. 普通 factual reasoning 从关键词共现改为 exact observation atom/action/span 绑定；原始推断
   只留 metadata，不直接参与 loss；
2. empty output、纯表头、帮助文本与 `unselected lines omitted` 固定为无效正证据；
3. 独立 validator 新增 21 条越证据 reasoning 回归，并逐条检查普通 reasoning bindings；
4. 排除节点仍从原始可见句选择，但只监督 exact span，且必须映射到明确反证范围；
5. 动作覆盖拆分为源 conclusion 意图和最终监督意图，公开 full/partial/zero/unscoped，禁止把
   0→0 记为成功；
6. 无动作的伪 stop core 被删除；q89 单 core 使用显式 replay，避免合成无证据训练节点；
7. schema 升级为 manifest/selection v4、audit metrics v3；所有派生池、五轮 schedule、文档和
   目标 tokenizer 报告必须重新生成。

## 15. v3 变更记录

相对 0807 v2：

1. 修复 device/filename glob 漏检并拆分三层指标；
2. 排除结论改为 atomic claim/action/span 绑定，51 条收紧为 8 条；
3. 固定 q0004/q0008/q0032 与 device-glob 回归；
4. 固定 6 动作截断改为无硬上限的最小 claim cover；
5. 三条 endpoint 合并为一条多目标 bundle，每轮 360 降到 216 行；
6. 去除监督文本中的 gate ID 和“已证明根因全集”断言；
7. IP/MPLS 环路 next-hop ownership 改由可见接口地址事实构建；
8. summary loss 降到 0.05，stop 保持 0.10；
9. callback 改为逐 step LR 审计；
10. 新增独立验证器及真实 Qwen tokenizer/loss-mask 报告。

## 16. 后续修改强制流程

1. 先记录不可变输入和变更原因；
2. 修改生成/验证/训练脚本；
3. 提升 manifest/selection schema；
4. 重生成全部 JSONL、selection、manifest 和 audit metrics；
5. 更新根 README、日期 README、本文、审计报告和 changelog；
6. 在目标 Qwen 环境重新生成 tokenizer/loss-mask 报告；
7. 再次重生成 manifest，使报告与脚本哈希进入追踪；
8. 运行官方验证、独立验证、py_compile、bash -n、dry-run 和 `git diff --check`；
9. 所有文档计数、哈希与 manifest 一致后才允许提交、推送或正式训练。
