# 2026-08-09 SFT v7 生成与复现规范

本文是 0809 的权威复现记录。任何数据或脚本修改都必须同步更新本文件、同目录 README、根 README、manifest 和审计报告；随后重新生成、执行两套 validator、真实目标 tokenizer preflight，并记录运行环境门禁结果。

## 1. 冻结身份

- release：`0809`
- 逻辑 data root：`data/2026-08-09`
- manifest schema：`qwen36-0809-agent-error-aware-sft.v7`
- parent：`data/2026-08-05`，只读
- raw：84 题 × 10 条严格成功轨迹，共 840 条，与 0805 按相对路径逐字节一致
- split：72 train + 12 error-mining/dev，按 `case_id` 整题隔离
- validation case：`q2 q12 q19 q20 q29 q38 q65 q71 q85 q86 q99 q100`
- Agent eval 错误轨迹：只用于定义一般失败族，不进入训练 messages 或 loss

manifest 不保存 Windows/服务器绝对 data root。converter、validator、audit 和训练入口从自身 checkout 解析 `data/2026-08-09`，因此同一发布可在不同根目录验证；绝对路径只允许出现在诊断输出和 preflight 环境记录中，不参与跨主机相等门禁。

设计依据保存为 `curation/DESIGN_AUDIT_INPUT.md`。manifest 的 tracked dependency 闭环包含：generator 及其 0807 导入依赖、两套 validator、audit、preflight、训练入口、LR plugin、正式配置、本 README/复现文档、父级与发布侧 selection/clusters/filter/endpoint audit。不得依赖仓库外 `trajectory-analysis` 文件才能复现。

## 2. 不可变父级与训练侧清理

父级输入：

```text
data/2026-08-05/raw/
data/2026-08-05/curation/accepted_trajectory_selection.json
data/2026-08-05/curation/causal_path_clusters_per_case.json
data/2026-08-05/sft/qwen3_6_27b_reasoning_causal_path_train.jsonl
data/2026-08-05/sft/qwen3_6_27b_reasoning_causal_path_train_core.jsonl
data/2026-08-05/sft/qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl
data/2026-08-05/sft/qwen3_6_27b_reasoning_causal_path_validation.jsonl
data/2026-08-05/sft/reasoning_causal_path_manifest.json
```

生成前重算 raw 文件数、字节数和树哈希；任何差异终止。父级训练消息随后执行：

1. 所有 tool call（包括 loss=0 历史）必须从 user prompt 解析唯一当前 snapshot；跨 snapshot、不可解析 snapshot 和 `saved_configs` glob 的 call 及 FIFO 对应 response 一并删除。当前删除 183 个 call，按原因计数可重叠：glob 100、snapshot mismatch/unresolved 87。
2. 对 719 条父级 core 从 messages 重算同题全部严格 endpoint binding。若首个正 loss 前任一 binding 已完整，删除后续正工具并改为证据特有 stop；当前修复 50 行、移除 93 个正工具调用。q0023 的 1/1 BPDU 完整节点移除两个后续调用。
3. 对仍处于 planning/reasoning 且尾部有 1–2 个调用的父级节点执行完整 action 结构门禁。只有首个正 loss 前的 `tool_response` 可作为事实；不完整状态改写为“已匹配事实 + matched/expected + 缺失要求 + 待验证假设 + 当前动作”，并禁止闭环、唯一根因和最终答案。最终 308 条 action 对应的 308 条父级 source 全部记录结构 gate metadata。
4. validation 保持冻结比较口径，不执行训练侧增强；任何新增训练 metadata 都必须声明 `error_mining_agent_trajectory_used=false` 和 `validation_case_used_for_augmentation=false`。

基础设施失败和人为中断只供 runner 控制，不进入 raw、curation、SFT、报表或评测分母。

## 3. 六类决定性事实门禁

| 逐字可见事实 | 支持标签 | 仍需独立验证的邻近候选 |
| --- | --- | --- |
| `Protocol Status : Disabled` | 全局STP未使能 | BPDU filter、链路状态、泛化 STP |
| current configuration 中 `stp port bpdu-filter enable` | STP BPDU被过滤 | 全局STP、端口 down、日志文本 |
| 同前缀三条 route next-hop 精确绑定到三台设备接口 IP owner，并形成有向三环 | 存在IP路由环路 | MPLS、BGP、L3VPN |
| static-LSP next-hop 设备三环，且每跳 out-label 等于下一跳 in-label | 存在MPLS标签环路 | IP 环、L3VPN、无序标签集合 |
| `Preempt : NO` | VRRP工作在非抢占模式 | VRRP Master角色规划不合理 |
| 源主机 `/24` VLAN → 每台 Vlanif Master → 显式 MST mapping → 同实例 ALTE/discarding | VRRP Master角色规划不合理 | nonpreempt、缺源 VLAN 的局部 MST 现象 |

文件名、命令文本、表头、空输出、自动摘要和最终答案不能充当证据。metadata 保存 `command/snapshot/device/output_line/fact_kind`、解析结构和事实 SHA-256；两套 validator 都从样本先前 `tool_response` 逐字回查。恢复事实还必须回读真实成功 event/item/command/output/hash/span。

## 4. 可达 action、停止与错误候选排除

`counterfactual_action_selection` 从成功训练路径选择标签相关、1–2 个调用的真实节点：只允许当前 snapshot 的 Linux 相对只读 `exec_command + arguments.cmd`，禁止 discovery、glob 和完整证据后继续调用。源历史 loss 清零，只监督当前 thinking 0.35、rationale 0.25、tool call 0.05；tool response 仅作后续上下文。

共 308 条 action，覆盖 72/72 题。相同命令可在不同真实证据状态下出现；audit 以真实首正 loss 前缀加命令报告可达上下文，不把它们说成不同命令，也不以人工 marker 强制唯一。

action 的 factual grounding 以“首个正 loss”作为因果边界，并对同题所有 strict endpoint binding 完整匹配，而不是只扫描三类关键词。官方与独立 validator 各自从 messages 重算状态、核对 metadata 摘要，并以结构同义 fixture 覆盖 IP 三节点环、MPLS 标签闭环、BPDU Filter、关键路径闭合和否定表达。评审点名的 17 条 action 必须存在且保持未完整；当前 308/308 action 均为显式待验证，完整 action 为 0。

“已经形成 IP/MPLS 环路”属于证据不完整时禁止的肯定收敛句；“没有形成 IP/MPLS 环路”属于有价值的错误候选排除，不得被同一规则误删。两套 validator 都包含肯定命中和否定保留 fixture。

错误候选排除只由两部分组成：

- 父级真实 `hypothesis_elimination` 103 行，覆盖 58 题；
- 8 条从六个 label 分层选择的 exact same-target elimination replay，用于透明调整排除曝光。replay messages 与来源逐字一致，`counts_as_new_semantic_target=false`，不计为新路径或新语义。

另外 14 题没有真实排除节点，不补造排除。endpoint 仅做候选范围校准：正证据确认当前标签，但不能推出未经检查的邻近标签不存在；`rejected_labels=[]`、`hypothesis_elimination_supervised=false`，候选保持未验证。

malformed protocol、rejected action、错误最小集合和错误续查类别只存 metadata，SFT 不把坏 JSON/错答案当正目标。显式 hard negative 学习必须另建 grammar eval 或 preference/ranking 数据。

## 5. 一条真实证据路径一个 endpoint bundle

父级 711 个 endpoint 节点还原为 237 个 group 并重新运行上述事实门禁。父级不足的复杂题只从该题冻结的十条 0805 成功 `events.jsonl` 恢复，同 query、同 snapshot、成功只读命令是硬条件；validation 与 Agent eval 轨迹禁止使用。

- 116 条不同原始证据路径覆盖全部 72 题；47 题 1 条、9 题 2 条、13 题 3 条、3 题 4 条。
- 40 条恢复路径含 216 个事实，回指 45 个真实 event 文件；每个事实保存成功 item、完整 command、command/output SHA-256、逐字 raw span 及规范化 observation。
- 每条真实路径只生成一行 `endpoint_bundle`。原始工具历史后连续监督三个 assistant message：
  1. thinking：证据归纳、事实到正向标签边界、候选范围校准、当前证据最小集合与停止原因；
  2. grounded summary：支持标签、未验证候选、当前证据集合和停止判断；
  3. 验证过的最终 `<result>`。
- 不插入零 loss assistant objective marker，不再用 closure/contrastive 改写把同一路径计两次。
- 若一题有 `n` 条真实路径，每条 bundle 的 thinking/summary/final loss 分别为 `0.2/n`、`0.3/n`、`1.0/n`；每题 endpoint 总权重严格为 1。

endpoint 的 `endpoint_objectives` 必须同时包含 evidence_summary、label_boundary、candidate_scope_calibration、minimal_set、decision_ready、decision。`raw_first_positive_prefix_sha256` 基于真实首个正 loss 前全部可见内容；官方/独立 validator 要求 116 行对应 116 个不同原始前缀，并拒绝 unsupported closed-world exclusion。

## 6. 固定预算与 schedule

每轮完全相同：

```text
719 base core
+ 308 reachable action
+   8 exact same-target elimination replay
+ 116 real-path endpoint bundle
= 1151 rows

effective batch = 2 GPU × 1 micro batch × 4 accumulation = 8
optimizer steps = ceil(1151 / 8) = 144
fixed epoch 3 checkpoint = global step 432
```

所有新增目标每轮都曝光，故固定 checkpoint-432 前覆盖 308 action、8 replay 和 116 endpoint；不再把 boundary/minimal 推迟到第 4–5 轮。每轮有 1151 个唯一 row ID、1143 份唯一 message payload；仅 8 个透明 replay 与来源重复。endpoint 的每题路径总 loss 归一化，主决策不会因路径数量不同而失衡。

启发式加权 token 比例：thinking 58.323252%、结论/回答 37.136251%、tool call 4.540497%；counterfactual action 23.708317%、endpoint bundle 7.940465%、真实 hypothesis elimination 7.212316%、elimination replay 0.548548%。精确 tokenizer 比例以第 9 节报告为准。

## 7. 自然多目标的审计口径

发布数据包含 784 个不同原始首正 loss 前缀；116 组/325 行存在多个兼容正目标：61 组 reasoning+action、34 组 planning+action、21 组 planning 内部变体。它们来自原始/可达的自然 continuation，不是相互矛盾的结果。

validator 不再硬编码“所有目标前缀唯一”，也绝不插入推理时不存在的 assistant marker。audit 必须同时报告 raw prefix 数、多目标组/行和目标类型组合。endpoint 本身仍要求一条真实路径一个唯一 raw prefix。condition-normalized target diversity 只替换 snapshot/query/config path 等机械字符串，保留设备、观察、标签和结果。

必须区分两个维度：输入侧有 116/116 个不同 raw evidence prefix；输出侧精确正目标与 condition-normalized 正目标都只有 16/116 个唯一值，排除 q73–q86 inclusive-OR 组后为 7/104。后者反映同答案/同证据模板的真实重复，不通过机械释义强行增大；报告“116 条真实路径”时不得再写成“116/116 正目标唯一”。

## 8. 可移植发布与运行时身份

manifest 的 outputs、tracked files 和 design input 均使用仓库相对路径。两套 validator 必须在 Windows checkout 与不同绝对根目录的 Linux checkout 各运行一次。

正式入口允许通过环境变量覆盖 MODEL/config/plugin/Python/swift 路径，但放行前必须验证：

1. 实际 config 与 LR plugin 字节哈希等于 manifest 冻结版本；
2. Python 与 swift 的调用入口来自同一环境 bin；检查保留虚拟环境 symlink 入口，不用 `resolve()` 把 Python 误归到基础解释器；
3. 实际 ms-swift/transformers 版本等于 preflight；
4. MODEL_PATH 中 tokenizer.json、tokenizer_config.json、config.json、model.safetensors.index.json 等身份文件大小和 SHA-256 等于 preflight；
5. 所有 manifest outputs/dependencies 已 Git 跟踪且工作树干净；
6. stage 5 early-exit 前必须确认 checkpoint-432 存在。

`RUNTIME_GATE_ONLY=1` 可在不启动正式训练的情况下执行 Git、validator、preflight、identity 和 plugin registration 门禁。完整发布后还必须在双卡环境做 1–2 step full-state resume/LR smoke，核对 stage、epoch、global step、optimizer state 和强制 LR。
plugin registration smoke 显式注入合法的 stage 1、首轮 LR 和临时 audit 路径，只验证 external plugin 能注册 callback；不会加载模型或写 LR 训练审计。

## 9. 真实目标 tokenizer/loss-mask 预检

命令：

```bash
python -B scripts/check_0809_target_tokenizer_preflight.py \
  --data-root data/2026-08-09 \
  --model /path/to/Qwen3.6-27B
```

v7 归档环境为 Qwen3.6-27B、Python 3.12.3、ms-swift 4.4.2、transformers 5.12.1。脚本编码语义池、验证集、五份 core 和五份 endpoint，共 12 份/7151 行：min 1221、p99 4561、max 5635/16384、overlong 0、loss-mask failure 0。每轮精确 weighted supervised token 为 102247.45001797，endpoint bundle 10.398206%，action target 22.793820%。该报告由 v7 数据重新生成，旧版本哈希未复用。

preflight 冻结 tokenizer.json、tokenizer_config.json、chat_template.jinja、config.json、generation_config.json 和 model.safetensors.index.json。任何数据、loss、模板、模型身份或依赖版本变化都必须重跑；旧报告不得复用。

## 10. 复现顺序

```powershell
python -B scripts/convert_0809_agent_error_aware_sft.py --data-root data/2026-08-09
python -B scripts/validate_0809_agent_error_aware_sft.py --data-root data/2026-08-09
python -B scripts/independent_validate_0809_sft.py --data-root data/2026-08-09
python -B scripts/audit_0809_agent_error_aware_sft.py --data-root data/2026-08-09
```

preflight 写回后再次运行 converter → audit → converter，直到 manifest 冻结的 outputs/dependencies/audit/preflight 哈希稳定，再运行两套 validator。`RELEASE_RECORD.json` 记录 HEAD、dirty 状态、manifest/raw/output tree 哈希；未形成 Git commit 前只是工作树冻结记录，不等价于已发布版本。

正式训练：

```bash
bash scripts/train_qwen36_0809_agent_error_aware_5epoch.sh
```

正式配置固定双卡 DDP、每卡 batch 1、累积 4、五阶段 LR `2e-5, 1.5e-5, 1e-5, 6e-6, 3e-6`，每阶段用上一阶段 checkpoint 完整恢复 model/optimizer/scheduler/Trainer 状态。固定使用 epoch 3/checkpoint-432，不按 Agent 小样本、eval loss 或事后结果重新选点。

## 11. 评测边界

现有 12 题只用于 error-mining/dev 和历史 checkpoint 横向比较，不能单独证明新拓扑泛化。正式结论需要未参与反推的 topology-heldout Agent 集。q73–q86 使用 scoring v2 inclusive-OR；基础设施失败与人为中断不进分母；无有效答案、错误归因、错误集合和 malformed tool call 均计模型错误。

训练后至少报告：六类 full/partial/none、首次决定性事实前命令数、full 后继续命令数、exact/FP/FN/额外项率、malformed tool-call 率、各故障族、global step、样本曝光和 supervised token。正式训练前仍必须完成双卡 resume/LR smoke 和小规模 Agent canary。
