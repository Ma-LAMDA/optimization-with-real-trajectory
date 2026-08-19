# 2026-08-09 Agent 错误感知 SFT

0809 以 `data/2026-08-05/` 为只读父版本：840 条严格成功 raw 轨迹、72/12 整题隔离划分和因果路径聚类保持不变；不修改 0805、0807 或 `data/simulation/`。0809 有独立 generator、manifest、两套 validator、audit、真实目标 tokenizer preflight、训练配置和正式入口。

当前为 v7，落实 `trajectory-analysis/2026-08-11_2002_0809_sft_v6_scheme_continued_audit.md`：action 门禁从少数关键词扫描升级为同题完整结构 binding 重算；完整证据改为停止，未完整证据改为显式待验证假设。endpoint 不再凭正证据闭世界排除邻近标签，而是监督候选范围校准。

## v7 核心规则

- manifest 只记录 `data/2026-08-09` 等 checkout 内逻辑路径；设计输入复制为 `curation/DESIGN_AUDIT_INPUT.md`，生成器、父级选择/聚类、发布 curation、配置、训练入口和校验器全部进入 manifest 依赖闭环。
- 删除全部“下一节点只……”类零 loss assistant 控制句。新增目标直接接在真实 Agent 可见历史后；自然的兼容多目标会在 audit 中报告，不用训练时专有 marker 掩盖。
- 重新门禁父级 237 个 endpoint group：保留/恢复 116 条不同原始证据路径。每条路径只产生一个自然 completion，在其中同时监督证据归纳、正向标签边界、候选范围校准、当前证据最小集合、停止判断和最终 `<result>`。
- 72 题真实路径数分布：47 题 1 条、9 题 2 条、13 题 3 条、3 题 4 条。每条路径 loss 乘 `1 / 本题真实路径数`，因此每题 endpoint 总权重相同，不因路径多而被放大。
- 40 个复杂题的恢复证据只来自同一 query、同一 snapshot 的 0805 成功事件。40 条恢复路径包含 216 个事实，逐条回指 45 个真实 `events.jsonl` 中的成功 item、原始命令、输出哈希和逐字 span；carrier bundle 不充当事实来源。
- 308 条 action 都来自训练题成功路径的真实可达上下文，覆盖 72/72 题。两套 validator 从首个正 loss 前的 `tool_response` 重算同题所有严格 endpoint binding，不信任 metadata；308/308 均为结构未完整状态，完整 action 为 0。
- 每条 action 明示已匹配事实、`matched/expected`、缺失要求和“待验证假设”；结果返回前禁止输出最终答案或宣布闭环、唯一根因。评审点名的 17 条均纳入结构 fixture。工具 call loss 为 0.05，thinking/rationale 分别为 0.35/0.25。
- endpoint 输入侧仍是 116 个不同 raw evidence prefix；输出侧精确与 condition-normalized 正目标均为 16/116 唯一（排除 q73–q86 后 7/104）。两者分别报告，不用机械释义制造虚假多样性。
- 保留父级 103 条真实 `hypothesis_elimination`（58 题）和 8 条按 label 透明采样的 exact same-target replay。其余 14 题不补造排除；116 条 endpoint 的邻近标签均保持未验证，`rejected_labels` 声明为 0。
- 跨 snapshot、无法解析 snapshot 和 `saved_configs` glob 的 call/response 从训练历史删除。50 条在完整结构证据后仍有正工具调用的父级节点已改为停止，共移除 93 个调用；q0023 的 1/1 BPDU 完整节点移除两个后续调用。基础设施失败和中断不进入日期数据、统计或训练。

## 数据规模与训练预算

| 产物 | 行数 | 说明 |
| --- | ---: | --- |
| 父级 core | 719 | 清理工具历史并修复 post-closure 行 |
| action pool | 308 | 全部训练题的真实可达高信息动作上下文 |
| elimination replay | 8 | 透明同目标采样，不冒充新样本语义 |
| endpoint pool | 116 | 一条真实证据路径一个综合 completion |
| 训练语义池 | 1151 | 719 + 308 + 8 + 116 |
| 验证 | 245 | 冻结 12 题，仅作 error-mining/dev |
| 每轮 core schedule | 1035 | 719 base + 308 action + 8 replay |
| 每轮 endpoint schedule | 116 | 全部真实证据路径，每题权重归一化 |
| 每轮总曝光 | 1151 | 有效 batch 8，144 optimizer step |

五轮 schedule 相同且每轮曝光全部目标；固定 checkpoint 为 epoch 3 / global step 432，不做 Agent 或 eval-loss checkpoint 选优。硬件协议与 0805 相同：双卡 DDP、每卡 batch 1、梯度累积 4、有效 batch 8、五阶段固定 LR、完整 model/optimizer/scheduler/Trainer 状态续训。

启发式加权信号每轮相同：thinking 58.32%、结论/回答 37.14%、工具调用 4.54%；action 23.71%、endpoint bundle 7.94%、真实错误候选排除 7.21%、elimination replay 0.55%。精确 tokenizer 统计以本版本重新生成的 `TARGET_TOKENIZER_PREFLIGHT.json` 为准。

## 真实目标 tokenizer 预检

v7 已在 SeetaCloud 的 Qwen3.6-27B、Python 3.12.3、ms-swift 4.4.2、transformers 5.12.1 环境，对语义池、验证集和五轮 core/endpoint 共 12 份数据重新执行真实模板编码与逐 token loss-mask 检查；旧版本报告未复用：

- 总计 7151 行；min 1221、p99 4561、最大 5635/16384；overlong 0；loss-mask failure 0。
- 每轮精确加权 supervised token 为 102247.45001797；endpoint bundle 占 10.398206%，action target 占 22.793820%。
- 归档 tokenizer.json、tokenizer_config.json、chat_template.jinja、config.json、generation_config.json 和 model.safetensors.index.json 的大小与 SHA-256，并绑定 ms-swift/transformers 版本。

正式入口会重新计算实际 `MODEL_PATH` 的上述身份文件，比较实际 config/plugin 字节与 manifest，并核对 Python、swift 和 transformers 版本；override 路径可以变化，但内容和运行环境身份不能变化。

## 生成与验证

在仓库根目录运行：

```powershell
python -B scripts/convert_0809_agent_error_aware_sft.py --data-root data/2026-08-09
python -B scripts/validate_0809_agent_error_aware_sft.py --data-root data/2026-08-09
python -B scripts/independent_validate_0809_sft.py --data-root data/2026-08-09
python -B scripts/audit_0809_agent_error_aware_sft.py --data-root data/2026-08-09
```

目标环境预检：

```bash
python -B scripts/check_0809_target_tokenizer_preflight.py \
  --data-root data/2026-08-09 \
  --model /path/to/Qwen3.6-27B
```

正式训练入口是只读消费者，不现场运行 converter/audit/preflight：

```bash
bash scripts/train_qwen36_0809_agent_error_aware_5epoch.sh
```

训练前还必须满足：干净 Git 发布、Linux 异根目录两套 validator、runtime identity gate、双卡 1–2 step full-state resume/LR smoke，以及小规模 Agent canary。hard negative 仍只存 metadata；坏 JSON 不作为 SFT 正目标，需用 grammar eval 或 preference/ranking 数据显式学习。

## 评测与维护边界

当前 12 题已经用于错误模式分析，只能作为 error-mining/dev 和历史 checkpoint 横向诊断；最终泛化结论必须使用未参与反推的新 topology-heldout Agent 集。0809 与 0807 统一使用 `scripts/final_answer_scoring.py` 的 `agent-final-answer.v4.2026-08-12-incomplete-result-exact-recovery`：正确性只看最终答案，q73–q86 inclusive-OR 由同一入口处理；过程中的错误归因或 malformed tool call 只作诊断，不能覆盖一个精确可接受的最终答案。v4 只新增一个保守恢复分支：若全篇恰好只有一个完整 fenced `<result>` JSON 字符串列表、仅缺 `</result>`，且列表精确匹配可接受答案，则计为正确；其他残缺、冲突或多结果输出仍计错。终态无有效最终答案计模型错误；基础设施失败、超时和人为中断不进有效分母，保留原始证据后安全重试。

最终 Agent 协议固定为 epoch 3/checkpoint-432、上述 12 题各 5 次、reasoning effort high、单次 3600 秒、一个 TP=2 vLLM 实例和两个 runner（总并发 2），基础设施最多重试 3 次。12×5 验证在至少完成 3 个有效完整轮次且累计 30 个有效模型错误时允许提前停止，但必须同时报告观察准确率和 60 格理论上限。

以后对 0809 来源、划分、事实门禁、loss、采样、system prompt、工具协议、tokenizer、训练入口或生成文件的任何修改，都必须在同一变更中更新本 README、`REPRODUCIBILITY.md`、根 README、manifest 和审计报告，重生成全部派生文件并通过两套校验与目标 tokenizer 检查。
