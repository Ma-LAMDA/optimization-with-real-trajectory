# 0809 v7 设计审计输入

本文件是 0809 v7 生成器、两套校验器和正式训练入口共同冻结的仓库内设计输入。v7 延续此前可移植、可达 endpoint 和固定训练预算规则，并落实外部只读报告 `trajectory-analysis/2026-08-11_2002_0809_sft_v6_scheme_continued_audit.md`。外部 sibling 目录只作来源说明，不是生成或训练依赖；本文件完整保存实际执行规则。

## 必须执行的规则

1. manifest 只能冻结逻辑路径 `data/2026-08-09`。生成机绝对路径只可进入 diagnostics，不参与跨主机相等门禁。
2. design input 必须位于本仓库，manifest 的 path 与 SHA-256 必须指向本文件；converter 和 validator 不得读取仓库外报告。
3. 删除推理时不存在的零 loss assistant 控制句。objective 数、真实 evidence path 数和输出措辞唯一数必须分开报告。
4. action 是否仍应调用工具必须从序列化 `messages` 重算，不能信任 converter 写入的 metadata。因果边界是首个 `loss_scale > 0` 的消息；门禁只读取该边界前真实 `tool_response`。
5. 对 action 的同题全部严格 endpoint binding 逐一匹配字面事实。任一 binding 完整时，不得再监督工具调用；必须删除后续正 loss 调用并生成证据特有的停止目标。
6. binding 不完整时，正监督只能包含：已经逐字可见的事实、`matched/expected` 数量、缺失事实要求、待验证假设、以及 1–2 个当前 snapshot 的只读动作。结果返回前不得输出最终答案，也不得宣称闭环、唯一根因或最终标签。
7. validator 必须含结构同义表达的正负 fixture，例如三节点静态路由环、静态标签交换闭环、BPDU Filter、关键路径闭合；“没有形成 IP/MPLS 环路”和条件式待验证假设不得被误杀。
8. q0015/q0025/q0026/q0027/q0028/q0031/q0034/q0035/q0037/q0039/q0060/q0061/q0066 中评审点名的 17 条 action 必须保留为结构未完整的待验证节点。`q0023_path_02_success_09_step_04` 的 1/1 完整 BPDU 事实必须改为停止，原两个后续正工具调用不得进入 action pool。
9. endpoint 的正证据只支持当前标签和当前证据确认集合。没有候选专属负证据时，不得把 `LABEL_NEIGHBORS` 声明为“排除、证伪或不存在”；这些候选必须保持未验证。`rejected_labels` 为空，`hypothesis_elimination_supervised=false`。
10. 真实错误根因排除只来自父级可见 `hypothesis_elimination` 节点及其透明同目标 replay。endpoint 的候选范围校准不得计入排除覆盖。
11. q73–q86 的 inclusive-OR 语义、拓扑 overlap、压缩 `cat` 输出、action 的按题权重/重复和 endpoint 的逐题 tokenizer loss 差异不在本次改动范围；不得借本轮审计擅自改变。
12. 固定 epoch 3 前必须看到全部新目标，并保持每轮 1151 行、有效 batch 8、144 optimizer step、固定 checkpoint-432。
13. 0809 与 0807 的最终 Agent 横向比较必须调用仓库唯一判分入口 `scripts/final_answer_scoring.py`，版本固定为 `agent-final-answer.v4.2026-08-12-incomplete-result-exact-recovery`。正确性只由最终答案决定；过程工具/推理错误只作诊断，不能覆盖精确可接受答案。全篇唯一、完整 fenced `<result>` JSON 字符串列表仅缺 `</result>` 时，只在精确命中可接受答案的情况下恢复；其他 malformed、冲突或多结果输出不恢复。q73–q86 inclusive-OR 由同一 scorer 处理；终态无有效答案计模型错误，基础设施失败、超时和人为中断不进有效分母。

## v7 已实现的结果

- 116 条真实 evidence path 各生成一个 endpoint bundle；按 query 的真实路径数以 `1 / path_count` 归一化。
- endpoint objective 为 evidence summary、positive label boundary、candidate-scope calibration、current-evidence minimal set、decision-ready stop 和最终 `<result>`。116/116 条 rejected-label 声明为空，unsupported exclusion 为 0。
- 308 条 action 全部由消息重算为结构不完整并改写成待验证节点；`matched/expected` 分布归档在 manifest 和 `AUDIT_METRICS.json`。完整 action 为 0。
- 父级完整证据后继续调查的 50 行已改为 stop，共移除 93 个正 loss 工具调用；消息重算后残余完整续查为 0。
- 保留 103 条真实 `hypothesis_elimination`，覆盖 58 题；另保留 8 条透明 exact same-target replay。其余 14 题不补造排除。
- 308 条 action 的最终 `messages` 均不同；每轮仅 8 条声明过的 replay 与来源逐字重复。
- 每轮使用 719 条 base core、308 条 action、8 条 elimination replay 和 116 条 endpoint bundle，共 1151 行；固定 checkpoint-432 前看到全部目标。
- formal config 与 manifest 冻结同一 Agent 协议：12 道冻结题各 5 次、epoch 3/checkpoint-432、reasoning effort high、单次 3600 秒、一个 TP=2 vLLM 实例与两个 runner（总并发 2）、最多 3 次基础设施重试，以及 12×5 的合规错误上限提前停止规则。scorer、韧性 launcher、重试策略、提前停止器和汇总器均进入 tracked dependency 闭环。

## 放行条件

- Windows 官方与独立 validator 通过；
- 独立语义扫描确认完整续查、unsupported endpoint exclusion、未声明重复和未来答案泄漏均为 0；
- 真实 Qwen tokenizer 对语义池、验证和十份逐轮 schedule 的 overlong 与 loss-mask failure 均为 0；
- Linux 异根目录 checkout 下两套 validator 通过；
- 正式入口 runtime identity gate 通过；
- README、复现规范、审计、manifest、release record、脚本和数据保持同一生成闭环。
