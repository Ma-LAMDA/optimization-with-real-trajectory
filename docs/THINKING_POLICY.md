# Thinking 强制策略

自 2026-08-04 起，本项目的能力评测和对比实验必须显式开启 Qwen 的 thinking。
该规则适用于 Base、LoRA、checkpoint sweep、部署 A/B、留出集验证和任何据以形成
模型能力结论的完整 Codex Agent 运行。

## 强制配置

- 统一通过 `scripts/run_agent_validation.sh` 发起；该控制器默认并强制使用
  `REASONING_EFFORT=high`，拒绝 `none`。
- 控制器将 `model_reasoning_effort` 显式传给 Codex CLI；Responses API 的 vLLM
  适配层据此向 Qwen chat template 传入 `enable_thinking=true`。
- 启动日志必须包含 `thinking=enabled` 和实际 `reasoning_effort`。
- `validation_summary.json`、`report.md` 和 `attempts.csv` 必须记录 thinking 请求状态，
  并统计每次运行的 `reasoning_output_tokens` 与有可观测 reasoning 输出的运行次数。

## 报告与可比性

- 若一次评测的已完成运行均为 `reasoning_output_tokens=0`，报告必须明确标为
  “thinking 已请求但未观察到可见 reasoning 输出”；不得把它与已观察到 thinking
  输出的实验混为同一条件。
- Base/LoRA A/B 只有在题集、prompt、工具、时限、拓扑、采样设置和 thinking 配置
  一致时，才能做能力差异结论。
- 原始事件流可以保留在运行机上用于审计；提交仓库时只归档紧凑的统计产物，不提交
  大体积原始 thinking 文本。

## 历史结果

该策略不追溯改写已经开始或完成的运行。特别是
`base-eval-tp2x1-concurrency2-20260731T174703Z` 在本策略落地前已经启动，必须保持
同一口径完成；其报告将标明没有观察到可见 thinking 输出，不能与后续 thinking-on
实验直接合并。
