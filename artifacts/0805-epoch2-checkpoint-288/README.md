# 0805 epoch-2 inference checkpoint

This directory is the portable inference bundle for the epoch-2 boundary of the
0805 five-stage Qwen3.6-27B LoRA run. The source checkpoint is `checkpoint-288`
from training run `0805-formal-5epoch-2gpu-20260807T103558+0800`.

The large adapter is stored with Git LFS. Install Git LFS and run `git lfs pull`
after cloning. A checkout containing only the small LFS pointer is not usable.

## Identity and static metrics

- source Git commit: `34fa0dbff027e7ab1241f229042441e7412e1223`
- epoch: `2.0`
- global step: `288`
- eval loss: `0.15015004575252533`
- eval token accuracy: `0.9293184783699688`
- LoRA rank / alpha / dropout: `8 / 32 / 0.05`
- adapter SHA-256: `f39313e8f28160a58fcdd8a35916382f8959e12c335ccb5c26252885647ad06b`

`manifest.json` records the training provenance, hashes, software versions, and
the fixed 12-question x 5-repeat Agent protocol.

## Bundle scope

The bundle is complete for LoRA inference and Agent performance validation. It
intentionally excludes `optimizer.pt`, scheduler state, and RNG state, so it is
not a training-resume checkpoint. The exact Qwen3.6-27B base model is not copied
into Git; its per-file hashes are recorded in
`source_base_model_files_sha256.txt`.

## Verify after cloning

From the repository root:

```bash
git lfs pull --include="artifacts/0805-epoch2-checkpoint-288/adapter_model.safetensors"
python artifacts/0805-epoch2-checkpoint-288/verify_bundle.py \
  --base-model /absolute/path/to/Qwen3.6-27B \
  --repo-root "$PWD"
```

The verifier checks all bundle hashes, the Safetensors index, LoRA parameters,
epoch/global-step identity, static eval metrics, the base-model files, and the
training input files available in the checkout.

## Run the fixed Agent validation

The validation machine needs two compatible GPUs, the exact base model, Git
LFS, Codex CLI, vLLM, and the training runtime versions listed in
`source_environment.txt`. From the repository root:

```bash
BASE_MODEL_PATH=/absolute/path/to/Qwen3.6-27B \
VLLM_ENV=/absolute/path/to/vllm-venv \
PYTHON_BIN=/absolute/path/to/python \
CODEX_BIN=/absolute/path/to/codex \
bash artifacts/0805-epoch2-checkpoint-288/run_agent_validation.sh
```

The launcher reproduces the formal protocol: TP=2, concurrency=2,
`reasoning_effort=high`, Qwen reasoning/tool parsers, raw reasoning capture,
automatic prefix caching, the frozen 12 questions, and five repeats per
question. Results are written to a new timestamped directory under `output/`.

Set `VERIFY_BASE_MODEL=0` only when the base model has already been checked
against `source_base_model_files_sha256.txt`; the adapter and checkpoint
identity checks are never skipped.
