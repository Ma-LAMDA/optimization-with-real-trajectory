#!/usr/bin/env bash
set -euo pipefail
WT=/root/autodl-tmp/optimization-with-real-trajectory-nightly-20260805
BASE_OUTPUT=/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly
CONTROL="${BASE_OUTPUT}/control"
RUN_ID="${P1_RUN_ID:?P1_RUN_ID is required}"
RUN_ROOT="${BASE_OUTPUT}/0804/${RUN_ID}"
TRAIN_DIR="${RUN_ROOT}/train"
TRAIN_ENV=/root/autodl-tmp/envs/qwen36-sft
MODEL=/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B
PLUGIN="${CONTROL}/epoch_step_lr_plugin.py"
DATA_ROOT="${WT}/data/2026-08-04"
TRAIN_DATA="${DATA_ROOT}/sft/qwen3_6_27b_reasoning_trajectory_best1_train.jsonl"
VAL_DATA="${DATA_ROOT}/sft/qwen3_6_27b_reasoning_trajectory_best1_validation.jsonl"
mkdir -p "${RUN_ROOT}" "${TRAIN_DIR}"
exec > >(tee -a "${RUN_ROOT}/training_workflow.log") 2>&1
echo "[$(date -Iseconds)] P1 0804 five-epoch training start"
echo "$$" >"${CONTROL}/p1_training.pid"
printf '%s\n' "${RUN_ROOT}" >"${CONTROL}/p1_run_root"
cd "${WT}"
HEAD="$(git rev-parse HEAD)"
if [[ "${HEAD}" != "a4f22e13a10e2b8d32868999f811f53997a18995" ]]; then
  echo "Unexpected nightly worktree commit: ${HEAD}" >&2
  exit 2
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo "GPU compute process appeared before P1 launch; aborting safely." >&2
  exit 3
fi
"${TRAIN_ENV}/bin/python" -B scripts/convert_0804_best_trajectory_reasoning_sft.py
"${TRAIN_ENV}/bin/python" -B scripts/validate_0804_best_trajectory_reasoning_sft.py
"${TRAIN_ENV}/bin/python" -B scripts/check_0804_best1_token_lengths.py \
  --model "${MODEL}" --max-length 16384 --train "${TRAIN_DATA}" --validation "${VAL_DATA}"
sha256sum "${TRAIN_DATA}" "${VAL_DATA}" "${DATA_ROOT}/sft/reasoning_trajectory_best1_manifest.json" >"${RUN_ROOT}/data_sha256.txt"
"${TRAIN_ENV}/bin/python" - <<'PY' >"${RUN_ROOT}/environment.txt"
import importlib.metadata as m
for name in ("ms-swift","torch","transformers","peft","accelerate"):
    try: print(f"{name}={m.version(name)}")
    except m.PackageNotFoundError: print(f"{name}=missing")
PY
git status --short --branch >"${RUN_ROOT}/git_status_before.txt"
git diff >"${RUN_ROOT}/git_diff_before.patch"
git rev-parse HEAD >"${RUN_ROOT}/git_commit.txt"
nvidia-smi -q >"${RUN_ROOT}/nvidia_smi_before.txt"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EPOCH_LR_AUDIT_PATH="${RUN_ROOT}/epoch_lr_audit.jsonl"
"${TRAIN_ENV}/bin/swift" sft \
  --model "${MODEL}" \
  --check_model false \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules all-linear \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --max_length 16384 \
  --truncation_strategy delete \
  --loss_scale default \
  --is_binary_loss_scale false \
  --preserve_thinking true \
  --add_non_thinking_prefix false \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --gradient_checkpointing true \
  --packing false \
  --split_dataset_ratio 0 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --logging_steps 1 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 5 \
  --load_best_model_at_end false \
  --seed 42 \
  --data_seed 42 \
  --report_to none \
  --external_plugins "${PLUGIN}" \
  --callbacks epoch_step_lr \
  --add_version false \
  --output_dir "${TRAIN_DIR}"
"${TRAIN_ENV}/bin/python" "${WT}/scripts/summarize_sft_training.py" \
  --output-dir "${TRAIN_DIR}" --result "${RUN_ROOT}/training_summary.json" --git-commit "${HEAD}"
"${TRAIN_ENV}/bin/python" - "${RUN_ROOT}" <<'PY'
from pathlib import Path
import json,sys,datetime
root=Path(sys.argv[1])
checkpoints=sorted(str(p) for p in (root/"train").glob("checkpoint-*"))
(root/"training_complete.json").write_text(json.dumps({
 "completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 "checkpoints":checkpoints,
 "checkpoint_count":len(checkpoints)
},ensure_ascii=False,indent=2)+"\n")
PY
echo "[$(date -Iseconds)] P1 training complete"
