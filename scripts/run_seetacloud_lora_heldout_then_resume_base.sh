#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/qwen-codex-eval}"
BASE_RUN_PREFIX="${BASE_RUN_PREFIX:?BASE_RUN_PREFIX is required}"
BASE_LAUNCHER_PID="${BASE_LAUNCHER_PID:?BASE_LAUNCHER_PID is required}"
BASE_CONTROLLER_PID="${BASE_CONTROLLER_PID:?BASE_CONTROLLER_PID is required}"
ACTIVE_RUN_PIDS="${ACTIVE_RUN_PIDS:?ACTIVE_RUN_PIDS is required}"
EXPECTED_BASE_COMPLETED="${EXPECTED_BASE_COMPLETED:-58}"
LORA_RUN_PREFIX="${LORA_RUN_PREFIX:?LORA_RUN_PREFIX is required}"
LORA_CASE_IDS="${LORA_CASE_IDS:-12,24,40,72,86,100}"
LORA_REPEATS="${LORA_REPEATS:-5}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-${REPO_ROOT}/output/qwen36-27b-lora-0731-step760-plus200-v2/train/v0-20260731-203846/checkpoint-100}"
POLL_SECONDS="${POLL_SECONDS:-10}"
LOG="${LOG:-${OUTPUT_ROOT}/${LORA_RUN_PREFIX}-pause-lora-resume.log}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

count_base_completed() {
  find "${OUTPUT_ROOT}" -maxdepth 2 \
    -path "*/${BASE_RUN_PREFIX}-q*/.runner_exit_code" -type f | wc -l
}

wait_for_pid_exit() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
}

wait_for_gpu_free() {
  local deadline=$((SECONDS + 600))
  while true; do
    local active
    active="$(
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null || true
    )"
    if [[ -z "${active}" ]]; then
      return
    fi
    if (( SECONDS >= deadline )); then
      echo "GPU processes did not exit before the 10-minute deadline:" >&2
      echo "${active}" >&2
      exit 1
    fi
    sleep 5
  done
}

echo "[$(date -Iseconds)] waiting for the frozen Base pair: ${ACTIVE_RUN_PIDS}"
IFS=',' read -r -a active_pids <<<"${ACTIVE_RUN_PIDS}"
for pid in "${active_pids[@]}"; do
  if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid active run PID: ${pid}" >&2
    exit 1
  fi
  wait_for_pid_exit "${pid}"
done

completed="$(count_base_completed)"
if (( completed < EXPECTED_BASE_COMPLETED )); then
  echo "Expected at least ${EXPECTED_BASE_COMPLETED} completed Base attempts, found ${completed}" >&2
  exit 1
fi
echo "[$(date -Iseconds)] frozen Base pair ended; completed=${completed}"

if kill -0 "${BASE_CONTROLLER_PID}" 2>/dev/null; then
  echo "[$(date -Iseconds)] terminating the stopped Base controller without starting another pair"
  kill -TERM "${BASE_CONTROLLER_PID}" 2>/dev/null || true
  kill -CONT "${BASE_CONTROLLER_PID}" 2>/dev/null || true
  wait_for_pid_exit "${BASE_CONTROLLER_PID}"
fi
if kill -0 "${BASE_LAUNCHER_PID}" 2>/dev/null; then
  wait_for_pid_exit "${BASE_LAUNCHER_PID}"
fi
wait_for_gpu_free
echo "[$(date -Iseconds)] Base is paused and its vLLM service is down"

LORA_REPORT_DIR="${OUTPUT_ROOT}/${LORA_RUN_PREFIX}-report"
if [[ ! -f "${LORA_REPORT_DIR}/validation_summary.json" ]]; then
  echo "[$(date -Iseconds)] starting LoRA heldout full Agent evaluation"
  cd "${REPO_ROOT}"
  CHECKPOINT="${LORA_CHECKPOINT}" \
  CASE_IDS="${LORA_CASE_IDS}" \
  REPEATS="${LORA_REPEATS}" \
  RUN_PREFIX="${LORA_RUN_PREFIX}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  REPORT_DIR="${LORA_REPORT_DIR}" \
  BASELINE_SUMMARY="" \
    bash "${SCRIPT_DIR}/run_seetacloud_agent_checkpoint_eval.sh"
else
  echo "[$(date -Iseconds)] LoRA heldout summary already exists; skipping LoRA rerun"
fi
if [[ ! -f "${LORA_REPORT_DIR}/validation_summary.json" ]]; then
  echo "LoRA heldout evaluation did not produce a final summary" >&2
  exit 1
fi

BASE_REPORT_DIR="${OUTPUT_ROOT}/${BASE_RUN_PREFIX}-report"
if [[ ! -f "${BASE_REPORT_DIR}/validation_summary.json" ]]; then
  echo "[$(date -Iseconds)] LoRA evaluation complete; resuming Base with the original prefix"
  cd "${REPO_ROOT}"
  RUN_PREFIX="${BASE_RUN_PREFIX}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  REPORT_DIR="${BASE_REPORT_DIR}" \
    bash "${SCRIPT_DIR}/run_seetacloud_base_agent_eval.sh"
else
  echo "[$(date -Iseconds)] Base summary already exists; skipping Base resume"
fi

echo "[$(date -Iseconds)] Base complete; running heldout timeout replacements and composing full500"
PRIMARY_PREFIX="${BASE_RUN_PREFIX}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash "${SCRIPT_DIR}/run_seetacloud_base_full500_followup.sh"
echo "[$(date -Iseconds)] pause-LoRA-resume workflow complete"
