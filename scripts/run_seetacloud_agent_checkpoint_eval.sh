#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BRANCH="${BRANCH:-2026-07-31-sft}"
SYNC_REPO="${SYNC_REPO:-1}"
VLLM_ENV="${VLLM_ENV:-/root/autodl-tmp/qwen3.6-27b/.venv}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/output/qwen36-27b-lora-0731-step760-plus200-v2/train/v0-20260731-203846/checkpoint-100}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-trained}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3.6-27B-base}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
RUN_PREFIX="${RUN_PREFIX:-agent-ab-step760-plus100-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/qwen-codex-eval}"
REPORT_DIR="${REPORT_DIR:-${OUTPUT_ROOT}/${RUN_PREFIX}-report}"
CASE_IDS="${CASE_IDS:-4,5,20,89}"
REPEATS="${REPEATS:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
BASELINE_SUMMARY="${BASELINE_SUMMARY-${REPO_ROOT}/experiments/2026-07-31-qwen36-27b-base-eval/deployment-ab/summary.json}"
VLLM_LOG="${VLLM_LOG:-${OUTPUT_ROOT}/${RUN_PREFIX}-vllm.log}"
VLLM_BIN="${VLLM_ENV}/bin/vllm"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

cd "${REPO_ROOT}"
if [[ "${SYNC_REPO}" == "1" ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked repository changes exist; refusing to update the branch." >&2
    exit 1
  fi
  git fetch origin "${BRANCH}"
  git switch "${BRANCH}"
  git pull --ff-only origin "${BRANCH}"
fi
GIT_COMMIT="$(git rev-parse HEAD)"

for path in "${VLLM_BIN}" "${PYTHON_BIN}" "${MODEL_PATH}" "${CHECKPOINT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 1
  fi
done
if [[ -n "${BASELINE_SUMMARY}" && ! -e "${BASELINE_SUMMARY}" ]]; then
  echo "Required baseline summary is missing: ${BASELINE_SUMMARY}" >&2
  exit 1
fi
ACTIVE_GPU_PROCESSES="$(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${ACTIVE_GPU_PROCESSES}" ]]; then
  echo "GPU compute processes already exist; refusing to start Agent evaluation:" >&2
  echo "${ACTIVE_GPU_PROCESSES}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}"
VLLM_PID=""
cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT INT TERM

echo "[$(date -Iseconds)] Starting one TP=2 vLLM instance for ${CHECKPOINT}"
env -u VLLM_LOG \
  OMP_NUM_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0,1 \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${BASE_MODEL_NAME}" \
  --dtype bfloat16 \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size 2 \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 8 \
  --lora-modules "${MODEL_NAME}=${CHECKPOINT}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

READY_DEADLINE=$((SECONDS + 1800))
MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
while true; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  if "${PYTHON_BIN}" - "${MODELS_URL}" "${MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
except Exception:
    raise SystemExit(1)
names = {item.get("id") for item in payload.get("data", [])}
raise SystemExit(0 if sys.argv[2] in names else 1)
PY
  then
    break
  fi
  if (( SECONDS >= READY_DEADLINE )); then
    echo "vLLM did not become ready within 30 minutes." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  sleep 5
done

echo "[$(date -Iseconds)] Running full Agent A/B candidate with two concurrent workers"
PYTHON_BIN="${PYTHON_BIN}" \
CODEX_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1" \
MODEL_NAME="${MODEL_NAME}" \
CHECKPOINT="${CHECKPOINT}" \
GIT_COMMIT="${GIT_COMMIT}" \
CASE_IDS="${CASE_IDS}" \
REPEATS="${REPEATS}" \
TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
REASONING_EFFORT="${REASONING_EFFORT}" \
RUN_PREFIX="${RUN_PREFIX}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
REPORT_DIR="${REPORT_DIR}" \
BASELINE_SUMMARY="${BASELINE_SUMMARY}" \
  bash "${SCRIPT_DIR}/run_agent_validation.sh"

cleanup_vllm
VLLM_PID=""
trap - EXIT INT TERM
echo "[$(date -Iseconds)] Agent A/B complete: ${REPORT_DIR}/validation_summary.json"
