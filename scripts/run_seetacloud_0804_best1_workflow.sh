#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BRANCH="${BRANCH:-2026-07-31-sft}"
SYNC_REPO="${SYNC_REPO:-1}"
TRAIN_ENV="${TRAIN_ENV:-/root/autodl-tmp/envs/qwen36-sft}"
VLLM_ENV="${VLLM_ENV:-/root/autodl-tmp/qwen3.6-27b/.venv}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-08-04}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/sft/reasoning_trajectory_best1_manifest.json}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/output/qwen36-27b-lora-0804-best1-${RUN_ID}}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${RUN_ROOT}/train}"
TRAINING_SUMMARY="${TRAINING_SUMMARY:-${RUN_ROOT}/training_summary.json}"
AGENT_OUTPUT_ROOT="${AGENT_OUTPUT_ROOT:-${RUN_ROOT}/agent_runs}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/validation_eval}"
WORKFLOW_SUMMARY="${WORKFLOW_SUMMARY:-${RUN_ROOT}/workflow_summary.json}"
WORKFLOW_LOG="${WORKFLOW_LOG:-${RUN_ROOT}/workflow.log}"
VLLM_LOG="${VLLM_LOG:-${RUN_ROOT}/vllm.log}"

TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_ENV}/bin/python}"
TRAIN_SWIFT="${TRAIN_SWIFT:-${TRAIN_ENV}/bin/swift}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV}/bin/vllm}"
CODEX_BIN="${CODEX_BIN:-/usr/local/bin/codex}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
EVAL_STEPS="${EVAL_STEPS:-40}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
VALIDATION_REPEATS="${VALIDATION_REPEATS:-5}"
AGENT_TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS:-3600}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.6-27B-0804-best1}"
BASE_SERVED_MODEL_NAME="${BASE_SERVED_MODEL_NAME:-Qwen3.6-27B-base}"

AGENT_DATASET="${AGENT_DATASET:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/train_0629.jsonl}"
AGENT_TEMPLATE="${AGENT_TEMPLATE:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/IP user prompt with saved configs skills.txt}"

if [[ ! "${VALIDATION_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "VALIDATION_REPEATS must be a positive integer." >&2
  exit 1
fi
if [[ ! "${AGENT_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AGENT_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 1
fi
if [[ ! "${REASONING_EFFORT}" =~ ^(minimal|low|medium|high|xhigh|max)$ ]]; then
  echo "REASONING_EFFORT must explicitly enable thinking." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
touch "${WORKFLOW_LOG}"
exec > >(tee -a "${WORKFLOW_LOG}") 2>&1

echo "[$(date -Iseconds)] Starting 0804 best1 end-to-end workflow ${RUN_ID}"
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
CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "${BRANCH}" ]]; then
  echo "Expected branch ${BRANCH}, found ${CURRENT_BRANCH}." >&2
  exit 1
fi

for path in \
  "${TRAIN_PYTHON}" \
  "${TRAIN_SWIFT}" \
  "${VLLM_PYTHON}" \
  "${VLLM_BIN}" \
  "${CODEX_BIN}" \
  "${MODEL_PATH}" \
  "${AGENT_DATASET}" \
  "${AGENT_TEMPLATE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 1
  fi
done

ACTIVE_GPU_PROCESSES="$(
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${ACTIVE_GPU_PROCESSES}" ]]; then
  echo "GPU compute processes already exist; refusing to start 0804 workflow:" >&2
  echo "${ACTIVE_GPU_PROCESSES}" >&2
  exit 1
fi

echo "[$(date -Iseconds)] Source commit: ${GIT_COMMIT}"
"${TRAIN_PYTHON}" - <<'PY'
import importlib.metadata as metadata
for name in ("ms-swift", "torch", "transformers", "peft"):
    try:
        print(f"{name}={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=missing")
PY
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

mkdir -p "${TRAIN_OUTPUT_DIR}"
printf '%s\n' "${GIT_COMMIT}" >"${TRAIN_OUTPUT_DIR}/training_git_commit.txt"
sha256sum "${MANIFEST_PATH}" | awk '{print $1}' \
  >"${TRAIN_OUTPUT_DIR}/training_manifest_sha256.txt"

echo "[$(date -Iseconds)] Starting 0804 best1 LoRA SFT"
MODEL_PATH="${MODEL_PATH}" \
DATA_ROOT="${DATA_ROOT}" \
OUTPUT_DIR="${TRAIN_OUTPUT_DIR}" \
PYTHON_BIN="${TRAIN_PYTHON}" \
SWIFT_BIN="${TRAIN_SWIFT}" \
CUDA_VISIBLE_DEVICES=0 \
MAX_LENGTH="${MAX_LENGTH}" \
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
EVAL_STEPS="${EVAL_STEPS}" \
LEARNING_RATE="${LEARNING_RATE}" \
  bash scripts/train_qwen36_0804_best1_quick.sh

"${TRAIN_PYTHON}" scripts/summarize_sft_training.py \
  --output-dir "${TRAIN_OUTPUT_DIR}" \
  --result "${TRAINING_SUMMARY}" \
  --git-commit "${GIT_COMMIT}"
BEST_CHECKPOINT="$(
  "${TRAIN_PYTHON}" scripts/summarize_sft_training.py \
    --output-dir "${TRAIN_OUTPUT_DIR}" \
    --result "${TRAINING_SUMMARY}" \
    --git-commit "${GIT_COMMIT}" \
    --print-best-only
)"
echo "[$(date -Iseconds)] Minimum-eval-loss checkpoint: ${BEST_CHECKPOINT}"

VLLM_PID=""
cleanup_vllm() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[$(date -Iseconds)] Stopping vLLM ${VLLM_PID}"
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT INT TERM

echo "[$(date -Iseconds)] Starting one TP=2 vLLM instance with selected LoRA"
VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0,1 \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 8 \
  --lora-modules "${SERVED_MODEL_NAME}=${BEST_CHECKPOINT}" \
  --served-model-name "${BASE_SERVED_MODEL_NAME}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

MODELS_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
READY_DEADLINE=$((SECONDS + 1800))
while true; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready." >&2
    tail -100 "${VLLM_LOG}" >&2
    exit 1
  fi
  if "${VLLM_PYTHON}" - "${MODELS_URL}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        names = {item.get("id") for item in json.load(response).get("data", [])}
except Exception:
    raise SystemExit(1)
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

AGENT_VALIDATION_CASE_IDS="$(
  "${TRAIN_PYTHON}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
case_ids = json.load(open(sys.argv[1], encoding="utf-8"))["split"]["validation_case_ids"]
print(",".join(str(case_id) for case_id in case_ids))
PY
)"
echo "[$(date -Iseconds)] Running thinking=${REASONING_EFFORT} Codex Agent validation: cases=${AGENT_VALIDATION_CASE_IDS}, repeats=${VALIDATION_REPEATS}"

PYTHON_BIN="${TRAIN_PYTHON}" \
CODEX_BIN="${CODEX_BIN}" \
REASONING_EFFORT="${REASONING_EFFORT}" \
CODEX_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1" \
MODEL_NAME="${SERVED_MODEL_NAME}" \
CHECKPOINT="${BEST_CHECKPOINT}" \
GIT_COMMIT="${GIT_COMMIT}" \
CASE_IDS="${AGENT_VALIDATION_CASE_IDS}" \
REPEATS="${VALIDATION_REPEATS}" \
TIMEOUT_SECONDS="${AGENT_TIMEOUT_SECONDS}" \
RUN_PREFIX="agent-validation-${RUN_ID}" \
OUTPUT_ROOT="${AGENT_OUTPUT_ROOT}" \
REPORT_DIR="${EVAL_OUTPUT_DIR}" \
DATASET="${AGENT_DATASET}" \
TEMPLATE="${AGENT_TEMPLATE}" \
BASELINE_SUMMARY="" \
  bash scripts/run_agent_validation.sh

cleanup_vllm
VLLM_PID=""
trap - EXIT INT TERM

"${TRAIN_PYTHON}" scripts/finalize_lora_workflow.py \
  --training-summary "${TRAINING_SUMMARY}" \
  --validation-summary "${EVAL_OUTPUT_DIR}/validation_summary.json" \
  --manifest "${MANIFEST_PATH}" \
  --output "${WORKFLOW_SUMMARY}" \
  --run-id "${RUN_ID}" \
  --branch "${BRANCH}" \
  --git-commit "${GIT_COMMIT}" \
  --model-path "${MODEL_PATH}"

echo "[$(date -Iseconds)] 0804 workflow completed: ${WORKFLOW_SUMMARY}"
