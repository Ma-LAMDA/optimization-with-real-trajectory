#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to the exact Qwen3.6-27B directory}"
VLLM_ENV="${VLLM_ENV:?Set VLLM_ENV to the vLLM virtual environment}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CODEX_BIN="${CODEX_BIN:-codex}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VERIFY_BASE_MODEL="${VERIFY_BASE_MODEL:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-0805-e2}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen3.6-27B-base}"
CASE_IDS="${CASE_IDS:-2,12,19,20,29,38,65,71,85,86,99,100}"
REPEATS="${REPEATS:-5}"
RUN_PREFIX="${RUN_PREFIX:-0805-e2-portable}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PHASE_ROOT="${PHASE_ROOT:-${REPO_ROOT}/output/${RUN_PREFIX}-${TIMESTAMP}}"
OUTPUT_ROOT="${PHASE_ROOT}/runs"
REPORT_DIR="${PHASE_ROOT}/report"
CONTROL_DIR="${PHASE_ROOT}/control"
VLLM_LOG="${PHASE_ROOT}/vllm.log"
DATASET="${DATASET:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/train_0629.jsonl}"
TEMPLATE="${TEMPLATE:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/IP user prompt with saved configs skills.txt}"
RUNNER="${RUNNER:-${REPO_ROOT}/scripts/run_agent_validation_resilient.sh}"
MODEL_CATALOG="${MODEL_CATALOG:-${REPO_ROOT}/config/codex_qwen_model_catalog.json}"
SOURCE_COMMIT="34fa0dbff027e7ab1241f229042441e7412e1223"

for path in "${CHECKPOINT_DIR}" "${BASE_MODEL_PATH}" "${VLLM_ENV}/bin/vllm" "${DATASET}" "${TEMPLATE}" "${RUNNER}" "${MODEL_CATALOG}"; do
  [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
done
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Python not found: ${PYTHON_BIN}" >&2; exit 2; }
command -v "${CODEX_BIN}" >/dev/null 2>&1 || { echo "Codex CLI not found: ${CODEX_BIN}" >&2; exit 2; }

verify_args=("${SCRIPT_DIR}/verify_bundle.py" --repo-root "${REPO_ROOT}")
if [[ "${VERIFY_BASE_MODEL}" == "1" ]]; then
  verify_args+=(--base-model "${BASE_MODEL_PATH}")
fi
"${PYTHON_BIN}" "${verify_args[@]}"

active_gpu="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
[[ -z "${active_gpu}" ]] || { echo "GPU compute processes already exist: ${active_gpu}" >&2; exit 3; }

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}" "${CONTROL_DIR}"
VLLM_PID=""
cleanup() {
  local rc=$?
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  printf '%s\n' "${rc}" >"${CONTROL_DIR}/launcher.exit_code"
  exit "${rc}"
}
trap cleanup EXIT INT TERM

echo "[$(date -Iseconds)] Starting TP=2 vLLM with ${MODEL_NAME}"
env -u VLLM_LOG OMP_NUM_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${VLLM_ENV}/bin/vllm" serve "${BASE_MODEL_PATH}" \
  --served-model-name "${BASE_MODEL_NAME}" --dtype bfloat16 --host 127.0.0.1 --port "${PORT}" \
  --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --mamba-cache-mode align \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-lora --max-loras 2 --max-lora-rank 8 \
  --lora-modules "${MODEL_NAME}=${CHECKPOINT_DIR}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

deadline=$((SECONDS + 1800))
while true; do
  kill -0 "${VLLM_PID}" 2>/dev/null || { tail -100 "${VLLM_LOG}" >&2; exit 4; }
  if "${PYTHON_BIN}" - "${MODEL_NAME}" "${PORT}" <<'PY'
import json, sys, urllib.request
try:
    data = json.load(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[2]}/v1/models", timeout=5))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if sys.argv[1] in {item.get("id") for item in data.get("data", [])} else 1)
PY
  then
    break
  fi
  (( SECONDS < deadline )) || { echo "vLLM readiness timeout" >&2; exit 4; }
  sleep 5
done

PYTHON_BIN="${PYTHON_BIN}" CODEX_BIN="${CODEX_BIN}" \
CODEX_BASE_URL="http://127.0.0.1:${PORT}/v1" \
MODEL_NAME="${MODEL_NAME}" CHECKPOINT="${CHECKPOINT_DIR}" GIT_COMMIT="${SOURCE_COMMIT}" \
CASE_IDS="${CASE_IDS}" REPEATS="${REPEATS}" TIMEOUT_SECONDS=3600 REASONING_EFFORT=high \
RUN_PREFIX="${RUN_PREFIX}" OUTPUT_ROOT="${OUTPUT_ROOT}" REPORT_DIR="${REPORT_DIR}" CONTROL_DIR="${CONTROL_DIR}/agent" \
BASELINE_SUMMARY= DATASET="${DATASET}" TEMPLATE="${TEMPLATE}" \
CODEX_MODEL_CATALOG_TEMPLATE="${MODEL_CATALOG}" INFRA_MAX_RETRIES=3 \
bash "${RUNNER}"

"${PYTHON_BIN}" - "${REPORT_DIR}/validation_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
overall = d.get("overall", {})
if d.get("status") != "completed" or overall.get("attempts") != 60:
    raise SystemExit(f"incomplete validation summary: {d.get('status')} {overall}")
if overall.get("attempts_with_captured_reasoning") != 60 or overall.get("reasoning_items", 0) <= 0:
    raise SystemExit("reasoning capture is incomplete")
print(json.dumps({"status": "completed", "overall": overall}, ensure_ascii=False, indent=2))
PY

echo "[$(date -Iseconds)] Epoch-2 validation complete: ${REPORT_DIR}/validation_summary.json"
