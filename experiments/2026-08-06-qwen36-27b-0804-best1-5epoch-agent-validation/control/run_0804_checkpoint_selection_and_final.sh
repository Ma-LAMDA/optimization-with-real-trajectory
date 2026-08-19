#!/usr/bin/env bash
set -euo pipefail
WT=/root/autodl-tmp/optimization-with-real-trajectory-nightly-20260805
RUN_ROOT=/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0804/0804-5epoch-20260805T114046Z
CONTROL=/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/control
MODEL_PATH=/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B
VLLM_ENV=/root/autodl-tmp/qwen3.6-27b/.venv
PYTHON=/root/miniconda3/bin/python
CODEX=/usr/local/bin/codex
GIT_COMMIT=aa29f1430
BASE_MODEL=Qwen3.6-27B-base
VLLM_LOG="${RUN_ROOT}/agent_vllm.log"
GPU_CSV="${RUN_ROOT}/agent_gpu_1s.csv"
mkdir -p "${RUN_ROOT}/checkpoint_selection" "${RUN_ROOT}/final_validation"
exec > >(tee -a "${RUN_ROOT}/agent_orchestrator.log") 2>&1
echo "$$" >"${CONTROL}/p1_agent_orchestrator.pid"
VLLM_PID=""
SAMPLER_PID=""
cleanup() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then kill "${SAMPLER_PID}" 2>/dev/null || true; wait "${SAMPLER_PID}" 2>/dev/null || true; fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
if [[ ! -f "${RUN_ROOT}/training_complete.json" ]]; then echo "training incomplete" >&2; exit 2; fi
for step in 40 80 120 160 200; do
  [[ -d "${RUN_ROOT}/train/checkpoint-${step}" ]] || { echo "missing checkpoint-${step}" >&2; exit 2; }
done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; then
  echo "GPU compute process exists before Agent orchestrator" >&2; exit 3
fi
echo "timestamp,gpu_index,memory_used_mib,utilization_pct,temperature_c,power_w" >"${GPU_CSV}"
(
 while true; do
   ts="$(date -Iseconds)"
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits |
     awk -v ts="${ts}" -F',' '{gsub(/^ +| +$/,"",$0); for(i=1;i<=NF;i++) gsub(/^ +| +$/,"",$i); print ts","$1","$2","$3","$4","$5}' >>"${GPU_CSV}"
   sleep 1
 done
) &
SAMPLER_PID=$!
echo "[$(date -Iseconds)] Starting shared TP2 vLLM with five epoch adapters"
env -u VLLM_LOG OMP_NUM_THREADS=1 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0,1 \
 "${VLLM_ENV}/bin/vllm" serve "${MODEL_PATH}" \
 --served-model-name "${BASE_MODEL}" --dtype bfloat16 --host 127.0.0.1 --port 8000 \
 --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.90 \
 --enable-prefix-caching --mamba-cache-mode align \
 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
 --enable-lora --max-loras 5 --max-lora-rank 8 \
 --lora-modules \
 "Qwen3.6-27B-0804-e1=${RUN_ROOT}/train/checkpoint-40" \
 "Qwen3.6-27B-0804-e2=${RUN_ROOT}/train/checkpoint-80" \
 "Qwen3.6-27B-0804-e3=${RUN_ROOT}/train/checkpoint-120" \
 "Qwen3.6-27B-0804-e4=${RUN_ROOT}/train/checkpoint-160" \
 "Qwen3.6-27B-0804-e5=${RUN_ROOT}/train/checkpoint-200" \
 >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
deadline=$((SECONDS+1800))
while true; do
 if ! kill -0 "${VLLM_PID}" 2>/dev/null; then echo "vLLM exited during startup" >&2; tail -100 "${VLLM_LOG}" >&2; exit 4; fi
 if "${PYTHON}" - <<'PY'
import json,urllib.request
want={f"Qwen3.6-27B-0804-e{i}" for i in range(1,6)}
try:
 d=json.load(urllib.request.urlopen("http://127.0.0.1:8000/v1/models",timeout=5))
except Exception: raise SystemExit(1)
got={x.get("id") for x in d.get("data",[])}
raise SystemExit(0 if want<=got else 1)
PY
 then break; fi
 if ((SECONDS>=deadline)); then echo "vLLM readiness timeout" >&2; exit 4; fi
 sleep 5
done
echo "[$(date -Iseconds)] vLLM ready"

validate_summary() {
 local file="$1" expected="$2"
 [[ -f "${file}" ]] || return 1
 "${PYTHON}" - "${file}" "${expected}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));n=int(sys.argv[2]);o=d.get("overall",{})
ok=(d.get("status")=="completed" and o.get("attempts")==n and o.get("attempts_with_captured_reasoning")==n and o.get("reasoning_items",0)>0)
raise SystemExit(0 if ok else 1)
PY
}
run_eval() {
 local phase="$1" model="$2" checkpoint="$3" cases="$4" repeats="$5" out="$6" prefix="$7"
 local expected
 expected="$("${PYTHON}" - "${cases}" "${repeats}" <<'PY'
import sys
print(len(sys.argv[1].split(","))*int(sys.argv[2]))
PY
 )"
 local report="${out}/report"
 if validate_summary "${report}/validation_summary.json" "${expected}"; then
   echo "[$(date -Iseconds)] Reusing completed ${phase}"
   return 0
 fi
 mkdir -p "${out}"
 echo "[$(date -Iseconds)] Start ${phase}: model=${model} cases=${cases} repeats=${repeats}"
 PYTHON_BIN="${PYTHON}" CODEX_BIN="${CODEX}" \
 CODEX_BASE_URL=http://127.0.0.1:8000/v1 \
 MODEL_NAME="${model}" CHECKPOINT="${checkpoint}" GIT_COMMIT="${GIT_COMMIT}" \
 CASE_IDS="${cases}" REPEATS="${repeats}" TIMEOUT_SECONDS=3600 REASONING_EFFORT=high \
 RUN_PREFIX="${prefix}" OUTPUT_ROOT="${out}/runs" REPORT_DIR="${report}" CONTROL_DIR="${out}/control" \
 BASELINE_SUMMARY= DATASET="${WT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/train_0629.jsonl" \
 TEMPLATE="${WT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/IP user prompt with saved configs skills.txt" \
 CODEX_MODEL_CATALOG_TEMPLATE="${WT}/config/codex_qwen_model_catalog.json" \
 INFRA_MAX_RETRIES=3 bash "${WT}/scripts/run_agent_validation_resilient.sh"
 validate_summary "${report}/validation_summary.json" "${expected}"
 echo "[$(date -Iseconds)] Complete ${phase}"
}
SELECTION_CASES=12,20,38,71,86,100
for epoch in 1 2 3 4 5; do
 step=$((epoch*40))
 run_eval "checkpoint-selection-epoch-${epoch}" "Qwen3.6-27B-0804-e${epoch}" \
   "${RUN_ROOT}/train/checkpoint-${step}" "${SELECTION_CASES}" 2 \
   "${RUN_ROOT}/checkpoint_selection/epoch-${epoch}" "selection-e${epoch}"
done
"${PYTHON}" "${CONTROL}/select_0804_checkpoint.py" "${RUN_ROOT}"
SELECTED_EPOCH="$(cat "${RUN_ROOT}/checkpoint_selection/selected_epoch.txt")"
SELECTED_MODEL="$(cat "${RUN_ROOT}/checkpoint_selection/selected_model.txt")"
SELECTED_CHECKPOINT="$(cat "${RUN_ROOT}/checkpoint_selection/selected_checkpoint.txt")"
echo "[$(date -Iseconds)] Selected epoch ${SELECTED_EPOCH}: ${SELECTED_CHECKPOINT}"
run_eval "final-selected-extra" "${SELECTED_MODEL}" "${SELECTED_CHECKPOINT}" \
  "${SELECTION_CASES}" 3 "${RUN_ROOT}/final_validation/selected_extra" "final-selected-extra"
run_eval "final-nonselection" "${SELECTED_MODEL}" "${SELECTED_CHECKPOINT}" \
  2,19,29,65,85,99 5 "${RUN_ROOT}/final_validation/nonselection" "final-nonselection"
"${PYTHON}" "${CONTROL}/compose_0804_final.py" "${RUN_ROOT}"
cleanup
VLLM_PID="";SAMPLER_PID="";trap - EXIT INT TERM
"${PYTHON}" - "${RUN_ROOT}" <<'PY'
from pathlib import Path
import json,sys,datetime
root=Path(sys.argv[1]);d=json.load(open(root/"final_validation"/"validation_summary.json"))
(root/"p1_complete.json").write_text(json.dumps({"completed_at":datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),"overall":d["overall"],"selected_checkpoint":d["selected_checkpoint"]},ensure_ascii=False,indent=2)+"\n")
PY
echo "[$(date -Iseconds)] P1 0804 checkpoint selection and final validation complete"
