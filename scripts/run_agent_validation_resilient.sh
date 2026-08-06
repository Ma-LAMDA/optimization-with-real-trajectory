#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CODEX_BIN="${CODEX_BIN:-/usr/local/bin/codex}"
CODEX_BASE_URL="${CODEX_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen3.6-27B-trained}"
CHECKPOINT="${CHECKPOINT:-}"
GIT_COMMIT="${GIT_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"
CASE_IDS="${CASE_IDS:-12,24,40,72,86,100}"
REPEATS="${REPEATS:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
RUN_PREFIX="${RUN_PREFIX:-qwen36-agent-validation-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/qwen-codex-eval}"
REPORT_DIR="${REPORT_DIR:-${OUTPUT_ROOT}/${RUN_PREFIX}-report}"
CONTROL_DIR="${CONTROL_DIR:-${OUTPUT_ROOT}/${RUN_PREFIX}-control}"
RUNNER="${RUNNER:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/scripts/run_codex_ip_trajectories.py}"
DATASET="${DATASET:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/train_0629.jsonl}"
TEMPLATE="${TEMPLATE:-${REPO_ROOT}/experiments/2026-07-27-ip_codex_train0629_14x10/inputs/IP user prompt with saved configs skills.txt}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"
CODEX_MODEL_CATALOG_TEMPLATE="${CODEX_MODEL_CATALOG_TEMPLATE:-${REPO_ROOT}/config/codex_qwen_model_catalog.json}"
CODEX_MODEL_METADATA_SOURCE="${CODEX_MODEL_METADATA_SOURCE:-Qwen3.6-27B-trained}"
MODEL_METADATA_SMOKE_ONLY="${MODEL_METADATA_SMOKE_ONLY:-0}"
MODEL_METADATA_SMOKE_TIMEOUT_SECONDS="${MODEL_METADATA_SMOKE_TIMEOUT_SECONDS:-300}"
# Agent capability evaluations and A/B experiments must request visible
# reasoning.  `none` is intentionally rejected below: it causes vLLM's Qwen
# chat template to close the <think> block before generation.
REASONING_EFFORT="${REASONING_EFFORT:-high}"
INFRA_MAX_RETRIES="${INFRA_MAX_RETRIES:-3}"

if [[ ! "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer." >&2
  exit 1
fi
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TIMEOUT_SECONDS must be a positive integer." >&2
  exit 1
fi
if [[ ! "${MODEL_METADATA_SMOKE_ONLY}" =~ ^[01]$ ]]; then
  echo "MODEL_METADATA_SMOKE_ONLY must be 0 or 1." >&2
  exit 1
fi
if [[ ! "${MODEL_METADATA_SMOKE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODEL_METADATA_SMOKE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 1
fi
if [[ ! "${REASONING_EFFORT}" =~ ^(minimal|low|medium|high|xhigh|max)$ ]]; then
  echo "REASONING_EFFORT must request thinking (minimal, low, medium, high, xhigh, or max); 'none' is not permitted." >&2
  exit 1
fi
if [[ ! "${INFRA_MAX_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "INFRA_MAX_RETRIES must be a non-negative integer." >&2
  exit 1
fi
for path in \
  "${PYTHON_BIN}" \
  "${CODEX_BIN}" \
  "${RUNNER}" \
  "${DATASET}" \
  "${TEMPLATE}" \
  "${CODEX_MODEL_CATALOG_TEMPLATE}" \
  "${SCRIPT_DIR}/prepare_codex_model_catalog.py" \
  "${SCRIPT_DIR}/enrich_codex_events_with_reasoning.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path is missing: ${path}" >&2
    exit 1
  fi
done

read -r -a CASES <<<"${CASE_IDS//,/ }"
if (( ${#CASES[@]} == 0 )); then
  echo "CASE_IDS must contain at least one case." >&2
  exit 1
fi
for case_id in "${CASES[@]}"; do
  if [[ ! "${case_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid case id: ${case_id}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}" "${CONTROL_DIR}/logs"
LOG="${CONTROL_DIR}/controller.log"
CODEX_WRAPPER="${CONTROL_DIR}/codex-agent-validation"
CODEX_MODEL_CATALOG="${CONTROL_DIR}/model_catalog.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_codex_model_catalog.py" \
  --template "${CODEX_MODEL_CATALOG_TEMPLATE}" \
  --output "${CODEX_MODEL_CATALOG}" \
  --model "${MODEL_NAME}" \
  --metadata-source "${CODEX_MODEL_METADATA_SOURCE}"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  "exec \"${CODEX_BIN}\" -c 'model_providers.qwen_local.base_url=\"${CODEX_BASE_URL}\"' -c 'model_reasoning_effort=\"${REASONING_EFFORT}\"' -c 'hide_agent_reasoning=false' -c 'show_raw_agent_reasoning=true' -c 'model_catalog_json=\"${CODEX_MODEL_CATALOG}\"' \"\$@\"" \
  >"${CODEX_WRAPPER}"
chmod 700 "${CODEX_WRAPPER}"

if [[ "${MODEL_METADATA_SMOKE_ONLY}" == "1" ]]; then
  SMOKE_EVENTS="${CONTROL_DIR}/metadata_smoke_events.jsonl"
  SMOKE_STDERR="${CONTROL_DIR}/metadata_smoke_stderr.log"
  set +e
  printf '%s\n' \
    'Reason carefully without tools: which integer is larger, 17*19 or 18*18? Reply with exactly the larger integer.' | \
    timeout "${MODEL_METADATA_SMOKE_TIMEOUT_SECONDS}" \
      "${CODEX_WRAPPER}" exec \
        --json \
        --sandbox read-only \
        --skip-git-repo-check \
        --model "${MODEL_NAME}" \
        - \
      >"${SMOKE_EVENTS}" 2>"${SMOKE_STDERR}"
  smoke_rc=$?
  set -e
  if grep -Fq 'Defaulting to fallback metadata' "${SMOKE_EVENTS}" "${SMOKE_STDERR}"; then
    echo "Codex model metadata smoke failed: fallback metadata was used." >&2
    exit 1
  fi
  if (( smoke_rc != 0 )); then
    echo "Codex model metadata smoke failed with exit code ${smoke_rc}." >&2
    tail -50 "${SMOKE_EVENTS}" >&2 || true
    cat "${SMOKE_STDERR}" >&2 || true
    exit "${smoke_rc}"
  fi
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/enrich_codex_events_with_reasoning.py" \
    --events "${SMOKE_EVENTS}" \
    --require-reasoning; then
    echo "Codex model metadata smoke failed: raw reasoning enrichment failed." >&2
    exit 1
  fi
  if ! grep -Fq '"type":"turn.completed"' "${SMOKE_EVENTS}"; then
    echo "Codex model metadata smoke failed: no completed turn was recorded." >&2
    tail -50 "${SMOKE_EVENTS}" >&2 || true
    exit 1
  fi
  if ! "${PYTHON_BIN}" - "${SMOKE_EVENTS}" <<'PY'
import json
import sys

events_path = sys.argv[1]
with open(events_path, encoding="utf-8") as handle:
    events = [json.loads(line) for line in handle if line.strip()]
has_reasoning = any(
    event.get("type") == "item.completed"
    and isinstance(event.get("item"), dict)
    and event["item"].get("type") == "reasoning"
    and isinstance(event["item"].get("text"), str)
    and event["item"]["text"].strip()
    for event in events
)
raise SystemExit(0 if has_reasoning else 1)
PY
  then
    echo "Codex model metadata smoke failed: no non-empty raw reasoning item was recorded." >&2
    tail -50 "${SMOKE_EVENTS}" >&2 || true
    exit 1
  fi
  echo "Codex model metadata smoke passed: model=${MODEL_NAME} catalog=${CODEX_MODEL_CATALOG}"
  exit 0
fi

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${LOG}"
}

terminate_descendants() {
  local parent_pid="$1"
  local children
  local child_pid
  children="$(pgrep -P "${parent_pid}" 2>/dev/null || true)"
  for child_pid in ${children}; do
    terminate_descendants "${child_pid}"
  done
  for child_pid in ${children}; do
    kill -TERM "${child_pid}" 2>/dev/null || true
  done
}

cleanup_children() {
  local child_pid
  while read -r child_pid; do
    [[ -n "${child_pid}" ]] || continue
    terminate_descendants "${child_pid}"
    kill -TERM "${child_pid}" 2>/dev/null || true
  done < <(jobs -pr)
}
trap cleanup_children EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

manifest_status() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
try:
    status = json.load(open(sys.argv[1], encoding="utf-8")).get("status")
except Exception:
    raise SystemExit(1)
print(status or "unknown")
PY
}

retryable_infrastructure_failure() {
  local run_root="$1"
  "${PYTHON_BIN}" - "${run_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "manifest.json"
try:
    manifest = json.load(manifest_path.open(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
if manifest.get("status") == "interrupted":
    raise SystemExit(0)

metadata_paths = sorted(root.glob("q*_r*/attempt_*/metadata.json"))
if not metadata_paths:
    raise SystemExit(0)
for path in metadata_paths:
    try:
        metadata = json.load(path.open(encoding="utf-8"))
    except Exception:
        raise SystemExit(0)
    if metadata.get("exit_code") not in (0, None):
        raise SystemExit(0)
    if metadata.get("error_events") or metadata.get("invalid_jsonl_events"):
        raise SystemExit(0)
    if metadata.get("launch_error"):
        raise SystemExit(0)
    capture = metadata.get("reasoning_capture") or {}
    if capture and capture.get("status") != "captured":
        raise SystemExit(0)

# Exit code 0 plus turn completion and no infrastructure errors means the
# model itself failed to produce a valid final answer.  That is a scored wrong
# answer and must never be retried into a more favorable sample.
last = json.load(metadata_paths[-1].open(encoding="utf-8"))
event_counts = last.get("event_type_counts") or {}
if last.get("exit_code") == 0 and event_counts.get("turn.completed", 0) > 0:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_one() {
  local case_id="$1"
  local repeat="$2"
  local retry_count="${3:-0}"
  local repeat_text
  repeat_text="$(printf '%02d' "${repeat}")"
  local run_name="${RUN_PREFIX}-q${case_id}-r${repeat_text}"
  local run_root="${OUTPUT_ROOT}/${run_name}"
  local stdout_log="${CONTROL_DIR}/logs/q${case_id}-r${repeat_text}.stdout.log"
  local stderr_log="${CONTROL_DIR}/logs/q${case_id}-r${repeat_text}.stderr.log"
  local timeout_marker="${run_root}/.timeout_${TIMEOUT_SECONDS}s"
  local resume=0
  local prior_status=""
  local archive_root=""

  if (( retry_count == 0 )); then
    retry_count="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
      -name "${run_name}.infra_failed_*" 2>/dev/null | wc -l)"
  fi

  if [[ -f "${timeout_marker}" ]]; then
    log "skip timeout case=${case_id} repeat=${repeat}"
    return
  fi
  if [[ -f "${run_root}/manifest.json" ]]; then
    prior_status="$(manifest_status "${run_root}/manifest.json" 2>/dev/null || true)"
    if [[ "${prior_status}" == "succeeded" ]]; then
      log "skip succeeded case=${case_id} repeat=${repeat}"
      return
    elif [[ "${prior_status}" == "failed" || "${prior_status}" == "interrupted" ]]; then
      if ! retryable_infrastructure_failure "${run_root}"; then
        log "skip terminal model failure case=${case_id} repeat=${repeat} status=${prior_status}"
        return
      fi
      if (( retry_count >= INFRA_MAX_RETRIES )); then
        log "infrastructure retries exhausted case=${case_id} repeat=${repeat} status=${prior_status} retries=${retry_count}"
        return
      fi
      archive_root="${run_root}.infra_failed_$(date -u +%Y%m%dT%H%M%S%NZ)"
      mv -- "${run_root}" "${archive_root}"
      retry_count=$((retry_count + 1))
      log "archive infrastructure failure case=${case_id} repeat=${repeat} status=${prior_status} retry=${retry_count}/${INFRA_MAX_RETRIES} archive=${archive_root}"
      resume=0
    else
      resume=1
    fi
  fi

  log "start case=${case_id} repeat=${repeat} resume=${resume}"
  (
    cd "${REPO_ROOT}" || exit 125
    if (( resume == 1 )); then
      exec "${PYTHON_BIN}" "${RUNNER}" \
        --dataset "${DATASET}" \
        --template "${TEMPLATE}" \
        --output-root "${OUTPUT_ROOT}" \
        --codex-bin "${CODEX_WRAPPER}" \
        --model "${MODEL_NAME}" \
        --credit-retry-seconds 0 \
        --resume-run "${run_root}"
    else
      exec "${PYTHON_BIN}" "${RUNNER}" \
        --dataset "${DATASET}" \
        --template "${TEMPLATE}" \
        --output-root "${OUTPUT_ROOT}" \
        --case-ids "${case_id}" \
        --codex-bin "${CODEX_WRAPPER}" \
        --model "${MODEL_NAME}" \
        --sandbox danger-full-access \
        --repeats 1 \
        --credit-retry-seconds 0 \
        --run-name "${run_name}"
    fi
  ) >"${stdout_log}" 2>"${stderr_log}" &
  local runner_pid=$!
  local started
  started="$(date +%s)"
  local timed_out=0

  while kill -0 "${runner_pid}" 2>/dev/null; do
    if (( $(date +%s) - started >= TIMEOUT_SECONDS )); then
      timed_out=1
      mkdir -p "${run_root}"
      touch "${timeout_marker}"
      log "timeout case=${case_id} repeat=${repeat} pid=${runner_pid}"
      terminate_descendants "${runner_pid}"
      break
    fi
    sleep 5
  done
  if (( timed_out == 1 )); then
    for _ in $(seq 1 30); do
      kill -0 "${runner_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${runner_pid}" 2>/dev/null; then
      terminate_descendants "${runner_pid}"
      kill -TERM "${runner_pid}" 2>/dev/null || true
      sleep 2
      kill -KILL "${runner_pid}" 2>/dev/null || true
    fi
  fi

  local rc=0
  if wait "${runner_pid}"; then
    rc=0
  else
    rc=$?
  fi
  local enrichment_rc=0
  local events_path
  while IFS= read -r -d '' events_path; do
    if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/enrich_codex_events_with_reasoning.py" \
      --events "${events_path}" \
      --require-reasoning; then
      enrichment_rc=1
      log "reasoning enrichment failed case=${case_id} repeat=${repeat} events=${events_path}"
    fi
  done < <(find "${run_root}" -type f -name events.jsonl -print0 2>/dev/null)
  if (( rc == 0 && enrichment_rc != 0 )); then
    rc=70
  fi
  mkdir -p "${run_root}"
  printf '%s\n' "${rc}" >"${run_root}/.runner_exit_code"
  log "end case=${case_id} repeat=${repeat} rc=${rc} timeout=${timed_out} wall_seconds=$(($(date +%s)-started))"
  if (( rc != 0 && timed_out == 0 && retry_count < INFRA_MAX_RETRIES )) \
    && retryable_infrastructure_failure "${run_root}"; then
    archive_root="${run_root}.infra_failed_$(date -u +%Y%m%dT%H%M%S%NZ)"
    mv -- "${run_root}" "${archive_root}"
    retry_count=$((retry_count + 1))
    log "retry infrastructure failure case=${case_id} repeat=${repeat} rc=${rc} retry=${retry_count}/${INFRA_MAX_RETRIES} archive=${archive_root}"
    run_one "${case_id}" "${repeat}" "${retry_count}"
  elif (( rc != 0 && timed_out == 0 )); then
    log "retain scored model failure case=${case_id} repeat=${repeat} rc=${rc}"
  fi
}

log "Agent validation start prefix=${RUN_PREFIX} cases=${CASE_IDS} repeats=${REPEATS} topology=tp2x1/concurrency2 thinking=enabled raw_reasoning=captured reasoning_effort=${REASONING_EFFORT}"
for repeat in $(seq 1 "${REPEATS}"); do
  # Keep two runner slots occupied continuously.  The previous implementation
  # waited for both members of a pair, which left one of the two permitted
  # runners idle whenever its peer took longer.  Refill only after a runner
  # exits, so this remains capped at the required two Agent runners.
  active=()
  next_case_index=0
  while (( next_case_index < ${#CASES[@]} || ${#active[@]} > 0 )); do
    while (( next_case_index < ${#CASES[@]} && ${#active[@]} < 2 )); do
      case_id="${CASES[${next_case_index}]}"
      run_one "${case_id}" "${repeat}" &
      active+=("$!")
      next_case_index=$((next_case_index + 1))
    done

    # Wait only until at least one slot is free, then return to the refill
    # loop.  Polling child PIDs avoids relying on Bash-version-specific
    # `wait -n -p` behaviour while preserving the hard concurrency cap.
    completed=0
    while (( completed == 0 && ${#active[@]} > 0 )); do
      remaining=()
      for pid in "${active[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          remaining+=("${pid}")
        else
          wait "${pid}" || true
          completed=1
        fi
      done
      active=("${remaining[@]}")
      if (( completed == 0 )); then
        sleep 1
      fi
    done
  done
done

summary_args=(
  --output-root "${OUTPUT_ROOT}"
  --run-prefix "${RUN_PREFIX}"
  --case-ids "${CASES[@]}"
  --repeats "${REPEATS}"
  --dataset "${DATASET}"
  --report-dir "${REPORT_DIR}"
  --model "${MODEL_NAME}"
  --checkpoint "${CHECKPOINT}"
  --git-commit "${GIT_COMMIT}"
  --timeout-seconds "${TIMEOUT_SECONDS}"
  --reasoning-effort "${REASONING_EFFORT}"
)
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  summary_args+=(--baseline-summary "${BASELINE_SUMMARY}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_agent_validation.py" "${summary_args[@]}"
log "Agent validation complete summary=${REPORT_DIR}/validation_summary.json"
