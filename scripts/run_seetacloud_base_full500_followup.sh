#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/qwen-codex-eval}"
PRIMARY_PREFIX="${PRIMARY_PREFIX:?PRIMARY_PREFIX is required}"
PRIMARY_REPORT_DIR="${PRIMARY_REPORT_DIR:-${OUTPUT_ROOT}/${PRIMARY_PREFIX}-report}"
REPLACEMENT_PREFIX="${REPLACEMENT_PREFIX:-${PRIMARY_PREFIX}-heldout-timeout-replacements}"
REPLACEMENT_REPORT_DIR="${REPLACEMENT_REPORT_DIR:-${OUTPUT_ROOT}/${REPLACEMENT_PREFIX}-report}"
FINAL_REPORT_DIR="${FINAL_REPORT_DIR:-${OUTPUT_ROOT}/${PRIMARY_PREFIX}-full500-report}"
HISTORICAL_ATTEMPTS="${HISTORICAL_ATTEMPTS:-${OUTPUT_ROOT}/train0629_heldout8_base_vs_epoch7_20260728_attempts.csv}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG="${LOG:-${OUTPUT_ROOT}/${PRIMARY_PREFIX}-full500-followup.log}"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date -Iseconds)] full500 follow-up start primary=${PRIMARY_PREFIX} wait_pid=${WAIT_FOR_PID:-none}"
if [[ -n "${WAIT_FOR_PID}" ]]; then
  if [[ ! "${WAIT_FOR_PID}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WAIT_FOR_PID must be a positive integer" >&2
    exit 1
  fi
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
fi

PRIMARY_SUMMARY="${PRIMARY_REPORT_DIR}/validation_summary.json"
if [[ ! -f "${PRIMARY_SUMMARY}" ]]; then
  echo "Primary summary is absent after the primary launcher exited: ${PRIMARY_SUMMARY}" >&2
  exit 1
fi
if [[ ! -f "${HISTORICAL_ATTEMPTS}" ]]; then
  echo "Historical heldout attempts are absent: ${HISTORICAL_ATTEMPTS}" >&2
  exit 1
fi

if [[ ! -f "${REPLACEMENT_REPORT_DIR}/validation_summary.json" ]]; then
  echo "[$(date -Iseconds)] running q89/q90/q99 timeout replacements with tp2x1/concurrency2"
  cd "${REPO_ROOT}"
  CASE_IDS="89,90,99" \
  REPEATS=1 \
  RUN_PREFIX="${REPLACEMENT_PREFIX}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  REPORT_DIR="${REPLACEMENT_REPORT_DIR}" \
    bash "${SCRIPT_DIR}/run_seetacloud_base_agent_eval.sh"
else
  echo "[$(date -Iseconds)] replacement summary already exists; skipping replacement run"
fi

echo "[$(date -Iseconds)] composing audited 100x5 report"
"${PYTHON_BIN}" "${SCRIPT_DIR}/compose_base_full500_eval.py" \
  --primary-summary "${PRIMARY_SUMMARY}" \
  --historical-attempts "${HISTORICAL_ATTEMPTS}" \
  --replacement-summary "${REPLACEMENT_REPORT_DIR}/validation_summary.json" \
  --report-dir "${FINAL_REPORT_DIR}"
echo "[$(date -Iseconds)] full500 follow-up complete summary=${FINAL_REPORT_DIR}/validation_summary.json"
