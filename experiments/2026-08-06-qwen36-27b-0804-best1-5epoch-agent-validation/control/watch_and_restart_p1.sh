#!/usr/bin/env bash
set -euo pipefail

CONTROL=/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/control
RUN_ROOT=/root/autodl-tmp/optimization-with-real-trajectory/output/2026-08-05-nightly/0804/0804-5epoch-20260805T114046Z
LOG="${CONTROL}/p1_recovery_watch.log"
OLD_PID="$(cat "${CONTROL}/p1_agent_orchestrator.pid")"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"${LOG}"
}

log "watching orchestrator pid=${OLD_PID} for infrastructure-safe recovery"
while kill -0 "${OLD_PID}" 2>/dev/null; do
  sleep 10
done
log "orchestrator pid=${OLD_PID} exited"

if [[ -f "${RUN_ROOT}/p1_complete.json" ]]; then
  log "P1 already complete; no recovery launch needed"
  exit 0
fi

if ! mkdir "${CONTROL}/p1_restart.lock" 2>/dev/null; then
  log "another recovery owner already holds p1_restart.lock"
  exit 0
fi
trap 'rmdir "${CONTROL}/p1_restart.lock" 2>/dev/null || true' EXIT
printf '%s\n' "$$" >"${CONTROL}/p1_restart.lock/pid"

# The previous orchestrator owns vLLM, Codex runners, and the GPU sampler.
# Do not relaunch until its cleanup has removed every GPU compute process.
idle=0
for _ in $(seq 1 60); do
  compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [[ -z "${compute_pids}" ]]; then
    idle=1
    break
  fi
  sleep 5
done
if (( idle == 0 )); then
  log "GPU compute processes remained after cleanup window; leaving recovery to next heartbeat"
  exit 1
fi

log "GPU cleanup complete; restarting resumable P1 orchestrator"
nohup bash "${CONTROL}/run_0804_checkpoint_selection_and_final.sh" \
  >>"${RUN_ROOT}/agent_recovery_launcher.log" 2>&1 &
NEW_PID=$!
printf '%s\n' "${NEW_PID}" >"${CONTROL}/p1_agent_launcher.pid"
sleep 5
if ! kill -0 "${NEW_PID}" 2>/dev/null; then
  log "recovery launcher pid=${NEW_PID} exited during smoke window"
  exit 1
fi
log "recovery launcher started pid=${NEW_PID}"
