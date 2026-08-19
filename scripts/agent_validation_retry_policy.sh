#!/usr/bin/env bash

# Pure filesystem helpers shared by the Agent validation launcher and its
# regression test. Timeout evidence is moved to a timestamped sibling before a
# retry so it remains auditable and never occupies an effective result slot.

agent_validation_retry_archive_count() {
  local output_root="$1"
  local run_name="$2"
  find "${output_root}" -maxdepth 1 -type d \
    \( -name "${run_name}.infra_failed_*" -o -name "${run_name}.timeout_failed_*" \) \
    -print 2>/dev/null | wc -l | tr -d '[:space:]'
}

agent_validation_archive_timeout() {
  local run_root="$1"
  local timestamp="${2:-$(date -u +%Y%m%dT%H%M%S%NZ)}"
  local archive_root="${run_root}.timeout_failed_${timestamp}"

  [[ -d "${run_root}" ]] || {
    echo "Timeout run root does not exist: ${run_root}" >&2
    return 1
  }
  compgen -G "${run_root}/.timeout_*s" >/dev/null || {
    echo "Timeout marker is missing: ${run_root}" >&2
    return 1
  }
  [[ ! -e "${archive_root}" ]] || {
    echo "Timeout archive already exists: ${archive_root}" >&2
    return 1
  }
  mv -- "${run_root}" "${archive_root}"
  printf '%s\n' "${archive_root}"
}
