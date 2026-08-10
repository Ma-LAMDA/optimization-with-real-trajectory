#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/agent_validation_retry_policy.sh"

launcher="${SCRIPT_DIR}/run_agent_validation_resilient.sh"
grep -Fq 'source "${SCRIPT_DIR}/agent_validation_retry_policy.sh"' "${launcher}"
grep -Fq 'agent_validation_retry_archive_count "${OUTPUT_ROOT}" "${run_name}"' "${launcher}"
grep -Fq 'agent_validation_archive_timeout "${run_root}"' "${launcher}"
grep -Fq 'retry timeout case=${case_id} repeat=${repeat}' "${launcher}"

test_root="$(mktemp -d)"
cleanup() {
  [[ -n "${test_root:-}" && -d "${test_root}" && "${test_root}" == /tmp/* ]] && rm -rf -- "${test_root}"
}
trap cleanup EXIT

run_name=agent-q29-r02
run_root="${test_root}/${run_name}"
mkdir -p "${run_root}"
printf 'preserved timeout evidence\n' >"${run_root}/manifest.json"
touch "${run_root}/.timeout_3600s"

archive="$(agent_validation_archive_timeout "${run_root}" 20260810T010203Z)"
[[ "${archive}" == "${run_root}.timeout_failed_20260810T010203Z" ]]
[[ ! -e "${run_root}" ]]
[[ -f "${archive}/.timeout_3600s" ]]
grep -Fqx 'preserved timeout evidence' "${archive}/manifest.json"
[[ "$(agent_validation_retry_archive_count "${test_root}" "${run_name}")" == 1 ]]

mkdir -p "${test_root}/${run_name}.infra_failed_20260810T010204Z"
[[ "$(agent_validation_retry_archive_count "${test_root}" "${run_name}")" == 2 ]]
mkdir -p "${test_root}/unrelated.timeout_failed_20260810T010205Z"
[[ "$(agent_validation_retry_archive_count "${test_root}" "${run_name}")" == 2 ]]

echo "agent validation timeout retry policy: PASS"
