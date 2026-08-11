#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_ENV="${TRAIN_ENV:-/root/autodl-tmp/envs/qwen36-sft}"
PYTHON_BIN="${PYTHON_BIN:-${TRAIN_ENV}/bin/python}"
SWIFT_BIN="${SWIFT_BIN:-${TRAIN_ENV}/bin/swift}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/qwen3.6-27b/models/Qwen3.6-27B}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/2026-08-09}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/qwen36-27b-0809-agent-error-aware-5epoch}"
FORMAL_CONFIG_PATH="${FORMAL_CONFIG_PATH:-${REPO_ROOT}/config/qwen36_0809_formal_training.json}"
LR_PLUGIN_PATH="${LR_PLUGIN_PATH:-${REPO_ROOT}/scripts/qwen36_0809_fixed_stage_lr_plugin.py}"
MANIFEST_PATH="${MANIFEST_PATH:-${DATA_ROOT}/sft/0809_agent_error_aware_manifest.json}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-${DATA_ROOT}/sft/qwen3_6_27b_0809_validation.jsonl}"
PREFLIGHT_PATH="${PREFLIGHT_PATH:-${DATA_ROOT}/sft/TARGET_TOKENIZER_PREFLIGHT.json}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUNTIME_GATE_ONLY="${RUNTIME_GATE_ONLY:-0}"

if [[ "$(hostname)" == *Armstrong* ]]; then
  for path in "${REPO_ROOT}" "${TRAIN_ENV}" "${MODEL_PATH}" "${OUTPUT_DIR}"; do
    if [[ "${path}" != /Qdata/mayf/* ]]; then
      echo "Armstrong task data must stay under /Qdata/mayf: ${path}" >&2
      exit 1
    fi
  done
fi

for path in "${PYTHON_BIN}" "${SWIFT_BIN}" "${MODEL_PATH}" "${FORMAL_CONFIG_PATH}" "${LR_PLUGIN_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required 0809 training path is missing: ${path}" >&2
    exit 1
  fi
done

DATA_ROOT="$(${PYTHON_BIN} - "${DATA_ROOT}" "${REPO_ROOT}/data/2026-08-09" <<'PY'
from pathlib import Path
import sys
actual = Path(sys.argv[1]).resolve()
expected = Path(sys.argv[2]).resolve()
if actual != expected:
    raise SystemExit(f"0809 DATA_ROOT mismatch: {actual} != {expected}")
print(actual)
PY
)"
MANIFEST_PATH="${DATA_ROOT}/sft/0809_agent_error_aware_manifest.json"
VALIDATION_DATASET_PATH="${DATA_ROOT}/sft/qwen3_6_27b_0809_validation.jsonl"
PREFLIGHT_PATH="${DATA_ROOT}/sft/TARGET_TOKENIZER_PREFLIGHT.json"

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# Formal training is read-only with respect to the frozen release.  Generation,
# audit rendering and tokenizer preflight belong to the documented preparation
# phase and must have been committed before this entry is invoked.
"${PYTHON_BIN}" -B scripts/validate_0809_agent_error_aware_sft.py --data-root "${DATA_ROOT}"
"${PYTHON_BIN}" -B scripts/independent_validate_0809_sft.py --data-root "${DATA_ROOT}"

declare -A CFG=()
declare -a LEARNING_RATES=()
declare -a EXPECTED_STEPS=()
while IFS=$'\t' read -r key value; do
  case "${key}" in
    fixed_learning_rate_by_epoch) LEARNING_RATES+=("${value}") ;;
    expected_cumulative_optimizer_steps) EXPECTED_STEPS+=("${value}") ;;
    *) CFG["${key}"]="${value}" ;;
  esac
done < <(
  "${PYTHON_BIN}" - "${FORMAL_CONFIG_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
keys = (
    "schema_version", "data_root", "distributed_strategy", "world_size",
    "cuda_visible_devices", "model", "check_model", "tuner_type", "torch_dtype",
    "freeze_vit", "freeze_aligner", "target_modules", "lora_rank", "lora_alpha",
    "lora_dropout", "max_length", "truncation_strategy", "loss_scale",
    "is_binary_loss_scale", "preserve_thinking", "add_non_thinking_prefix", "epochs",
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "effective_batch_size", "expected_rows_per_epoch",
    "expected_optimizer_steps_per_epoch", "lr_scheduler_type", "warmup_ratio",
    "gradient_checkpointing", "packing", "split_dataset_ratio", "dataset_num_proc",
    "dataloader_num_workers", "logging_steps", "eval_strategy", "save_strategy",
    "save_total_limit", "load_best_model_at_end", "resume_only_model", "seed",
    "data_seed", "report_to", "add_version", "fixed_validation_epoch",
    "fixed_validation_global_step", "agent_checkpoint_selection",
)
def render(value):
    return str(value).lower() if isinstance(value, bool) else str(value)
for key in keys:
    print(f"{key}\t{render(config[key])}")
for value in config["fixed_learning_rate_by_epoch"]:
    print(f"fixed_learning_rate_by_epoch\t{value}")
for value in config["expected_cumulative_optimizer_steps"]:
    print(f"expected_cumulative_optimizer_steps\t{value}")
PY
)

if [[ "${CFG[schema_version]}" != "qwen36-0809-formal-training.v2" ]]; then
  echo "Unexpected 0809 training schema: ${CFG[schema_version]}" >&2
  exit 1
fi
if [[ "${CFG[data_root]}" != "data/2026-08-09" ]]; then
  echo "Training config data root is stale: ${CFG[data_root]}" >&2
  exit 1
fi
if [[ "${CFG[distributed_strategy]}" != "ddp" || "${CFG[world_size]}" != "2" ]]; then
  echo "0809 formal training requires two-process DDP." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES}" != "${CFG[cuda_visible_devices]}" || "${NPROC_PER_NODE}" != "${CFG[world_size]}" ]]; then
  echo "Runtime GPU/DDP settings differ from the frozen config." >&2
  exit 1
fi
if [[ "${CFG[per_device_train_batch_size]}" != "1" || "${CFG[gradient_accumulation_steps]}" != "4" || "${CFG[effective_batch_size]}" != "8" ]]; then
  echo "0809 requires per-device batch 1, accumulation 4, effective batch 8." >&2
  exit 1
fi
if [[ "${CFG[check_model]}" != "true" ]]; then
  echo "0809 formal training requires model compatibility checks." >&2
  exit 1
fi
if [[ "${CFG[expected_rows_per_epoch]}" != "1151" || "${CFG[expected_optimizer_steps_per_epoch]}" != "144" ]]; then
  echo "0809 must retain the 0805 row/step budget." >&2
  exit 1
fi
if [[ "${CFG[fixed_validation_epoch]}" != "3" || "${CFG[fixed_validation_global_step]}" != "432" || "${CFG[agent_checkpoint_selection]}" != "false" ]]; then
  echo "0809 must use the predeclared epoch-3/checkpoint-432 validation point." >&2
  exit 1
fi
if [[ "${#LEARNING_RATES[@]}" -ne 5 || "${#EXPECTED_STEPS[@]}" -ne 5 ]]; then
  echo "0809 requires five learning rates and five cumulative step boundaries." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PREFLIGHT_PATH}" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
preflight = Path(sys.argv[2])
record = manifest.get("target_tokenizer_preflight", {})
if record.get("status") != "passed" or not preflight.is_file():
    raise SystemExit(
        "0809 frozen release lacks a passing target-tokenizer preflight; run the documented preparation phase and commit the regenerated manifest first"
    )
PY

# Bind the actual runtime, not only the repository defaults, to the archived
# tokenizer/config/plugin identity. Deployment paths may differ across hosts;
# bytes and library/template identities may not.
"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${PREFLIGHT_PATH}" "${MODEL_PATH}" "${FORMAL_CONFIG_PATH}" "${LR_PLUGIN_PATH}" "${PYTHON_BIN}" "${SWIFT_BIN}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import swift
import transformers

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
preflight = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
model = Path(sys.argv[3]).resolve()
config = Path(sys.argv[4]).resolve()
plugin = Path(sys.argv[5]).resolve()
# Preserve the invoked virtual-environment entry points.  Python is commonly
# a symlink to the base interpreter, so resolve() would incorrectly make a
# same-environment python/swift pair look like two different environments.
python_bin = Path(sys.argv[6]).absolute()
swift_bin = Path(sys.argv[7]).absolute()

def lf_digest(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()

def raw_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

tracked = manifest["reproducibility"]["tracked_files"]
for actual, key in ((config, "formal_config"), (plugin, "lr_plugin")):
    if lf_digest(actual) != tracked[key]["sha256_lf_normalized"]:
        raise SystemExit(f"runtime {key} differs from archived identity: {actual}")
if python_bin.parent != swift_bin.parent:
    raise SystemExit(f"PYTHON_BIN and SWIFT_BIN must come from the same environment: {python_bin}, {swift_bin}")
if getattr(swift, "__version__", "unknown") != preflight["ms_swift_version"]:
    raise SystemExit("runtime ms-swift version differs from tokenizer preflight")
if transformers.__version__ != preflight["transformers_version"]:
    raise SystemExit("runtime transformers version differs from tokenizer preflight")
for filename, record in preflight["model_identity_files"].items():
    path = model / filename
    if not path.is_file() or path.stat().st_size != record["bytes"] or raw_digest(path) != record["sha256_raw"]:
        raise SystemExit(f"runtime model/tokenizer identity mismatch: {filename}")
required_identity = {"tokenizer.json", "tokenizer_config.json", "config.json", "model.safetensors.index.json"}
if not required_identity <= set(preflight["model_identity_files"]):
    raise SystemExit("preflight lacks required tokenizer/model weight-index identity")
print(
    "0809 runtime identity passed: "
    f"swift={preflight['ms_swift_version']} transformers={preflight['transformers_version']} "
    f"identity_files={len(preflight['model_identity_files'])}"
)
PY

while IFS= read -r release_path; do
  if ! git ls-files --error-unmatch "${release_path}" >/dev/null 2>&1; then
    echo "0809 formal training requires every release dependency to be Git tracked: ${release_path}" >&2
    exit 1
  fi
  if ! git diff --quiet -- "${release_path}" || ! git diff --cached --quiet -- "${release_path}"; then
    echo "0809 formal training refuses a modified release dependency: ${release_path}" >&2
    exit 1
  fi
done < <(
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
paths = {
    manifest_path.relative_to(Path.cwd()).as_posix(),
    "data/2026-08-09/sft/RELEASE_RECORD.json",
    "data/2026-08-09/sft/TARGET_TOKENIZER_PREFLIGHT.json",
    "data/2026-08-09/AUDIT_REPORT.md",
    "data/2026-08-09/curation/AUDIT_METRICS.json",
    "scripts/train_qwen36_0809_agent_error_aware_5epoch.sh",
}
paths.update(record["path"] for record in manifest["outputs"].values())
paths.update(
    record["path"] for record in manifest["reproducibility"]["tracked_files"].values()
)
for path in sorted(paths):
    print(path)
PY
)

LR_AUDIT_PATH="${OUTPUT_DIR}/control/learning_rate_audit.jsonl"

plugin_registration_smoke_check() {
  QWEN36_0809_TRAIN_STAGE=1 \
  QWEN36_0809_TARGET_LR="${LEARNING_RATES[0]}" \
  QWEN36_0809_LR_AUDIT_PATH="${LR_AUDIT_PATH}.registration-smoke" \
  "${PYTHON_BIN}" - "${LR_PLUGIN_PATH}" <<'PY'
import importlib.util
import sys
from pathlib import Path
from swift.callbacks import callbacks_map

path = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("qwen36_0809_fixed_stage_lr_plugin_smoke", path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load 0809 LR plugin: {path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if "qwen36_0809_fixed_stage_lr" not in callbacks_map:
    raise SystemExit("0809 LR callback was not registered by the external plugin")
print(f"0809 LR callback registration passed: {path}")
PY
}

"${PYTHON_BIN}" - "${DATA_ROOT}" "${FORMAL_CONFIG_PATH}" <<'PY'
import json
import math
import sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for epoch in range(1, 6):
    paths = [
        root / "sft" / f"qwen3_6_27b_0809_core_epoch_{epoch:02d}.jsonl",
        root / "sft" / f"qwen3_6_27b_0809_endpoint_epoch_{epoch:02d}.jsonl",
    ]
    counts = [sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line) for path in paths]
    total = sum(counts)
    if counts != [1035, 116] or total != config["expected_rows_per_epoch"]:
        raise SystemExit(f"epoch {epoch}: dataset counts {counts}, total={total}")
    if math.ceil(total / config["effective_batch_size"]) != config["expected_optimizer_steps_per_epoch"]:
        raise SystemExit(f"epoch {epoch}: optimizer-step arithmetic changed")
print(f"resolved_data_root={root}")
PY

if [[ "${RUNTIME_GATE_ONLY}" == "1" ]]; then
  plugin_registration_smoke_check
  echo "0809 runtime identity/data/LR-plugin gate passed; training intentionally not started."
  exit 0
fi

latest_checkpoint() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import re
import sys
items = []
for path in Path(sys.argv[1]).glob("checkpoint-*"):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match and path.is_dir():
        items.append((int(match.group(1)), path))
if items:
    print(max(items)[1])
PY
}

checkpoint_state() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import math
import sys
from pathlib import Path
state = json.loads((Path(sys.argv[1]) / "trainer_state.json").read_text(encoding="utf-8"))
epoch = float(state.get("epoch", -1))
rounded = round(epoch)
if not math.isclose(epoch, rounded, abs_tol=1e-6):
    raise SystemExit(f"checkpoint is not at an epoch boundary: {epoch}")
print(f"{rounded}\t{int(state['global_step'])}")
PY
}

existing_checkpoint="$(latest_checkpoint)"
completed_stage=0
if [[ -n "${existing_checkpoint}" ]]; then
  IFS=$'\t' read -r completed_stage completed_step < <(checkpoint_state "${existing_checkpoint}")
  if [[ "${completed_stage}" -lt 1 || "${completed_stage}" -gt 5 ]]; then
    echo "Existing checkpoint has invalid epoch ${completed_stage}." >&2
    exit 1
  fi
  if [[ "${completed_step}" != "${EXPECTED_STEPS[$((completed_stage - 1))]}" ]]; then
    echo "Existing checkpoint global step ${completed_step} violates the 0809 budget." >&2
    exit 1
  fi
elif [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output directory is nonempty but has no valid checkpoint; use a new OUTPUT_DIR." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/control"

if [[ "${completed_stage}" -ge 3 && ! -d "${OUTPUT_DIR}/checkpoint-432" ]]; then
  echo "Fixed epoch-3 checkpoint-432 is missing from a stage-${completed_stage} run." >&2
  exit 1
fi

if [[ "${completed_stage}" -eq 5 ]]; then
  echo "Formal 0809 training is already complete: ${existing_checkpoint}"
  echo "Fixed Agent checkpoint: ${OUTPUT_DIR}/checkpoint-432"
  exit 0
fi

for stage in 1 2 3 4 5; do
  if [[ "${stage}" -le "${completed_stage}" ]]; then
    continue
  fi
  stage_tag="$(printf '%02d' "${stage}")"
  core_dataset="${DATA_ROOT}/sft/qwen3_6_27b_0809_core_epoch_${stage_tag}.jsonl"
  endpoint_dataset="${DATA_ROOT}/sft/qwen3_6_27b_0809_endpoint_epoch_${stage_tag}.jsonl"
  target_lr="${LEARNING_RATES[$((stage - 1))]}"
  resume_args=()
  if [[ "${stage}" -gt 1 ]]; then
    if [[ -z "${existing_checkpoint}" ]]; then
      echo "Stage ${stage} requires the previous full-state checkpoint." >&2
      exit 1
    fi
    resume_args=(--resume_from_checkpoint "${existing_checkpoint}")
  fi
  export QWEN36_0809_TRAIN_STAGE="${stage}"
  export QWEN36_0809_TARGET_LR="${target_lr}"
  export QWEN36_0809_LR_AUDIT_PATH="${LR_AUDIT_PATH}"
  plugin_registration_smoke_check

  echo "Starting 0809 stage ${stage}/5: core=${core_dataset}, endpoint=${endpoint_dataset}, lr=${target_lr}"
  "${SWIFT_BIN}" sft \
    --model "${MODEL_PATH}" \
    --check_model "${CFG[check_model]}" \
    --tuner_type "${CFG[tuner_type]}" \
    --torch_dtype "${CFG[torch_dtype]}" \
    --freeze_vit "${CFG[freeze_vit]}" \
    --freeze_aligner "${CFG[freeze_aligner]}" \
    --target_modules "${CFG[target_modules]}" \
    --lora_rank "${CFG[lora_rank]}" \
    --lora_alpha "${CFG[lora_alpha]}" \
    --lora_dropout "${CFG[lora_dropout]}" \
    --dataset "${core_dataset}" "${endpoint_dataset}" \
    --val_dataset "${VALIDATION_DATASET_PATH}" \
    --max_length "${CFG[max_length]}" \
    --truncation_strategy "${CFG[truncation_strategy]}" \
    --loss_scale "${CFG[loss_scale]}" \
    --is_binary_loss_scale "${CFG[is_binary_loss_scale]}" \
    --preserve_thinking "${CFG[preserve_thinking]}" \
    --add_non_thinking_prefix "${CFG[add_non_thinking_prefix]}" \
    --num_train_epochs "${stage}" \
    --per_device_train_batch_size "${CFG[per_device_train_batch_size]}" \
    --per_device_eval_batch_size "${CFG[per_device_eval_batch_size]}" \
    --gradient_accumulation_steps "${CFG[gradient_accumulation_steps]}" \
    --learning_rate "${target_lr}" \
    --lr_scheduler_type "${CFG[lr_scheduler_type]}" \
    --warmup_ratio "${CFG[warmup_ratio]}" \
    --gradient_checkpointing "${CFG[gradient_checkpointing]}" \
    --packing "${CFG[packing]}" \
    --split_dataset_ratio "${CFG[split_dataset_ratio]}" \
    --dataset_num_proc "${CFG[dataset_num_proc]}" \
    --dataloader_num_workers "${CFG[dataloader_num_workers]}" \
    --logging_steps "${CFG[logging_steps]}" \
    --eval_strategy "${CFG[eval_strategy]}" \
    --save_strategy "${CFG[save_strategy]}" \
    --save_total_limit "${CFG[save_total_limit]}" \
    --load_best_model_at_end "${CFG[load_best_model_at_end]}" \
    --resume_only_model "${CFG[resume_only_model]}" \
    --seed "${CFG[seed]}" \
    --data_seed "${CFG[data_seed]}" \
    --report_to "${CFG[report_to]}" \
    --add_version "${CFG[add_version]}" \
    --output_dir "${OUTPUT_DIR}" \
    --external_plugins "${LR_PLUGIN_PATH}" \
    --callbacks qwen36_0809_fixed_stage_lr \
    "${resume_args[@]}"

  new_checkpoint="$(latest_checkpoint)"
  if [[ -z "${new_checkpoint}" || "${new_checkpoint}" == "${existing_checkpoint}" ]]; then
    echo "Stage ${stage} did not produce a new checkpoint." >&2
    exit 1
  fi
  IFS=$'\t' read -r completed_epoch completed_step < <(checkpoint_state "${new_checkpoint}")
  if [[ "${completed_epoch}" != "${stage}" || "${completed_step}" != "${EXPECTED_STEPS[$((stage - 1))]}" ]]; then
    echo "Stage ${stage} checkpoint boundary is epoch=${completed_epoch}, step=${completed_step}." >&2
    exit 1
  fi
  existing_checkpoint="${new_checkpoint}"
done

if [[ ! -d "${OUTPUT_DIR}/checkpoint-432" ]]; then
  echo "Fixed epoch-3 checkpoint-432 is missing." >&2
  exit 1
fi
echo "Formal 0809 five-stage training complete: ${existing_checkpoint}"
echo "Fixed Agent checkpoint: ${OUTPUT_DIR}/checkpoint-432"
