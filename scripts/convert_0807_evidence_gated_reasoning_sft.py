#!/usr/bin/env python3
"""Build the 0807 evidence-gated, case-balanced multi-turn SFT dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import itertools
import re
import shlex
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import convert_0804_best_trajectory_reasoning_sft as base


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-07"
SOURCE_CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
SOURCE_DATASET = ROOT / "data" / "simulation" / "train_0629.jsonl"
FROZEN_0804_CURATION = (
    ROOT / "data" / "2026-08-04" / "curation" / "accepted_trajectory_selection.json"
)
CLUSTER_SELECTION = DATA_ROOT / "curation" / "causal_path_clusters_per_case.json"
SFT_DIR = DATA_ROOT / "sft"
TRAIN_OUTPUT = SFT_DIR / "qwen3_6_27b_0807_train_semantic_pool.jsonl"
TRAIN_CORE_OUTPUT = SFT_DIR / "qwen3_6_27b_0807_core_pool.jsonl"
TRAIN_ENDPOINT_POOL_OUTPUT = (
    SFT_DIR / "qwen3_6_27b_0807_endpoint_pool.jsonl"
)
VALIDATION_OUTPUT = SFT_DIR / "qwen3_6_27b_0807_validation.jsonl"
MANIFEST_OUTPUT = SFT_DIR / "0807_evidence_gated_manifest.json"
TOKENIZER_PREFLIGHT_REPORT = SFT_DIR / "TARGET_TOKENIZER_PREFLIGHT.json"
REPRODUCIBILITY_DOC = DATA_ROOT / "REPRODUCIBILITY.md"
AUDIT_REPORT = DATA_ROOT / "AUDIT_REPORT.md"
AUDIT_METRICS = DATA_ROOT / "curation" / "AUDIT_METRICS.json"
CODEX_MODEL_CATALOG = ROOT / "config" / "codex_qwen_model_catalog.json"
CODEX_CLI_MODEL_SLUG = "Qwen3.6-27B-trained"

MAX_RETAINED_PATHS_PER_CASE = 4
TRAJECTORY_CLUSTER_THRESHOLD = 0.46
NODE_DUPLICATE_THRESHOLD = 0.76
TOOL_CALL_LOSS_SCALE = 0.02
THINKING_LOSS_SCALE_BY_SOURCE = {
    "pruned_original_visible_agent_message": 1.00,
    "observation_bound_reasoning_reconstruction": 0.60,
    "claim_bound_hypothesis_elimination": 0.60,
    "evidence_summary_bridge": 0.05,
    "stop_judgment_bridge": 0.10,
    "fixed_bridge_template": 0.00,
    "minimal_final_bridge": 0.00,
}
CONCLUSION_LOSS_SCALE_BY_SOURCE = {
    "original_visible_conclusion": 1.00,
    "evidence_aligned_reconstruction": 0.30,
    "source_grounded_hypothesis_elimination": 0.60,
    "path_evidence_synthesis": 0.05,
    "verified_path_stop_judgment": 0.10,
    "verified_final_answer": 1.00,
}
ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH = 1
CORE_EXPOSURES_PER_QUERY_PER_EPOCH = 2
ENDPOINT_SCHEDULE_EPOCHS = 5
ENDPOINT_TARGET_TYPES = ("endpoint_bundle",)
ENDPOINT_COMPONENT_TYPES = ("evidence_summary", "decision_ready", "decision")
ELIMINATION_MARKERS = (
    "排除",
    "不支持",
    "不能解释",
    "不成立",
    "不是根因",
    "并非根因",
    "暂不支持",
    "可排除",
    "反证",
)


def endpoint_schedule_output(epoch: int) -> Path:
    return SFT_DIR / f"qwen3_6_27b_0807_endpoint_epoch_{epoch:02d}.jsonl"


def core_schedule_output(epoch: int) -> Path:
    return SFT_DIR / f"qwen3_6_27b_0807_core_epoch_{epoch:02d}.jsonl"


OBSOLETE_OUTPUTS = [
    SFT_DIR / "reasoning_causal_path_manifest.json",
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train.jsonl",
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train_core.jsonl",
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl",
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_validation.jsonl",
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train_decision_pool.jsonl",
    *[
        SFT_DIR / f"qwen3_6_27b_reasoning_causal_path_train_decision_epoch_{epoch:02d}.jsonl"
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    ],
    *[
        SFT_DIR / f"qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_{epoch:02d}.jsonl"
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    ],
]


REPRODUCIBILITY_FILES = {
    "repository_rules": ROOT / "AGENTS.md",
    "root_readme": ROOT / "README.md",
    "date_readme": DATA_ROOT / "README.md",
    "reproducibility_document": REPRODUCIBILITY_DOC,
    "audit_report": AUDIT_REPORT,
    "audit_metrics": AUDIT_METRICS,
    "archive_converter": ROOT / "scripts" / "convert_accepted_only_100x10_to_sft.py",
    "inclusive_or_source_updater": ROOT / "scripts" / "update_q73_q86_inclusive_or.py",
    "inclusive_or_archive_sync": ROOT / "scripts" / "sync_0807_q73_q86_inclusive_or.py",
    "source_dataset": SOURCE_DATASET,
    "base_reasoning_converter": ROOT / "scripts" / "convert_0804_best_trajectory_reasoning_sft.py",
    "causal_path_converter": Path(__file__).resolve(),
    "causal_path_validator": ROOT / "scripts" / "validate_0807_evidence_gated_reasoning_sft.py",
    "independent_release_validator": ROOT / "scripts" / "independent_validate_0807_sft.py",
    "audit_generator": ROOT / "scripts" / "audit_0807_evidence_gated_sft.py",
    "tokenizer_preflight": ROOT / "scripts" / "check_0807_target_tokenizer_preflight.py",
    "training_entry": ROOT / "scripts" / "train_qwen36_0807_evidence_gated_5epoch.sh",
    "fixed_lr_callback": ROOT / "scripts" / "qwen36_0807_epoch_lr_callback.py",
}
if TOKENIZER_PREFLIGHT_REPORT.exists():
    REPRODUCIBILITY_FILES["target_tokenizer_preflight_report"] = (
        TOKENIZER_PREFLIGHT_REPORT
    )


def target_tokenizer_release_status() -> tuple[str, dict[str, Any] | None]:
    draft = "rule_validated_draft_requires_target_tokenizer_preflight"
    if not TOKENIZER_PREFLIGHT_REPORT.exists():
        return draft, None
    report = json.loads(TOKENIZER_PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    if (
        report.get("schema_version")
        != "0807-target-tokenizer-loss-mask-preflight.v1"
        or report.get("status") != "passed"
        or int(report.get("template", {}).get("release_max_length", 0)) != 16384
        or int(report.get("totals", {}).get("over_max_length_rows", -1)) != 0
        or int(report.get("totals", {}).get("loss_mask_failures", -1)) != 0
    ):
        return draft, report
    expected = {
        "train_core_pool": TRAIN_CORE_OUTPUT,
        "train_endpoint_pool": TRAIN_ENDPOINT_POOL_OUTPUT,
        "validation": VALIDATION_OUTPUT,
    }
    datasets = report.get("datasets", {})
    for name, path in expected.items():
        record = datasets.get(name, {})
        if not path.exists() or record.get("sha256_lf_normalized") != base.digest_file(path):
            return draft, report
    return "rule_and_target_tokenizer_validated_release_candidate", report

CODEX_CLI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Run a read-only shell command in the repository workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Linux shell command to execute.",
                    },
                    "justification": {
                        "type": "string",
                        "description": "Why this command is needed for the diagnosis.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Optional repository-relative working directory.",
                    },
                    "max_output_tokens": {
                        "type": "integer",
                        "description": "Optional maximum number of output tokens.",
                    },
                },
                "required": ["cmd"],
            },
        },
    }
]


def load_codex_cli_system_prompt() -> str:
    catalog = json.loads(CODEX_MODEL_CATALOG.read_text(encoding="utf-8"))
    matches = [
        model
        for model in catalog.get("models", [])
        if model.get("slug") == CODEX_CLI_MODEL_SLUG
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {CODEX_CLI_MODEL_SLUG} entry in {CODEX_MODEL_CATALOG}"
        )
    prompt = matches[0].get("base_instructions")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{CODEX_CLI_MODEL_SLUG} has no base_instructions")
    return prompt


CODEX_CLI_SYSTEM_PROMPT = load_codex_cli_system_prompt()
CODEX_CLI_SYSTEM_PROMPT_SHA256 = hashlib.sha256(
    CODEX_CLI_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()

PATH_PATTERN = re.compile(
    r"(CampusNetwork[^\\/'\"]*)[\\/]+([^\\/'\"]+)[\\/]+([^\\/'\"]+\.txt)",
    re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*|\n+")

FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lldp", ("lldp",)),
    ("bpdu", ("bpdu",)),
    ("stp", ("mstp", "stp")),
    ("vrrp", ("vrrp", "master", "backup", "preempt")),
    ("mpls", ("mpls", "lsp", "ldp", "label")),
    ("srv6", ("srv6", "segment-routing", "policy")),
    ("isis", ("isis",)),
    ("ospf", ("ospf",)),
    ("bgp", ("bgp", "vpnv4")),
    ("routing", ("routing-table", "route", "routing", "traceroute")),
    ("interface", ("interface", "eth-trunk", "port")),
    ("arp", ("arp",)),
    ("mac", ("mac-address", "mac address")),
    ("config", ("current-configuration", "configuration", "config")),
    ("log", ("logbuffer", "log ", "log'", "log\"")),
    ("alarm", ("alarm",)),
    ("cpu", ("cpu",)),
    ("memory", ("memory",)),
)

FAMILY_NAMES = {
    "lldp": "LLDP 邻接拓扑",
    "bpdu": "BPDU 过滤状态",
    "stp": "STP/MSTP 状态",
    "vrrp": "VRRP 角色与抢占状态",
    "mpls": "MPLS/LSP 转发状态",
    "srv6": "SRv6 策略状态",
    "isis": "ISIS 路由状态",
    "ospf": "OSPF 路由与度量",
    "bgp": "BGP/VPN 路由状态",
    "routing": "三层路由状态",
    "interface": "接口状态",
    "arp": "ARP 状态",
    "mac": "MAC 学习状态",
    "config": "生效配置",
    "log": "日志",
    "alarm": "告警",
    "cpu": "CPU 状态",
    "memory": "内存状态",
    "discovery": "数据目录",
    "other": "补充状态",
}

PRIMARY_FAMILIES = {
    "全局STP未使能": {"stp", "config"},
    "STP BPDU被过滤": {"bpdu", "stp", "config"},
    "存在IP路由环路": {"routing", "bgp", "ospf", "isis", "config"},
    "存在MPLS标签环路": {"mpls", "isis", "bgp", "routing", "config"},
    "VRRP Master角色规划不合理": {"vrrp", "routing", "ospf", "config"},
    "VRRP工作在非抢占模式": {"vrrp", "config"},
}

SUPPORT_FAMILIES = {
    "全局STP未使能": {"stp", "bpdu", "interface", "mac", "config"},
    "STP BPDU被过滤": {"bpdu", "stp", "interface", "mac", "config"},
    "存在IP路由环路": {
        "routing", "bgp", "ospf", "isis", "arp", "interface", "config"
    },
    "存在MPLS标签环路": {
        "mpls", "isis", "bgp", "routing", "interface", "config"
    },
    "VRRP Master角色规划不合理": {
        "vrrp", "routing", "ospf", "interface", "config"
    },
    "VRRP工作在非抢占模式": {"vrrp", "interface", "config"},
}

LOW_VALUE_FAMILIES = {"cpu", "memory", "alarm"}
DISCOVERY_MARKERS = ("get-childitem", "get-location", "test-path")
POWERSHELL_WRAPPER_PATTERN = re.compile(
    r'^"[^"\n]*powershell\.exe"\s+-Command\s+"(?P<command>.*)"$',
    re.IGNORECASE | re.DOTALL,
)
PS_QUOTED_ITEM = r"(?:'[^']*'|\"[^\"]*\")"
PS_QUOTED_LIST = rf"{PS_QUOTED_ITEM}(?:,{PS_QUOTED_ITEM})*"
SAVED_CONFIGS_PATH_MARKER = "\\optimization-with-real-trajectory\\saved_configs\\"


def normalize_filename(value: str) -> str:
    stem = value.lower().removesuffix(".txt")
    stem = re.sub(r"_+", "_", stem)
    return stem


def normalize_supervised_command(value: str) -> str:
    """Remove the repeated process wrapper while retaining an executable PS command."""
    command = base.normalize_text(value)
    match = POWERSHELL_WRAPPER_PATTERN.match(command)
    if match:
        command = match.group("command")
    # Archived Windows commands contain JSON/shell-escaped duplicate path
    # separators.  A single separator is the canonical executable PowerShell
    # form and avoids supervising the same escaping boilerplate on every call.
    command = re.sub(r"\\{2,}", r"\\", command)
    command = command.replace(r'\"', '"')
    return command.strip()


def parse_powershell_quoted_list(value: str) -> list[str]:
    if not re.fullmatch(PS_QUOTED_LIST, value):
        raise ValueError(f"unsupported PowerShell quoted-list syntax: {value}")
    return [
        single if single else double
        for single, double in re.findall(r"'([^']*)'|\"([^\"]*)\"", value)
    ]


def parse_powershell_pattern_list(value: str) -> list[str]:
    if re.fullmatch(PS_QUOTED_LIST, value):
        return parse_powershell_quoted_list(value)
    # A few archived commands passed regex anchors through a nested
    # PowerShell/shell quote sequence such as '"'^Ethernet... '"'.  Remove
    # only that audited quote token and retain the actual regex text.
    cleaned = value.replace("'\"'", "")
    if "','" in cleaned:
        values = cleaned.split("','")
    else:
        values = cleaned.split(",")
    values = [item.strip().strip("'") for item in values]
    if not values or any(not item for item in values):
        raise ValueError(f"unsupported PowerShell pattern-list syntax: {value}")
    return values


def repository_relative_saved_config_path(value: str) -> str:
    normalized = value.replace("/", "\\")
    root_suffix = "\\optimization-with-real-trajectory\\saved_configs"
    if normalized.lower().endswith(root_suffix.lower()):
        return "saved_configs"
    marker_index = normalized.lower().find(SAVED_CONFIGS_PATH_MARKER.lower())
    if marker_index < 0:
        raise ValueError(f"path is outside the archived saved_configs tree: {value}")
    relative = normalized[marker_index + len(SAVED_CONFIGS_PATH_MARKER) :]
    if not relative or relative.startswith("\\") or ".." in relative.split("\\"):
        raise ValueError(f"unsafe saved_configs path: {value}")
    return "saved_configs/" + relative.replace("\\", "/")


def shell_path(value: str, *, allow_glob: bool = False) -> str:
    relative = repository_relative_saved_config_path(value)
    if allow_glob and any(marker in relative for marker in "*?["):
        if not re.fullmatch(r"[A-Za-z0-9_./*?\[\]-]+", relative):
            raise ValueError(f"unsafe shell glob: {relative}")
        return relative
    return shlex.quote(relative)


def parse_switches(value: str, switches_with_values: set[str]) -> dict[str, str | bool]:
    tokens = value.strip().split()
    parsed: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("-") or token in parsed:
            raise ValueError(f"unsupported or repeated PowerShell option: {value}")
        if token in switches_with_values:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("-"):
                raise ValueError(f"missing value for {token}: {value}")
            parsed[token] = tokens[index + 1]
            index += 2
        else:
            parsed[token] = True
            index += 1
    return parsed


def translation_justification(kind: str, relative_paths: list[str]) -> str:
    target = relative_paths[0]
    suffix = f" and {len(relative_paths) - 1} related paths" if len(relative_paths) > 1 else ""
    if kind == "read":
        return f"Read archived diagnostic evidence from {target}{suffix}."
    if kind == "search":
        return f"Search archived diagnostic evidence under {target}{suffix}."
    if kind == "list":
        return f"List archived diagnostic files under {target}{suffix}."
    return f"Check whether archived diagnostic evidence exists at {target}."


def translate_powershell_to_exec_command(value: str) -> dict[str, str]:
    """Strictly translate admitted read-only PowerShell into the eval Linux protocol."""
    command = normalize_supervised_command(value)

    if command.startswith("Select-String -SimpleMatch -Path "):
        command = "Select-String -Path " + command.removeprefix(
            "Select-String -SimpleMatch -Path "
        )
        pattern_index = command.find(" -Pattern ")
        if pattern_index < 0:
            raise ValueError(f"unsupported Select-String argument order: {command}")
        command = command[:pattern_index] + " -SimpleMatch" + command[pattern_index:]

    positional_select = re.fullmatch(
        rf"Select-String -Path (?P<paths>{PS_QUOTED_LIST}) -SimpleMatch "
        rf"(?P<patterns>{PS_QUOTED_LIST})"
        rf"(?P<after_options>(?:\s+-Context\s+\d+,\d+)*)",
        command,
    )
    if positional_select:
        command = (
            f"Select-String -Path {positional_select.group('paths')} -SimpleMatch "
            f"-Pattern {positional_select.group('patterns')}"
            f"{positional_select.group('after_options')}"
        )

    match = re.fullmatch(
        rf"Get-Content -LiteralPath (?P<paths>{PS_QUOTED_LIST})(?P<options>.*)",
        command,
    )
    if match:
        source_paths = parse_powershell_quoted_list(match.group("paths"))
        relative_paths = [repository_relative_saved_config_path(path) for path in source_paths]
        options = parse_switches(match.group("options"), {"-TotalCount", "-Tail", "-ErrorAction"})
        unknown = set(options) - {"-Raw", "-TotalCount", "-Tail", "-ErrorAction"}
        if unknown or ("-TotalCount" in options and "-Tail" in options):
            raise ValueError(f"unsupported Get-Content options: {command}")
        if options.get("-ErrorAction") not in (None, "SilentlyContinue"):
            raise ValueError(f"unsupported Get-Content ErrorAction: {command}")
        targets = " ".join(shell_path(path) for path in source_paths)
        if "-TotalCount" in options:
            cmd = f"head -q -n {int(str(options['-TotalCount']))} -- {targets}"
        elif "-Tail" in options:
            cmd = f"tail -q -n {int(str(options['-Tail']))} -- {targets}"
        else:
            cmd = f"cat -- {targets}"
        if "-ErrorAction" in options:
            cmd += " 2>/dev/null"
        return {
            "cmd": cmd,
            "justification": translation_justification("read", relative_paths),
            "kind": "get_content_to_posix_read",
        }

    match = re.fullmatch(
        rf"Select-String -Path (?P<paths>{PS_QUOTED_LIST})"
        rf"(?P<before_options>(?:\s+-SimpleMatch)*) -Pattern "
        rf"(?P<patterns>.*?)(?P<after_options>"
        rf"(?:\s+-(?:SimpleMatch|Context\s+\d+,\d+))*)",
        command,
    )
    if match:
        source_paths = parse_powershell_quoted_list(match.group("paths"))
        patterns = parse_powershell_pattern_list(match.group("patterns"))
        relative_paths = [repository_relative_saved_config_path(path) for path in source_paths]
        options = parse_switches(
            match.group("before_options") + match.group("after_options"),
            {"-Context"},
        )
        unknown = set(options) - {"-Context", "-SimpleMatch"}
        if unknown:
            raise ValueError(f"unsupported Select-String options: {command}")
        arguments = ["grep", "-nH", "-i"]
        if "-SimpleMatch" in options:
            arguments.append("-F")
        if "-Context" in options:
            context_match = re.fullmatch(r"(\d+),(\d+)", str(options["-Context"]))
            if not context_match:
                raise ValueError(f"unsupported Select-String context: {command}")
            before, after = map(int, context_match.groups())
            if before:
                arguments.extend(["-B", str(before)])
            if after:
                arguments.extend(["-A", str(after)])
        for pattern in patterns:
            arguments.extend(["-e", shlex.quote(pattern)])
        arguments.append("--")
        arguments.extend(shell_path(path, allow_glob=True) for path in source_paths)
        return {
            "cmd": " ".join(arguments),
            "justification": translation_justification("search", relative_paths),
            "kind": "select_string_to_grep",
        }

    match = re.fullmatch(
        rf"Get-ChildItem -LiteralPath (?P<paths>{PS_QUOTED_LIST})(?P<options>.*)",
        command,
    )
    if match:
        source_paths = parse_powershell_quoted_list(match.group("paths"))
        relative_paths = [repository_relative_saved_config_path(path) for path in source_paths]
        option_text = match.group("options")
        positional_name_filter = None
        positional_match = re.search(
            rf"(?<!\S)-Name\s+(?P<filter>{PS_QUOTED_LIST}|[^\s]+)(?=\s+-|$)",
            option_text,
        )
        if positional_match:
            positional_name_filter = positional_match.group("filter")
            option_text = (
                option_text[: positional_match.start()]
                + "-Name"
                + option_text[positional_match.end() :]
            )
        options = parse_switches(option_text, {"-Filter", "-Include"})
        unknown = set(options) - {
            "-Directory", "-Recurse", "-File", "-Filter", "-Include", "-Name", "-Force"
        }
        if unknown or ("-Directory" in options and "-File" in options):
            raise ValueError(f"unsupported Get-ChildItem options: {command}")
        explicit_filters = [
            str(options[key]) for key in ("-Filter", "-Include") if key in options
        ]
        if positional_name_filter:
            explicit_filters.append(positional_name_filter)
        if len(explicit_filters) > 1:
            raise ValueError(f"ambiguous Get-ChildItem filters: {command}")
        arguments = ["find", *(shell_path(path) for path in source_paths), "-mindepth", "1"]
        if "-Recurse" not in options:
            arguments.extend(["-maxdepth", "1"])
        if "-Directory" in options:
            arguments.extend(["-type", "d"])
        elif "-File" in options:
            arguments.extend(["-type", "f"])
        if explicit_filters:
            filters = parse_powershell_pattern_list(explicit_filters[0])
            if len(filters) == 1:
                arguments.extend(["-name", shlex.quote(filters[0])])
            else:
                arguments.append(r"\(")
                for index, pattern in enumerate(filters):
                    if index:
                        arguments.append("-o")
                    arguments.extend(["-name", shlex.quote(pattern)])
                arguments.append(r"\)")
        arguments.extend(["-printf", shlex.quote("%P\\n" if "-Name" in options else "%p\\n")])
        return {
            "cmd": " ".join(arguments) + " | sort",
            "justification": translation_justification("list", relative_paths),
            "kind": "get_child_item_to_find",
        }

    match = re.fullmatch(
        rf"Test-Path -LiteralPath (?P<paths>{PS_QUOTED_LIST})(?P<options>.*)",
        command,
    )
    if match:
        source_paths = parse_powershell_quoted_list(match.group("paths"))
        if len(source_paths) != 1 or match.group("options").strip():
            raise ValueError(f"unsupported Test-Path options: {command}")
        relative_paths = [repository_relative_saved_config_path(source_paths[0])]
        target = shell_path(source_paths[0])
        return {
            "cmd": f"test -e {target} && printf 'True\\n' || printf 'False\\n'",
            "justification": translation_justification("exists", relative_paths),
            "kind": "test_path_to_posix_test",
        }

    raise ValueError(f"unsupported PowerShell command; refusing supervised fallback: {command}")


def estimate_context_token_count(value: str) -> int:
    """Approximate the context-only Codex transport counter without claiming loss use."""
    return len(re.findall(r"[A-Za-z0-9_./:-]+|[^\s]", value))


WINDOWS_SAVED_CONFIGS_ROOT_PATTERN = re.compile(
    r"[A-Za-z]:\\[^\r\n]*?\\optimization-with-real-trajectory\\saved_configs"
    r"(?=\\|[\s:,\r\n]|$)",
    re.IGNORECASE,
)


def replace_windows_archive_paths(value: str) -> str:
    normalized_lines: list[str] = []
    for line in value.splitlines():
        line = WINDOWS_SAVED_CONFIGS_ROOT_PATTERN.sub("saved_configs", line)
        path_start = line.find("saved_configs")
        if path_start >= 0:
            path_end = line.lower().find(".txt", path_start)
            if path_end >= 0:
                path_end += 4
            else:
                path_end = len(line)
            line = (
                line[:path_start]
                + line[path_start:path_end].replace("\\", "/")
                + line[path_end:]
            )
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def normalize_find_output(action: dict[str, Any], value: str) -> str:
    source_command = str(action["source_powershell_command"])
    names_only = "-Name" in source_command
    source_match = re.fullmatch(
        rf"Get-ChildItem -LiteralPath (?P<paths>{PS_QUOTED_LIST})(?P<options>.*)",
        source_command,
    )
    source_paths = (
        parse_powershell_quoted_list(source_match.group("paths")) if source_match else []
    )
    source_roots = [repository_relative_saved_config_path(path) for path in source_paths]
    current_directory: str | None = source_roots[0] if len(source_roots) == 1 else None
    normalized: list[str] = []
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("...") and line.endswith("..."):
            normalized.append(line)
            continue
        directory_match = re.match(r"^(?:目录|Directory)\s*:\s*(.+)$", line, re.IGNORECASE)
        if directory_match:
            current_directory = replace_windows_archive_paths(
                directory_match.group(1)
            ).replace("\\", "/")
            continue
        if re.match(r"^(?:Mode\s+|LastWriteTime\s+|[-\s]{8,}$)", line, re.IGNORECASE):
            continue
        entry_match = re.match(
            r"^[d-][a-z-]{4,}\s+\S+\s+\S+\s+(?:\d+\s+)?(?P<name>.+?)\s*$",
            line,
            re.IGNORECASE,
        )
        if entry_match:
            name = entry_match.group("name").replace("\\", "/")
            normalized.append(
                name
                if names_only or not current_directory
                else f"{current_directory.rstrip('/')}/{name}"
            )
            continue
        fallback = replace_windows_archive_paths(line).replace("\\", "/")
        if not names_only and current_directory and not fallback.startswith("saved_configs/"):
            fallback = f"{current_directory.rstrip('/')}/{fallback}"
        normalized.append(fallback)
    return "\n".join(normalized)


def normalize_grep_output(value: str) -> str:
    lines = replace_windows_archive_paths(value).splitlines()
    normalized: list[str] = []
    current: str | None = None
    for raw_line in lines:
        path_start = raw_line.find("saved_configs/")
        if path_start >= 0:
            if current is not None:
                normalized.append(current)
            current = raw_line[path_start:].strip()
            continue
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("...") or line == "--":
            if current is not None:
                normalized.append(current)
                current = None
            normalized.append(line)
        elif current is not None:
            # PowerShell's formatted Select-String output wraps long paths and
            # result lines at console width.  GNU grep emits one physical line.
            current += line
        else:
            normalized.append(line)
    if current is not None:
        normalized.append(current)
    # Character-limited source excerpts can end halfway through the next
    # PowerShell path.  That fragment is transport noise, not evidence.
    cleaned = [
        re.sub(r"[A-Za-z]:\\.*$", "", line).rstrip()
        for line in normalized
    ]
    return "\n".join(line for line in cleaned if line)


def normalize_tool_output_for_exec_command(action: dict[str, Any], value: str) -> str:
    translation = str(action["command_translation"])
    if translation == "get_child_item_to_find":
        return normalize_find_output(action, value)
    if translation == "select_string_to_grep":
        return normalize_grep_output(value)
    return value


def codex_exec_command_output(action: dict[str, Any], output: str) -> str:
    chunk_id = hashlib.sha256(
        f"{action['command']}\n{output}".encode("utf-8")
    ).hexdigest()[:6]
    exit_code = int(action.get("exit_code") or 0)
    return (
        f"Chunk ID: {chunk_id}\n"
        "Wall time: 0.0000 seconds\n"
        f"Process exited with code {exit_code}\n"
        f"Original token count: {estimate_context_token_count(output)}\n"
        f"Output:\n{output}"
    )


def rewrite_messages_to_codex_cli_protocol(
    messages: list[dict[str, Any]], stages: list[dict[str, Any]], target_index: int
) -> None:
    call_actions = [
        action
        for stage in stages[: target_index + 1]
        for action in stage["actions"]
    ]
    response_actions = [
        action
        for stage in stages[:target_index]
        for action in stage["actions"]
    ]
    call_messages = [message for message in messages if message["role"] == "tool_call"]
    response_messages = [
        message for message in messages if message["role"] == "tool_response"
    ]
    if len(call_messages) != len(call_actions) or len(response_messages) != len(response_actions):
        raise ValueError("base message sequence and translated action sequence differ")
    for message, action in zip(call_messages, call_actions):
        message["content"] = json.dumps(
            {
                "name": "exec_command",
                "arguments": {"cmd": action["command"]},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    for message, action in zip(response_messages, response_actions):
        legacy = json.loads(message["content"])
        normalized_output = normalize_tool_output_for_exec_command(
            action, str(legacy["output"])
        )
        message["content"] = codex_exec_command_output(action, normalized_output)


def infer_families(text: str) -> set[str]:
    # `saved_configs` is a transport directory, not evidence that every
    # action belongs to the configuration family.
    lowered = re.sub(r"\bsaved_configs\b", "", text.lower())
    families: set[str] = set()
    for family, patterns in FAMILY_PATTERNS:
        for pattern in patterns:
            # Protocol tokens must be delimited. In particular, standalone
            # LDP is MPLS evidence, while the substring inside LLDP is not.
            if re.fullmatch(r"[a-z0-9]+", pattern):
                matched = re.search(
                    rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])",
                    lowered,
                ) is not None
            else:
                matched = pattern in lowered
            if matched:
                families.add(family)
                break
    if any(marker in lowered for marker in DISCOVERY_MARKERS):
        families.add("discovery")
    return families or {"other"}


def action_semantics(action: dict[str, Any]) -> dict[str, Any]:
    command = str(action["command"])
    matches = PATH_PATTERN.findall(command.replace("\\\\", "\\"))
    targets = sorted(
        {
            f"{snapshot.lower()}/{device.lower()}/{normalize_filename(filename)}"
            for snapshot, device, filename in matches
            if not any(
                marker in value
                for value in (snapshot, device, filename)
                for marker in "*?["
            )
        }
    )
    snapshots = sorted({snapshot.lower() for snapshot, _, _ in matches})
    devices = sorted(
        {
            device.lower()
            for _, device, _ in matches
            if not any(marker in device for marker in "*?[")
        }
    )
    filenames = sorted({normalize_filename(filename) for _, _, filename in matches})
    snapshot_glob = any(
        any(marker in snapshot for marker in "*?[")
        for snapshot, _, _ in matches
    )
    device_glob = any(
        any(marker in device for marker in "*?[")
        for _, device, _ in matches
    )
    filename_glob = any(
        any(marker in filename for marker in "*?[")
        for _, _, filename in matches
    )
    # The command/file name describes the requested evidence class.  Do not
    # derive clustering features from the returned configuration body: a full
    # config often mentions every protocol and would make unrelated paths look
    # artificially identical.
    families = sorted(infer_families(command))
    if targets:
        keys = sorted(
            {
                f"{target}/{family}"
                for target in targets
                for family in families
                if family not in {"discovery", "other"}
            }
        )
    else:
        keys = sorted(f"unknown/{family}" for family in families if family != "other")
    return {
        "targets": targets,
        "snapshots": snapshots,
        "devices": devices,
        "filenames": filenames,
        "families": families,
        "keys": keys,
        "has_snapshot_glob": snapshot_glob,
        "has_device_glob": device_glob,
        "has_filename_glob": filename_glob,
        "has_path_glob": snapshot_glob or device_glob or filename_glob,
        "glob_scopes": [
            scope
            for scope, present in (
                ("snapshot", snapshot_glob),
                ("device", device_glob),
                ("filename", filename_glob),
            )
            if present
        ],
        "is_discovery": "discovery" in families,
    }


def question_snapshot(raw: dict[str, Any]) -> str:
    match = re.search(
        r"(CampusNetwork(?:-for-perf)?_\d+)",
        str(raw["source_record"]["question"]),
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"{raw['id']}: question does not name one CampusNetwork snapshot")
    return match.group(1).lower()


def fault_context(raw: dict[str, Any]) -> tuple[set[str], str]:
    devices, reasons = base.fault_parts(raw)
    if len(set(reasons)) != 1:
        raise ValueError(f"{raw['id']}: expected one merged fault type")
    return {device.lower() for device in devices}, reasons[0]


def action_priority(
    action: dict[str, Any], semantics: dict[str, Any], raw: dict[str, Any]
) -> float:
    families = set(semantics["families"])
    snapshots = set(semantics["snapshots"])
    target_snapshot = question_snapshot(raw)
    score = float(action.get("selection_score", 0.0)) / 20.0
    score += 5.0 * (target_snapshot in snapshots)
    score += 2.0 * bool(semantics["devices"])
    score += 2.0 * bool(families - LOW_VALUE_FAMILIES - {"discovery", "other"})
    # Full configuration dumps are useful fallbacks but extremely redundant.
    # Prefer equally relevant operational state without consulting the label;
    # endpoint recovery can still retain exact config evidence when required.
    score -= 2.0 * ("config" in families)
    score -= 8.0 * bool(snapshots and target_snapshot not in snapshots)
    score -= 8.0 * bool(semantics["has_path_glob"])
    score -= 7.0 * bool(families & LOW_VALUE_FAMILIES)
    score -= 4.0 * semantics["is_discovery"]
    return round(score, 6)


def stage_claim_units(text: str) -> set[str]:
    """Extract label-independent device/protocol claims from one source turn."""
    units: set[str] = set()
    # Scope device/family pairs to one local clause. A sentence such as
    # "check PE1 routing、PE2 interfaces" must not invent the Cartesian
    # pairs PE1/interface and PE2/routing.
    for sentence in re.split(r"[、，,；;。！？!?\n]+", text):
        devices = {
            value.lower()
            for value in re.findall(
                r"\b(?:PE\d|[A-Za-z]+_SW_\d+|FW_\d+)\b", sentence, re.IGNORECASE
            )
        }
        families = sentence_concepts(sentence) - devices
        units.update(f"device:{device}" for device in devices)
        units.update(f"family:{family}" for family in families)
        units.update(
            f"pair:{device}/{family}"
            for device in devices
            for family in families
        )
    return units


def action_claim_units(semantics: dict[str, Any]) -> set[str]:
    devices = set(semantics.get("devices", []))
    families = set(semantics.get("families", [])) - {"other", "discovery", "config"}
    return {
        *(f"device:{device}" for device in devices),
        *(f"family:{family}" for family in families),
        *(f"pair:{device}/{family}" for device in devices for family in families),
    }


def prune_actions(
    actions: list[dict[str, Any]], raw: dict[str, Any], stage_index: int,
    source_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evaluated: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for action in actions:
        semantics = action_semantics(action)
        evaluated.append((action_priority(action, semantics, raw), action, semantics))

    target_snapshot = question_snapshot(raw)
    same_snapshot = [
        item for item in evaluated
        if set(item[2].get("snapshots", [])) == {target_snapshot}
        and not item[2].get("has_path_glob")
    ]
    non_discovery = [item for item in same_snapshot if not item[2]["is_discovery"]]
    pool = non_discovery if non_discovery else same_snapshot
    if pool and all(item[2]["is_discovery"] for item in pool):
        pool = sorted(pool, key=lambda item: (-item[0], int(item[1]["event_line"])))[:1]
    if any(not (set(item[2]["families"]) & LOW_VALUE_FAMILIES) for item in pool):
        pool = [
            item
            for item in pool
            if not (set(item[2]["families"]) & LOW_VALUE_FAMILIES)
        ]

    deduplicated: dict[str, tuple[float, dict[str, Any], dict[str, Any]]] = {}
    for item in pool:
        score, action, semantics = item
        key = "|".join(semantics["keys"] or semantics["targets"])
        if not key:
            key = base.digest_text(str(action["command"]))[:20]
        previous = deduplicated.get(key)
        if previous is None or (score, -int(action["event_line"])) > (
            previous[0],
            -int(previous[1]["event_line"]),
        ):
            deduplicated[key] = item

    ranked_pool = sorted(
        deduplicated.values(),
        key=lambda item: (-item[0], int(item[1]["event_line"])),
    )
    required_units = stage_claim_units(source_text)
    coverable_units = set().union(
        *(action_claim_units(item[2]) for item in ranked_pool)
    ) if ranked_pool else set()
    uncovered = required_units & coverable_units
    ranked: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    remaining = list(ranked_pool)
    while uncovered and remaining:
        best = max(
            remaining,
            key=lambda item: (
                len(action_claim_units(item[2]) & uncovered),
                item[0],
                -int(item[1]["event_line"]),
            ),
        )
        gain = action_claim_units(best[2]) & uncovered
        if not gain:
            break
        ranked.append(best)
        uncovered -= gain
        remaining.remove(best)
    # A source turn with no extractable claim still needs one concrete,
    # highest-value action to teach a focused next step.  No numerical action
    # ceiling is used: the selected set is entirely determined by claim cover.
    if not ranked and ranked_pool:
        ranked = [ranked_pool[0]]
    kept_ids = {item[1]["item_id"] for item in ranked}
    kept: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    for score, action, semantics in evaluated:
        public = {
            "item_id": action["item_id"],
            "event_line": action["event_line"],
            "priority": score,
            "semantics": semantics,
        }
        if action["item_id"] in kept_ids:
            enriched = copy.deepcopy(action)
            source_command = str(action["command"])
            source_powershell_command = normalize_supervised_command(source_command)
            translation = translate_powershell_to_exec_command(source_powershell_command)
            enriched["source_command"] = source_command
            enriched["source_powershell_command"] = source_powershell_command
            enriched["source_powershell_command_sha256_lf_normalized"] = base.digest_text(
                source_powershell_command
            )
            enriched["command"] = translation["cmd"]
            enriched["justification"] = translation["justification"]
            enriched["command_translation"] = translation["kind"]
            enriched["command_protocol"] = "codex_cli.exec_command.arguments.cmd"
            enriched["command_was_translated"] = translation["cmd"] != source_powershell_command
            enriched["result_protocol"] = "codex_cli.function_call_output.linux_normalized"
            enriched["supervised_cmd_sha256_lf_normalized"] = base.digest_text(
                translation["cmd"]
            )
            enriched["causal_priority"] = score
            enriched["causal_semantics"] = semantics
            kept.append(enriched)
        else:
            omitted.append(public)
    kept.sort(key=lambda action: int(action["event_line"]))
    covered_after = set().union(
        *(action_claim_units(action["causal_semantics"]) for action in kept)
    ) if kept else set()
    coverage_before = required_units & coverable_units
    coverage_after = required_units & covered_after
    def coverage_status(covered: set[str]) -> str:
        if not required_units:
            return "unscoped"
        if not covered:
            return "zero"
        if covered == required_units:
            return "full"
        return "partial"
    return kept, omitted, {
        "policy": "minimal_label_independent_claim_cover",
        "coverage_scope": "source_conclusion_next_action_intent",
        "original_action_count": len(actions),
        "eligible_action_count": len(ranked_pool),
        "kept_action_count": len(kept),
        "required_claim_units": sorted(required_units),
        "covered_claim_units_before": sorted(coverage_before),
        "covered_claim_units_after": sorted(coverage_after),
        "claim_coverage_before": (
            round(len(coverage_before) / len(required_units), 6)
            if required_units else 1.0
        ),
        "claim_coverage_after": (
            round(len(coverage_after) / len(required_units), 6)
            if required_units else 1.0
        ),
        "claim_coverage_status_before": coverage_status(coverage_before),
        "claim_coverage_status_after": coverage_status(coverage_after),
        "claim_cover_retained_without_decrease": coverage_after == coverage_before,
    }


def sentence_concepts(text: str) -> set[str]:
    concepts = infer_families(text) - {"other", "discovery"}
    concepts.update(value.lower() for value in re.findall(r"\b(?:PE\d|[A-Za-z]+_SW_\d+)\b", text))
    return concepts


def action_description(actions: list[dict[str, Any]]) -> str:
    pairs: list[str] = []
    for action in actions:
        semantics = action["causal_semantics"]
        devices = semantics["devices"] or ["相关节点"]
        families = [
            family
            for family in semantics["families"]
            if family not in {"other", "config", "discovery"}
        ] or [
            family
            for family in semantics["families"]
            if family not in {"other", "discovery"}
        ]
        family = FAMILY_NAMES.get(families[0] if families else "other", "补充状态")
        for device in devices[:2]:
            value = f"{device} 的{family}"
            if value not in pairs:
                pairs.append(value)
    return "、".join(pairs[:4]) or "最有区分度的运行状态"


GROUNDING_TOKEN_PATTERN = re.compile(
    r"(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?|"
    r"\b(?:PE\d|[A-Za-z]+_SW_\d+|(?:Eth-?Trunk|Vlanif|Ethernet|GE)\d+(?:[/_.-]\d+)*)\b|"
    r"\b[A-Za-z][A-Za-z0-9_./:-]{2,}\b",
    re.IGNORECASE,
)
GROUNDING_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "current",
    "display", "saved_configs", "campusnetwork", "true", "false",
}
PROCEDURAL_MARKERS = (
    "我会", "我将", "我先", "下一步", "接下来", "需要", "继续", "准备", "计划",
    "先检查", "先查看", "验证是否", "确认是否", "排查", "读取", "搜索", "对比",
    "优先", "再", "随后", "然后",
)
ASSERTION_MARKERS = (
    "已经", "已确认", "已发现", "我已", "发现", "显示", "表明", "可见", "说明",
    "均有", "保存了", "启用了", "开启了", "关闭了", "收敛到", "因此", "所以",
    "根因是", "可以确定", "直接支持", "证明",
)
PROCEDURAL_PREFIX_PATTERN = re.compile(
    r"^(?:我会|我将|我先|下一步|接下来|需要|继续|准备|计划|先检查|先查看|"
    r"验证是否|确认是否|排查|读取|搜索|对比|优先|"
    r"(?:再|随后|然后)(?:我会|检查|查看|核对|验证|确认|排查|读取|搜索|对比))"
)
MIXED_CLAUSE_PATTERN = re.compile(r"(?<=[，,；;：:])\s*")


def sentence_grounding_terms(sentence: str) -> set[str]:
    return {
        value.lower()
        for value in GROUNDING_TOKEN_PATTERN.findall(sentence)
        if value.lower() not in GROUNDING_STOPWORDS
        and not value.lower().startswith("campusnetwork")
    }


def is_safe_procedural_sentence(sentence: str) -> bool:
    stripped = sentence.strip().lstrip("而并且且但、，,；;：: ")
    return (
        PROCEDURAL_PREFIX_PATTERN.match(stripped) is not None
        and any(marker in stripped for marker in PROCEDURAL_MARKERS)
        and not any(marker in sentence for marker in ASSERTION_MARKERS)
    )


def split_mixed_reasoning_sentence(sentence: str) -> list[str]:
    """Keep pure future plans separate from factual clauses.

    A wholly procedural sentence remains intact. A mixed or factual sentence
    is split only at clause punctuation; each resulting fact must then pass
    ordinary exact-observation grounding or be dropped.
    """
    value = sentence.strip()
    if not value:
        return []
    if is_safe_procedural_sentence(value):
        return [value]
    return [
        clause.strip()
        for clause in MIXED_CLAUSE_PATTERN.split(value)
        if clause.strip()
    ]


def ground_source_sentence(
    sentence: str, prior_actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Rewrite a factual source sentence into exact earlier observation atoms.

    The source sentence is used only to select relevant evidence. It is never
    supervised verbatim: command paths, empty results, headers, and truncated
    configuration omissions cannot support a positive factual assertion.
    """
    if is_safe_procedural_sentence(sentence):
        return {
            "kind": "procedural_plan",
            "sentence": sentence,
            "source_sentence": sentence,
            "action_ids": [],
            "claim_bindings": [],
        }
    sentence_terms = sentence_grounding_terms(sentence)
    sentence_devices = {
        value.lower()
        for value in re.findall(
            r"\b(?:PE\d|[A-Za-z]+_SW_\d+|FW_\d+)\b", sentence, re.IGNORECASE
        )
    }
    content_terms = {
        term for term in sentence_terms
        if term not in sentence_devices
        and term not in {
            "isis", "ldp", "mpls", "lsp", "bgp", "vrrp", "stp", "mst",
            "route", "routing", "interface", "status", "current", "normal",
        }
    }
    sentence_families = infer_families(sentence) - {"other", "discovery", "config"}
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for action in prior_actions:
        semantics = action.get("causal_semantics", {})
        family_overlap = sentence_families & (
            set(semantics.get("families", [])) - {"other", "discovery", "config"}
        )
        device_overlap = sentence_devices & set(semantics.get("devices", []))
        for line_index, source_line in enumerate(
            visible_action_output(action, compact=True).splitlines()
        ):
            span = re.sub(r"\s+", " ", source_line).strip()
            if not is_substantive_observation_line(span):
                continue
            lowered = span.lower()
            matched_terms = sorted(
                term for term in content_terms
                if observation_contains_anchor(lowered, term)
            )
            if not matched_terms and not family_overlap:
                continue
            score = (
                4.0 * len(matched_terms)
                + 2.0 * len(family_overlap)
                + 0.5 * len(device_overlap)
            )
            candidates.append((score, -int(action["event_line"]), -line_index, {
                "action_id": str(action["action_id"]),
                "observation_span": span[:240],
                "matched_content_terms": matched_terms,
                "matched_families": sorted(family_overlap),
            }))
    if not candidates:
        return None
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, _, _, evidence in sorted(candidates, reverse=True):
        key = (evidence["action_id"], evidence["observation_span"])
        if key in seen:
            continue
        selected.append(evidence)
        seen.add(key)
        if len(selected) == 2:
            break
    claims = [f"回显“{item['observation_span']}”" for item in selected]
    supervised_sentence = (
        f"已核对此前可见回显：{'；'.join(claims)}。"
        "当前只保留这些直接显示的状态；未显示的路径、地址归属或配置缺失不作推断。"
    )
    return {
        "kind": "observation_bound_reasoning",
        "sentence": supervised_sentence,
        "source_sentence": sentence,
        "action_ids": list(dict.fromkeys(item["action_id"] for item in selected)),
        "claim_bindings": [
            {
                "claim": claim,
                "evidence": [{
                    "action_id": evidence["action_id"],
                    "observation_span": evidence["observation_span"],
                }],
                "matched_content_terms": evidence["matched_content_terms"],
                "matched_families": evidence["matched_families"],
            }
            for claim, evidence in zip(claims, selected)
        ],
    }


def evidence_observation(action: dict[str, Any], raw: dict[str, Any]) -> str:
    """Select one compact observation without consulting the verified label."""
    semantics = action["causal_semantics"]
    keywords = {
        *(str(value).lower() for value in semantics["devices"]),
        *(str(value).lower() for value in semantics["families"]),
        *(str(value).lower() for value in semantics["filenames"]),
    }
    status_markers = {
        "disable", "disabled", "enable", "enabled", "down", "up", "error",
        "inactive", "active", "unreachable", "reachable", "not", " no ",
        "best", "valid", "select", "advertised", "cost", "preference", "localpref",
        "未", "无", "关闭", "开启", "故障", "异常", "正常", "丢包", "时延",
    }
    candidates: list[tuple[float, int, str]] = []
    for index, source_line in enumerate(
        visible_action_output(action, compact=True).splitlines()
    ):
        line = re.sub(r"\s+", " ", source_line).strip()
        if (
            not line
            or len(line) < 3
            or re.fullmatch(r"[-=+|<>\s]+", line)
            or line.startswith(("(ed):", "...", "Legend:", "Flags:"))
            or re.match(r"^\([^)]+\):", line)
            or line.lower().startswith(("display ", "route flags:"))
            or line.startswith(("Chunk ID:", "Wall time:", "Process exited", "Original token count:", "Output:"))
        ):
            continue
        lowered = f" {line.lower()} "
        score = 0.0
        score += 4.0 * sum(keyword in lowered for keyword in keywords if keyword)
        score += 2.5 * sum(marker in lowered for marker in status_markers)
        if re.search(r"\b(?:true|false|yes|no|0|1)\b", lowered):
            score += 1.0
        if line.startswith(("<", "[", "...")):
            score -= 1.0
        candidates.append((score, -index, line[:180]))
    if candidates:
        return max(candidates)[2]
    target = str(action.get("target") or action.get("command") or "该项检查")
    return f"{target[:160]} 已成功返回可复核结果"


def grounded_observation_description(
    actions: list[dict[str, Any]], raw: dict[str, Any]
) -> str:
    observations: list[str] = []
    for action in actions[:2]:
        observations.append(
            f"{action_description([action])}回显“{evidence_observation(action, raw)}”"
        )
    return "；".join(observations)


def visible_action_output(action: dict[str, Any], *, compact: bool) -> str:
    """Return exactly the normalized output that is visible before an endpoint."""
    output = str(action.get("output_excerpt", ""))
    if compact and len(output) > base.HISTORICAL_OUTPUT_LIMIT:
        output = (
            output[: base.HISTORICAL_OUTPUT_LIMIT].rstrip()
            + "\n... [historical excerpt truncated] ..."
        )
    return normalize_tool_output_for_exec_command(action, output)


def output_line_device(
    line: str, action: dict[str, Any], target_snapshot: str
) -> str | None:
    match = re.search(
        rf"saved_configs/{re.escape(target_snapshot)}/([^/:\s]+)(?:/|:)",
        line,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).lower()
    devices = list(action["causal_semantics"].get("devices", []))
    return devices[0] if len(devices) == 1 else None


def evidence_fact(
    *, kind: str, action: dict[str, Any], snapshot: str, device: str,
    observation: str, **attributes: Any
) -> dict[str, Any]:
    return {
        "kind": kind,
        "action_id": str(action["action_id"]),
        "snapshot": snapshot,
        "device": device,
        "observation": observation,
        **attributes,
    }


def question_source_host(raw: dict[str, Any]) -> str | None:
    """Return the source host named immediately before the question's ping verb."""
    question = str(raw.get("source_record", {}).get("question") or "")
    # The archived GBK-to-UTF8 mojibake may place a Unicode letter/number directly
    # after the ASCII word "ping", so a trailing Unicode word boundary is unsafe.
    match = re.search(r"\b([A-Za-z][A-Za-z0-9_-]+)\s+ping", question, re.IGNORECASE)
    return match.group(1).lower() if match else None


def campus_vlan_from_host_address(address: str, prefix_length: int | None) -> int | None:
    """Map the topology's 10.1.<VLAN>.0/24 host subnet to its VLAN ID."""
    octets = address.split(".")
    if len(octets) != 4 or octets[:2] != ["10", "1"] or prefix_length != 24:
        return None
    vlan = int(octets[2])
    return vlan if 1 <= vlan <= 4094 else None


def extract_direct_evidence_facts(
    evidence_stages: list[dict[str, Any]], raw: dict[str, Any], *, compact_history: bool
) -> list[dict[str, Any]]:
    """Extract label-independent atomic facts from the endpoint-visible history."""
    target_snapshot = question_snapshot(raw)
    facts: list[dict[str, Any]] = []
    last_stage_index = len(evidence_stages) - 1
    for stage_index, stage in enumerate(evidence_stages):
        compact = compact_history and stage_index < last_stage_index
        for action in stage["actions"]:
            semantics = action["causal_semantics"]
            snapshots = set(semantics.get("snapshots", []))
            if (
                semantics.get("has_path_glob")
                or snapshots != {target_snapshot}
            ):
                continue
            output = visible_action_output(action, compact=compact)
            source_host = question_source_host(raw)
            for raw_line in output.splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if (
                    not line
                    or "historical excerpt truncated" in line
                    or re.fullmatch(r"[-=+|<>\s]+", line)
                ):
                    continue
                device = output_line_device(line, action, target_snapshot)
                if not device:
                    continue
                lowered = line.lower()
                if "vrrp" in semantics.get("families", set()):
                    interface_context = re.search(r"\bvlanif\s*(\d+)\b", lowered)
                    if interface_context:
                        virtual_router = re.search(r"\bvirtual\s+router\s+(\d+)\b", lowered)
                        facts.append(evidence_fact(
                            kind="vrrp_interface_context", action=action,
                            snapshot=target_snapshot, device=device, observation=line,
                            vlan=int(interface_context.group(1)),
                            vrid=(int(virtual_router.group(1)) if virtual_router else None),
                        ))
                if re.search(r"protocol status\s*:?\s*disabled\b", lowered):
                    facts.append(evidence_fact(
                        kind="stp_global_disabled", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                    ))
                if "stp port bpdu-filter enable" in lowered:
                    facts.append(evidence_fact(
                        kind="stp_bpdu_filter_enabled", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                    ))
                if (
                    re.search(r"\bpreempt\s*:\s*no\b", lowered)
                    or re.search(r"\bvrrp\s+vrid\s+\d+\s+preempt\s+disable\b", lowered)
                ):
                    facts.append(evidence_fact(
                        kind="vrrp_preempt_disabled", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                    ))
                master_summary = re.search(
                    r"^(\d+)\s+master\s+vlanif\s*(\d+)\b", lowered
                )
                if re.search(r"\bstate\s*:\s*master\b", lowered) or master_summary:
                    facts.append(evidence_fact(
                        kind="vrrp_master", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                        vlan=(int(master_summary.group(2)) if master_summary else None),
                        vrid=(int(master_summary.group(1)) if master_summary else None),
                    ))
                alternate = re.search(
                    r"^(\d+)\s+\S+\s+alte\b.*\bdiscarding\b", lowered
                )
                if alternate:
                    facts.append(evidence_fact(
                        kind="stp_alternate_discarding", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                        instance=int(alternate.group(1)),
                    ))

                mapping = re.search(r"\binstance\s+(\d+)\s+vlan\s+(.+)$", lowered)
                if mapping:
                    vlans = sorted({
                        int(value) for value in re.findall(r"\b\d+\b", mapping.group(2))
                        if 1 <= int(value) <= 4094
                    })
                    if vlans:
                        facts.append(evidence_fact(
                            kind="mst_vlan_instance_mapping", action=action,
                            snapshot=target_snapshot, device=device, observation=line,
                            instance=int(mapping.group(1)), vlans=vlans,
                        ))

                host_address = re.search(
                    r"\binet(?:\s+addr:|\s+)((?:\d{1,3}\.){3}\d{1,3})"
                    r"(?:/(\d{1,2}))?\b",
                    lowered,
                )
                if host_address and source_host and device == source_host:
                    prefix_length = (
                        int(host_address.group(2)) if host_address.group(2) else None
                    )
                    vlan = campus_vlan_from_host_address(
                        host_address.group(1), prefix_length
                    )
                    if vlan is not None:
                        facts.append(evidence_fact(
                            kind="source_host_ipv4", action=action,
                            snapshot=target_snapshot, device=device, observation=line,
                            address=host_address.group(1), prefix_length=prefix_length,
                            vlan=vlan,
                        ))

                interface_address = re.search(
                    r"\bip\s+address\s+((?:\d{1,3}\.){3}\d{1,3})"
                    r"(?:\s+((?:\d{1,3}\.){3}\d{1,3}|\d{1,2}))?",
                    lowered,
                )
                if interface_address and "route-static" not in lowered:
                    facts.append(evidence_fact(
                        kind="interface_ipv4_address", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                        address=interface_address.group(1),
                        mask=interface_address.group(2),
                    ))

                ips = re.findall(r"(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?", line)
                if "ip route-static" in lowered and len(ips) >= 2:
                    facts.append(evidence_fact(
                        kind="ip_static_route", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                        prefix=ips[0].split("/", 1)[0], next_hop=ips[-1].split("/", 1)[0],
                    ))
                elif re.search(r"(?:^|:)\s*(?:\d{1,3}\.){3}\d{1,3}/\d+\s+static\b", lowered) and len(ips) >= 2:
                    facts.append(evidence_fact(
                        kind="ip_static_route", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                        prefix=ips[0].split("/", 1)[0], next_hop=ips[-1].split("/", 1)[0],
                    ))

                static_lsp = re.search(
                    r"\bstatic-lsp\s+(?:ingress|transit)\s+\S+.*?"
                    r"in-label\s+(\d+).*?nexthop\s+((?:\d{1,3}\.){3}\d{1,3})"
                    r".*?out-label\s+(\d+)",
                    lowered,
                )
                if static_lsp:
                    facts.append(evidence_fact(
                        kind="mpls_static_lsp_hop", action=action,
                        snapshot=target_snapshot, device=device, observation=line,
                        in_label=int(static_lsp.group(1)),
                        next_hop=static_lsp.group(2),
                        out_label=int(static_lsp.group(3)),
                    ))
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for fact in facts:
        key = (
            fact["kind"], fact["action_id"], fact["device"], fact["observation"],
            fact.get("prefix"), fact.get("next_hop"), fact.get("in_label"),
            fact.get("out_label"), fact.get("address"),
            fact.get("vlan"), fact.get("vrid"), fact.get("instance"),
            tuple(fact.get("vlans", [])),
        )
        unique.setdefault(key, fact)
    return list(unique.values())


def select_cycle_facts(
    facts: list[dict[str, Any]], *, mpls: bool
) -> list[dict[str, Any]] | None:
    """Prove a three-device cycle from visible forwarding and IP ownership facts."""
    kind = "mpls_static_lsp_hop" if mpls else "ip_static_route"
    candidates = [fact for fact in facts if fact["kind"] == kind]
    ownership_facts = [
        fact for fact in facts if fact["kind"] == "interface_ipv4_address"
    ]
    owners_by_address: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in ownership_facts:
        owners_by_address[str(fact["address"])].append(fact)
    if not mpls:
        prefixes = sorted({str(fact["prefix"]) for fact in candidates})
        grouped = [
            [fact for fact in candidates if fact["prefix"] == prefix]
            for prefix in prefixes
        ]
    else:
        grouped = [candidates]
    for group in grouped:
        by_device: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fact in group:
            owners = [
                owner for owner in owners_by_address.get(str(fact["next_hop"]), [])
                if owner["device"] != fact["device"]
            ]
            if len({owner["device"] for owner in owners}) == 1:
                enriched = copy.deepcopy(fact)
                enriched["next_hop_owner"] = owners[0]["device"]
                enriched["next_hop_owner_fact"] = owners[0]
                by_device[str(fact["device"])].append(enriched)
        for devices in itertools.combinations(sorted(by_device), 3):
            for selected_tuple in itertools.product(*(by_device[device] for device in devices)):
                selected = list(selected_tuple)
                successor = {
                    str(fact["device"]): str(fact["next_hop_owner"])
                    for fact in selected
                }
                if set(successor.values()) != set(devices):
                    continue
                start = devices[0]
                if successor.get(successor.get(successor.get(start, ""), ""), "") != start:
                    continue
                if any(successor[device] == device for device in devices):
                    continue
                ordered = [next(fact for fact in selected if fact["device"] == start)]
                ordered.append(next(fact for fact in selected if fact["device"] == successor[start]))
                ordered.append(
                    next(
                        fact for fact in selected
                        if fact["device"] == successor[ordered[-1]["device"]]
                    )
                )
                if len({fact["device"] for fact in ordered}) != 3:
                    continue
                if mpls and any(
                    ordered[index]["out_label"]
                    != ordered[(index + 1) % 3]["in_label"]
                    for index in range(3)
                ):
                    continue
                owner_facts = [
                    copy.deepcopy(fact["next_hop_owner_fact"]) for fact in ordered
                ]
                for fact in ordered:
                    fact.pop("next_hop_owner_fact", None)
                return [*ordered, *owner_facts]
    return None


def endpoint_evidence_gate(
    evidence_stages: list[dict[str, Any]], raw: dict[str, Any], *, compact_history: bool = True
) -> dict[str, Any]:
    """Validate the verified label after extracting facts without using it."""
    facts = extract_direct_evidence_facts(
        evidence_stages, raw, compact_history=compact_history
    )
    fault_devices, reason = fault_context(raw)
    selected: list[dict[str, Any]] | None = None
    rule = ""
    if reason == "全局STP未使能":
        rule = "same_snapshot_fault_device_protocol_status_disabled"
        selected = [
            fact for fact in facts
            if fact["kind"] == "stp_global_disabled" and fact["device"] in fault_devices
        ][:1]
    elif reason == "STP BPDU被过滤":
        rule = "same_snapshot_fault_device_bpdu_filter_enable"
        selected = [
            fact for fact in facts
            if fact["kind"] == "stp_bpdu_filter_enabled" and fact["device"] in fault_devices
        ][:1]
    elif reason == "VRRP工作在非抢占模式":
        rule = "same_snapshot_fault_device_preempt_no_or_disable"
        selected = [
            fact for fact in facts
            if fact["kind"] == "vrrp_preempt_disabled" and fact["device"] in fault_devices
        ][:1]
    elif reason == "VRRP Master角色规划不合理":
        rule = "same_snapshot_source_vlan_vrrp_mst_instance_role_misalignment"
        source_host = question_source_host(raw)
        host_facts = [
            fact for fact in facts
            if fact["kind"] == "source_host_ipv4" and fact["device"] == source_host
        ]
        selected_closure: list[dict[str, Any]] = []
        closed_devices: set[str] = set()
        for device in sorted(fault_devices):
            closure: list[dict[str, Any]] | None = None
            for host_fact in host_facts:
                vlan = int(host_fact["vlan"])
                mappings = [
                    fact for fact in facts
                    if fact["kind"] == "mst_vlan_instance_mapping"
                    and fact["device"] == device and vlan in fact.get("vlans", [])
                ]
                alternates = [
                    fact for fact in facts
                    if fact["kind"] == "stp_alternate_discarding"
                    and fact["device"] == device
                ]
                for master in (
                    fact for fact in facts
                    if fact["kind"] == "vrrp_master" and fact["device"] == device
                ):
                    master_context: dict[str, Any] | None = None
                    master_vlan = master.get("vlan")
                    if master_vlan is None:
                        contexts = [
                            fact for fact in facts
                            if fact["kind"] == "vrrp_interface_context"
                            and fact["device"] == device
                            and fact["action_id"] == master["action_id"]
                        ]
                        context_vlans = {int(fact["vlan"]) for fact in contexts}
                        if len(context_vlans) == 1:
                            # Prefer a real VRRP status line over a command echo when
                            # both expose the same Vlanif context.
                            master_context = max(
                                contexts,
                                key=lambda fact: (
                                    "virtual router" in str(
                                        fact.get("observation") or ""
                                    ).lower(),
                                    not str(fact.get("observation") or "")
                                    .lower()
                                    .startswith("display "),
                                ),
                            )
                            master_vlan = int(master_context["vlan"])
                    if master_vlan != vlan:
                        continue
                    for mapping in mappings:
                        alternate = next(
                            (
                                fact for fact in alternates
                                if int(fact.get("instance", -1))
                                == int(mapping["instance"])
                            ),
                            None,
                        )
                        if alternate:
                            closure = [
                                host_fact,
                                master,
                                *([master_context] if master_context else []),
                                mapping,
                                alternate,
                            ]
                            break
                    if closure:
                        break
                if closure:
                    break
            if closure:
                selected_closure.extend(closure)
                closed_devices.add(device)
        if closed_devices == fault_devices:
            unique_closure: dict[tuple[str, str, str], dict[str, Any]] = {}
            for fact in selected_closure:
                unique_closure.setdefault(
                    (str(fact["kind"]), str(fact["action_id"]), str(fact["observation"])),
                    fact,
                )
            selected = list(unique_closure.values())
    elif reason == "存在IP路由环路":
        rule = "three_device_same_prefix_static_route_next_hop_cycle"
        selected = select_cycle_facts(facts, mpls=False)
    elif reason == "存在MPLS标签环路":
        rule = "three_device_static_lsp_next_hop_and_label_cycle"
        selected = select_cycle_facts(facts, mpls=True)
    else:
        raise ValueError(f"{raw['id']}: unsupported endpoint gate reason {reason}")
    selected = selected or []
    covered_devices = (
        sorted(fault_devices)
        if reason == "VRRP Master角色规划不合理" and selected
        else sorted({fact["device"] for fact in selected})
    )
    passed = bool(selected) and set(covered_devices) == fault_devices
    return {
        "passed": passed,
        "gate_rule": rule,
        "target_snapshot": question_snapshot(raw),
        "expected_fault_device_count": len(fault_devices),
        "covered_devices": covered_devices,
        "selected_facts": selected if passed else [],
        "extracted_fact_count": len(facts),
        "verified_label_used_only_for_posthoc_gate": True,
    }


def endpoint_fact_description(fact: dict[str, Any]) -> str:
    names = {
        "stp_global_disabled": "全局 STP 运行状态",
        "stp_bpdu_filter_enabled": "端口 BPDU 过滤配置",
        "vrrp_preempt_disabled": "VRRP 抢占状态",
        "vrrp_master": "VRRP 当前角色",
        "vrrp_interface_context": "VRRP 三层接口上下文",
        "stp_alternate_discarding": "STP 转发路径",
        "mst_vlan_instance_mapping": "MST VLAN/实例映射",
        "source_host_ipv4": "源主机 IPv4/VLAN 归属",
        "ip_static_route": "静态路由下一跳",
        "mpls_static_lsp_hop": "静态 LSP 标签转发",
        "interface_ipv4_address": "接口地址归属",
    }
    return f"{fact['device']} 的{names[fact['kind']]}回显“{fact['observation']}”"


def optimize_stage_text(
    original_text: str,
    original_thinking: str,
    original_conclusion: str,
    actions: list[dict[str, Any]],
    prior_actions: list[dict[str, Any]],
    raw: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    action_concepts = set()
    action_devices = set()
    for action in actions:
        action_concepts.update(action["causal_semantics"]["families"])
        action_devices.update(action["causal_semantics"]["devices"])
    question_devices = {
        value.lower()
        for value in re.findall(
            r"\b(?:PE\d|[A-Za-z]+_SW_\d+)\b",
            str(raw["source_record"]["question"]),
        )
    }
    relevant_terms = action_concepts | action_devices | question_devices
    raw_source_sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(original_thinking)
        if sentence.strip()
    ]
    source_sentences = [
        clause
        for sentence in raw_source_sentences
        for clause in split_mixed_reasoning_sentence(sentence)
    ]
    grounded_records: list[dict[str, Any]] = []
    dropped_sentences: list[str] = []
    for sentence in source_sentences:
        relevant = (
            any(term in sentence.lower() for term in relevant_terms)
            or any(marker in sentence for marker in ("证据", "排除", "假设", "路径", "下一步"))
        )
        grounding = ground_source_sentence(sentence, prior_actions) if relevant else None
        if grounding is None:
            dropped_sentences.append(sentence)
            continue
        grounded_records.append(grounding)
        if len(grounded_records) == 4:
            break
    kept_sentences = [str(record["sentence"]) for record in grounded_records]
    if kept_sentences:
        thinking = "".join(kept_sentences)
    else:
        thinking = (
            "当前证据尚不足以完成最小归因，需要沿已出现的业务路径继续取证，"
            "并优先检查能够区分剩余候选的状态。"
        )

    conclusion_grounding = ground_source_sentence(original_conclusion, prior_actions)
    conclusion_overlap = sentence_concepts(original_conclusion) & action_concepts
    selected_action_units = set().union(
        *(action_claim_units(action["causal_semantics"]) for action in actions)
    ) if actions else set()
    conclusion_units = stage_claim_units(original_conclusion)
    conclusion_fully_action_aligned = bool(
        conclusion_units and conclusion_units <= selected_action_units
    )
    if (
        actions
        and conclusion_overlap
        and conclusion_fully_action_aligned
        and is_safe_procedural_sentence(original_conclusion)
    ):
        conclusion = original_conclusion
        conclusion_source = "original_visible_conclusion"
    elif actions:
        conclusion = (
            f"下一步核对{action_description(actions)}，用于区分剩余候选；"
            "在结果返回前不提前输出最终答案。"
        )
        conclusion_source = "evidence_aligned_reconstruction"
    else:
        conclusion = "当前节点没有新的可验证动作，不提前作事实结论。"
        conclusion_source = "evidence_aligned_reconstruction"

    if any(
        record["kind"] == "observation_bound_reasoning"
        for record in grounded_records
    ):
        thinking_source = "observation_bound_reasoning_reconstruction"
    elif kept_sentences:
        thinking_source = "pruned_original_visible_agent_message"
    else:
        thinking_source = "fixed_bridge_template"

    return thinking, conclusion, {
        "source_message_sha256_lf_normalized": base.digest_text(original_text),
        "source_thinking_sentence_count": len(source_sentences),
        "source_thinking_raw_sentence_count": len(raw_source_sentences),
        "source_thinking_raw_sentences": raw_source_sentences,
        "source_thinking_sentences": source_sentences,
        "retained_thinking_sentence_count": len(kept_sentences),
        "retained_thinking_sentence_records": grounded_records,
        "dropped_unsupported_thinking_sentences": dropped_sentences,
        "conclusion_grounding": conclusion_grounding,
        "source_conclusion_required_units": sorted(conclusion_units),
        "source_conclusion_action_covered_units": sorted(
            conclusion_units & selected_action_units
        ),
        "source_conclusion_fully_action_aligned": conclusion_fully_action_aligned,
        "thinking_source": thinking_source,
        "conclusion_source": conclusion_source,
        "hidden_chain_of_thought_claimed": False,
    }


def stage_feature_set(stage: dict[str, Any]) -> set[str]:
    features: set[str] = set()
    for action in stage["actions"]:
        semantics = action["causal_semantics"]
        features.update(semantics["keys"])
    features.update(f"concept:{value}" for value in sentence_concepts(stage["conclusion"]))
    return features


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def label_independent_message_keywords(text: str) -> set[str]:
    lowered = text.lower()
    values = {
        value.lower()
        for value in re.findall(
            r"(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?|"
            r"\b(?:PE\d|[A-Za-z]+_SW_\d+|[A-Za-z][A-Za-z0-9_.:/-]{2,})\b",
            text,
            re.IGNORECASE,
        )
        if "campusnetwork" not in value.lower()
    }
    values.update(
        term for term in base.PROTOCOL_TERMS if term in lowered
    )
    values.update(
        term
        for term in {
            "disabled", "enabled", "preempt", "master", "backup", "nexthop",
            "next-hop", "route-static", "static-lsp", "bpdu-filter",
        }
        if term in lowered
    )
    return {value for value in values if len(value) >= 2}


def source_command_score(
    command: dict[str, Any], visible_text: str, raw: dict[str, Any]
) -> float:
    if command["status"] != "completed" or command["exit_code"] not in (0, None):
        return float("-inf")
    semantics = action_semantics({"command": str(command["command"])})
    target_snapshot = question_snapshot(raw)
    searchable = f"{command['command']}\n{str(command.get('output', ''))[:4000]}".lower()
    terms = label_independent_message_keywords(visible_text)
    score = 5.0 * (set(semantics.get("snapshots", [])) == {target_snapshot})
    score += 2.0 * bool(semantics.get("devices"))
    score += 2.0 * bool(
        set(semantics.get("families", [])) - LOW_VALUE_FAMILIES - {"other", "discovery"}
    )
    score += min(6.0, 0.75 * sum(term in searchable for term in terms))
    score -= 6.0 * bool(semantics.get("has_path_glob"))
    score -= 4.0 * bool(semantics.get("is_discovery"))
    score -= 6.0 * bool(set(semantics.get("families", [])) & LOW_VALUE_FAMILIES)
    return round(score, 6)


def build_label_independent_stage_data(
    raw: dict[str, Any],
    messages: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build source stages without reading fault devices or verified reasons."""
    stages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        is_final = index == len(messages) - 1
        if is_final:
            thinking = (
                "前序证据已经完成收敛；按最小根因集合原则，只输出被证据直接支持的故障项。"
            )
            conclusion = base.normalize_text(str(raw["final_answer"]))
            selected_commands: list[dict[str, Any]] = []
        else:
            thinking, conclusion = base.split_stage_message(message["text"])
            lower = int(message["event_line"])
            upper = int(messages[index + 1]["event_line"])
            # Do not use the final answer text to rank the final investigative
            # segment. Earlier next messages are visible source reasoning.
            visible_text = str(message["text"])
            if index + 1 < len(messages) - 1:
                visible_text += "\n" + str(messages[index + 1]["text"])
            segment = [
                command for command in commands
                if lower < int(command["event_line"]) < upper
                and command["status"] == "completed"
                and command["exit_code"] in (0, None)
            ]
            unique: dict[str, dict[str, Any]] = {}
            for command in segment:
                unique.setdefault(str(command["command"]), command)
            selected_commands = sorted(
                unique.values(),
                key=lambda command: (
                    -source_command_score(command, visible_text, raw),
                    int(command["event_line"]),
                ),
            )
        keywords = label_independent_message_keywords(
            str(message["text"]) if is_final else visible_text
        )
        actions: list[dict[str, Any]] = []
        for action_index, command in enumerate(selected_commands, 1):
            excerpt, excerpted = base.excerpt_output(
                str(command["output"]), keywords, base.CURRENT_OUTPUT_LIMIT
            )
            actions.append(
                {
                    "action_id": f"S{index + 1:02d}-A{action_index:02d}",
                    "item_id": command["item_id"],
                    "event_line": command["event_line"],
                    "command": command["command"],
                    "output_excerpt": excerpt,
                    "output_is_excerpt": excerpted,
                    "exit_code": command["exit_code"],
                    "status": command["status"],
                    "selection_score": source_command_score(command, visible_text, raw),
                    "target": base.command_target(str(command["command"])),
                    "command_sha256_lf_normalized": base.digest_text(str(command["command"])),
                    "full_output_sha256_lf_normalized": base.digest_text(str(command["output"])),
                    "excerpt_sha256_lf_normalized": base.digest_text(excerpt),
                }
            )
        stages.append(
            {
                "source_message_index": index,
                "source_message_event_line": message["event_line"],
                "source_message_item_id": message["item_id"],
                "thinking": thinking,
                "conclusion": conclusion,
                "is_final": is_final,
                "actions": actions,
            }
        )
    return stages


def prepare_stages(
    raw: dict[str, Any], messages: list[dict[str, Any]], commands: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_stages = build_label_independent_stage_data(raw, messages, commands)
    retained: list[dict[str, Any]] = []
    prior_retained_actions: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    for source_stage in source_stages:
        stage = copy.deepcopy(source_stage)
        source_index = int(stage["source_message_index"])
        if stage["is_final"]:
            stage["text_optimization"] = {
                "thinking_source": "minimal_final_bridge",
                "conclusion_source": "verified_final_answer",
                "hidden_chain_of_thought_claimed": False,
                "source_message_sha256_lf_normalized": base.digest_text(messages[source_index]["text"]),
            }
            stage["causal_features"] = []
            retained.append(stage)
            continue

        kept_actions, omitted_actions, action_selection = prune_actions(
            stage["actions"], raw, source_index, stage["conclusion"]
        )
        stage["actions"] = kept_actions
        thinking, conclusion, optimization = optimize_stage_text(
            messages[source_index]["text"],
            stage["thinking"],
            stage["conclusion"],
            kept_actions,
            prior_retained_actions,
            raw,
        )
        stage["thinking"] = thinking
        stage["conclusion"] = conclusion
        stage["text_optimization"] = optimization
        selected_action_units = set().union(
            *(
                action_claim_units(action["causal_semantics"])
                for action in kept_actions
            )
        ) if kept_actions else set()
        supervised_intent_units = stage_claim_units(conclusion)
        supervised_covered_units = supervised_intent_units & selected_action_units
        if not supervised_intent_units:
            supervised_status = "unscoped"
        elif not supervised_covered_units:
            supervised_status = "zero"
        elif supervised_covered_units == supervised_intent_units:
            supervised_status = "full"
        else:
            supervised_status = "partial"
        action_selection.update({
            "supervised_intent_scope": "final_supervised_conclusion",
            "supervised_intent_required_units": sorted(supervised_intent_units),
            "supervised_intent_covered_units": sorted(supervised_covered_units),
            "supervised_intent_coverage": (
                round(len(supervised_covered_units) / len(supervised_intent_units), 6)
                if supervised_intent_units else 1.0
            ),
            "supervised_intent_coverage_status": supervised_status,
            "selected_action_bindings": [
                {
                    "action_id": str(action["action_id"]),
                    "claim_units": sorted(action_claim_units(action["causal_semantics"])),
                }
                for action in kept_actions
            ],
        })
        stage["action_selection"] = action_selection
        stage["omitted_actions"] = omitted_actions
        stage["causal_features"] = sorted(stage_feature_set(stage))

        if not kept_actions:
            omissions.append(
                {
                    "source_message_index": source_index,
                    "reason": "no_retained_causal_action; stop supervision is endpoint-only",
                    "omitted_action_count": len(omitted_actions),
                }
            )
            continue
        retained.append(stage)
        prior_retained_actions.extend(kept_actions)

    if not retained or not retained[-1]["is_final"]:
        raise ValueError(f"{raw['id']}: final stage was lost")
    if len(retained) == 1:
        raise ValueError(f"{raw['id']}: no causal checkpoint survived pruning")
    return retained, {
        "source_checkpoint_count": len(source_stages),
        "retained_prefix_checkpoint_count": len(retained),
        "omitted_checkpoint_count": len(omissions),
        "omitted_checkpoints": omissions,
    }


def trajectory_path_features(stages: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    stage_sets = [set(stage.get("causal_features", [])) for stage in stages if not stage["is_final"]]
    union = set().union(*stage_sets) if stage_sets else set()
    union = {feature for feature in union if not feature.endswith("/cpu") and not feature.endswith("/memory")}
    transitions: set[str] = set()
    previous: set[str] | None = None
    for current in stage_sets:
        families = {feature.rsplit("/", 1)[-1] for feature in current if not feature.startswith("concept:")}
        if previous is not None:
            prior_families = {feature.rsplit("/", 1)[-1] for feature in previous if not feature.startswith("concept:")}
            transitions.update(f"{left}>{right}" for left in prior_families for right in families)
        previous = current
    return union, transitions


def trajectory_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    feature_score = jaccard(left["path_features"], right["path_features"])
    transition_score = jaccard(left["path_transitions"], right["path_transitions"])
    return round(0.82 * feature_score + 0.18 * transition_score, 6)


def choose_medoid(members: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = []
    for candidate in members:
        average = statistics.mean(
            trajectory_similarity(candidate, other) for other in members
        )
        ranked.append((average, float(candidate["quality"]["score"]), candidate))
    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            int(item[2]["raw"]["attempt_index"]),
            int(item[2]["raw"]["success_slot"]),
        )
    )
    return ranked[0][2]


def cluster_trajectories(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            -float(item["quality"]["score"]),
            int(item["raw"]["attempt_index"]),
            int(item["raw"]["success_slot"]),
        ),
    )
    clusters: list[dict[str, Any]] = []
    for candidate in ordered:
        similarities = [trajectory_similarity(candidate, cluster["medoid"]) for cluster in clusters]
        if similarities and max(similarities) >= TRAJECTORY_CLUSTER_THRESHOLD:
            cluster_index = max(
                range(len(clusters)),
                key=lambda index: (similarities[index], len(clusters[index]["members"]), -index),
            )
            clusters[cluster_index]["members"].append(candidate)
            clusters[cluster_index]["medoid"] = choose_medoid(clusters[cluster_index]["members"])
        else:
            clusters.append({"members": [candidate], "medoid": candidate})

    for cluster in clusters:
        # The medoid defines cluster geometry, but a training representative
        # should be the best grounded member rather than merely the most
        # geometrically central member.
        cluster["representative"] = sorted(
            cluster["members"],
            key=lambda item: (
                -float(item["quality"]["score"]),
                len(item["stages"]),
                int(item["raw"]["attempt_index"]),
                int(item["raw"]["success_slot"]),
            ),
        )[0]
    clusters.sort(
        key=lambda cluster: (
            -len(cluster["members"]),
            -float(cluster["representative"]["quality"]["score"]),
            int(cluster["representative"]["raw"]["success_slot"]),
        )
    )
    for index, cluster in enumerate(clusters, 1):
        cluster["cluster_id"] = f"path_{index:02d}"
    return clusters


def retain_path_clusters(clusters: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    median_quality = statistics.median(float(item["quality"]["score"]) for item in candidates)
    eligible = [cluster for cluster in clusters if len(cluster["members"]) >= 2]
    eligible.extend(
        cluster
        for cluster in clusters
        if len(cluster["members"]) == 1
        and float(cluster["representative"]["quality"]["score"]) >= median_quality
    )
    if not eligible:
        eligible = clusters[:1]

    selected: list[dict[str, Any]] = []
    for cluster in eligible:
        if len(selected) >= MAX_RETAINED_PATHS_PER_CASE:
            break
        if selected and max(
            trajectory_similarity(cluster["medoid"], existing["medoid"])
            for existing in selected
        ) >= 0.88:
            continue
        selected.append(cluster)
    if not selected:
        selected = clusters[:1]
    return selected


def source_elimination_statements(stage: dict[str, Any]) -> list[str]:
    """Return exact visible-source sentences that explicitly reject a candidate."""
    if stage["is_final"]:
        return []
    statements = [
        sentence.strip()
        for sentence in stage["text_optimization"].get(
            "source_thinking_raw_sentences", []
        )
        if sentence.strip()
        and any(marker in sentence for marker in ELIMINATION_MARKERS)
    ]
    return list(dict.fromkeys(statements))[:2]


def elimination_action_description(actions: list[dict[str, Any]]) -> str:
    pairs: list[str] = []
    for action in actions:
        semantics = action["causal_semantics"]
        devices = semantics["devices"] or ["相关节点"]
        families = [
            family
            for family in semantics["families"]
            if family not in {"other", "config"}
        ] or [family for family in semantics["families"] if family != "other"]
        family = FAMILY_NAMES.get(families[0] if families else "other", "补充状态")
        for device in devices[:2]:
            value = f"{device} 的{family}"
            if value not in pairs:
                pairs.append(value)
    return "、".join(pairs[:4]) or "已返回的区分性运行状态"


ELIMINATION_GENERIC_TERMS = {
    "display", "route", "routing", "table", "information", "interface",
    "mpls", "lsp", "vpn", "bgp", "isis", "ldp", "status", "current",
    "normal", "link", "port", "ip", "wan", "lan",
}


def elimination_atomic_claims(statements: list[str]) -> list[str]:
    """Keep factual clauses and discard unsupported source inference prose."""
    claims: list[str] = []
    for statement in statements:
        evidence_part = re.split(r"[：:]", statement, maxsplit=1)[-1]
        evidence_part = re.split(r"(?:因此|因而|所以)", evidence_part, maxsplit=1)[0]
        for clause in re.split(r"[，,、；;。！？!?]+", evidence_part):
            claim = clause.strip(" ：:")
            if (
                not claim
                or any(marker in claim for marker in ELIMINATION_MARKERS)
                or any(marker in claim for marker in ("下一步", "核验", "检查", "继续"))
            ):
                continue
            claims.append(claim)
    return list(dict.fromkeys(claims))


def elimination_candidate_phrase(statements: list[str]) -> str:
    text = " ".join(statements)
    quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
    if quoted:
        return "、".join(f"“{value}”" for value in quoted[:3])
    match = re.search(r"(?:排除|不支持|不能解释|不成立)([^：:。；;]+)", text)
    if match:
        return f"“{match.group(1).strip(' ：:，,')[:80]}”"
    return "该候选解释"


def evidence_rejection_scope(claim_bindings: list[dict[str, Any]]) -> str:
    observations = " ".join(
        evidence["observation_span"].lower()
        for binding in claim_bindings
        for evidence in binding["evidence"]
    )
    if "crc" in observations and re.search(r"crc\s*:\s*0\b", observations):
        return "接口 CRC 物理错误"
    if "last physical down time" in observations and re.search(
        r"last physical down time\s*:\s*-", observations
    ):
        return "当前接口持续 down"
    if "relay ip out-interface" in observations or re.search(
        r"\b(?:direct|ibgp|isis-l\d).*\b(?:ethernet|eth-trunk|vlanif|ge)\d",
        observations,
    ):
        return "与已显示出接口不一致的转发绕行"
    if re.search(r"\b(?:input|output).*\b(?:rate|utility).*\b0(?:\.0+)?%", observations):
        return "当前接口带宽拥塞"
    if re.search(r"\bmtu\s*[: ]\s*\d+", observations):
        return "与已显示数值不一致的 MTU 假设"
    return "与该回显直接矛盾的子候选"


def is_substantive_observation_line(line: str) -> bool:
    lowered = line.lower().strip()
    if (
        not lowered
        or lowered in {"(empty output)", "output:"}
        or "historical excerpt truncated" in lowered
        or "unselected lines omitted" in lowered
        or re.fullmatch(r"[-=+|<>\s]+", lowered)
        or lowered.startswith((
            "flag ", "flags:", "legend", "route flags:", "peer information",
            "route information", "lsp information", "display ", "chunk id:",
            "wall time:", "process exited", "original token count:",
        ))
        or lowered in {
            "port vrf status ip address speed mtu",
            "mstid port role stp state protection cost edged",
            "interface physical protocol ip address description",
        }
    ):
        return False
    return bool(re.search(
        r"(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?|\d+%|"
        r"(?:ethernet|eth-trunk|vlanif|ge)\d+(?:[/_.-]\d+)*|"
        r"\b(?:up|down|enabled?|disabled?|master|backup|active|inactive|"
        r"error|drop|crc|mtu|cost|preempt|direct|ibgp)\b|"
        r"(?:正常|异常|未启用|已启用|无丢包|无错误)",
        lowered,
        re.IGNORECASE,
    ))


def elimination_claim_anchors(claim: str) -> set[str]:
    lowered = claim.lower()
    anchors = {
        value.lower()
        for value in re.findall(
            r"(?:\d{1,3}\.){3}\d{1,3}(?:/\d+)?|\d+%|"
            r"(?:ethernet|eth-trunk|vlanif|ge)\d+(?:[/_.-]\d+)*|"
            r"\b(?:mtu|cost)\s*\d+\b|"
            r"\b(?:up|down|enabled?|disabled?|master|backup|active|inactive|"
            r"error|drop|crc|preempt|direct|ibgp)\b|"
            r"(?:正常|异常|未启用|已启用|无丢包|无错误)",
            lowered,
            re.IGNORECASE,
        )
    }
    terms = {
        value.lower()
        for value in re.findall(r"[A-Za-z][A-Za-z0-9_./<>:-]{2,}", lowered)
        if value.lower() not in ELIMINATION_GENERIC_TERMS
        and not re.fullmatch(r"pe\d|[a-z]+_sw_\d+", value.lower())
    }
    return anchors | terms


def observation_contains_anchor(observation: str, anchor: str) -> bool:
    if re.fullmatch(r"[a-z][a-z0-9_./<>:-]*", anchor):
        return bool(re.search(
            rf"(?<![a-z0-9_]){re.escape(anchor)}(?![a-z0-9_])",
            observation,
            re.IGNORECASE,
        ))
    return anchor in observation


def bind_elimination_claims(
    candidate: dict[str, Any], target_index: int, statements: list[str]
) -> list[dict[str, Any]]:
    """Bind each retained factual clause to exact earlier observation spans."""
    observations: list[dict[str, Any]] = []
    for stage in candidate["stages"][:target_index]:
        for action in stage["actions"]:
            if action["causal_semantics"].get("is_discovery"):
                continue
            for source_line in visible_action_output(action, compact=True).splitlines():
                line = re.sub(r"\s+", " ", source_line).strip()
                if is_substantive_observation_line(line):
                    observations.append({
                        "action": action,
                        "action_id": str(action["action_id"]),
                        "observation_span": line[:240],
                        "normalized": line.lower(),
                    })
    bindings: list[dict[str, Any]] = []
    for claim in elimination_atomic_claims(statements):
        anchors = elimination_claim_anchors(claim)
        if not anchors:
            continue
        remaining = set(anchors)
        selected: list[dict[str, Any]] = []
        pool = list(observations)
        while remaining and pool and len(selected) < 3:
            best = max(
                pool,
                key=lambda item: len(
                    {
                        anchor for anchor in remaining
                        if observation_contains_anchor(item["normalized"], anchor)
                    }
                ),
            )
            covered = {
                anchor for anchor in remaining
                if observation_contains_anchor(best["normalized"], anchor)
            }
            if not covered:
                break
            selected.append(best)
            remaining -= covered
            pool.remove(best)
        negative_claim = any(
            marker in claim.lower() for marker in ("无", "未", "没有", "0%")
        )
        selected_text = " ".join(item["normalized"] for item in selected)
        negative_visible = any(
            marker in selected_text
            for marker in (" 0", "no ", "none", "not ", "无", "未", "disable")
        )
        if remaining or (negative_claim and not negative_visible):
            continue
        evidence = [
            {
                "action_id": item["action_id"],
                "observation_span": item["observation_span"],
            }
            for item in selected
        ]
        bindings.append({
            "claim": claim,
            "anchors": sorted(anchors),
            "evidence": evidence,
            "supervised_fact": "；".join(
                f"{item['action_id']}回显“{item['observation_span']}”"
                for item in selected
            ),
        })
    return bindings


def choose_path_elimination_source(
    cluster: dict[str, Any],
) -> tuple[
    dict[str, Any], int, list[str], list[dict[str, Any]], list[dict[str, Any]]
] | None:
    """Choose one faithful, evidence-backed elimination node from a path cluster."""
    ranked: list[
        tuple[
            tuple[float, ...], dict[str, Any], int, list[str],
            list[dict[str, Any]], list[dict[str, Any]],
        ]
    ] = []
    representative_id = str(cluster["representative"]["raw"]["id"])
    for candidate in cluster["members"]:
        for target_index, stage in enumerate(candidate["stages"][:-1]):
            if target_index == 0:
                continue
            statements = source_elimination_statements(stage)
            if not statements:
                continue
            claim_bindings = bind_elimination_claims(candidate, target_index, statements)
            if (
                not claim_bindings
                or evidence_rejection_scope(claim_bindings)
                == "与该回显直接矛盾的子候选"
            ):
                continue
            action_ids = list(dict.fromkeys(
                evidence["action_id"]
                for binding in claim_bindings
                for evidence in binding["evidence"]
            ))
            by_action_id = {
                str(action["action_id"]): action
                for prior_stage in candidate["stages"][:target_index]
                for action in prior_stage["actions"]
            }
            evidence_actions = [by_action_id[action_id] for action_id in action_ids]
            score = (
                float(len(claim_bindings)),
                float(sum(len(binding["claim"]) for binding in claim_bindings)),
                float(candidate["quality"]["score"]),
                float(str(candidate["raw"]["id"]) == representative_id),
                -float(target_index),
                -float(candidate["raw"]["attempt_index"]),
            )
            ranked.append(
                (
                    score, candidate, target_index, statements,
                    claim_bindings, evidence_actions,
                )
            )
    if not ranked:
        return None
    _, candidate, target_index, statements, claim_bindings, evidence_actions = max(
        ranked, key=lambda item: item[0]
    )
    return candidate, target_index, statements, claim_bindings, evidence_actions


def build_hypothesis_elimination_candidate(
    candidate: dict[str, Any],
    target_index: int,
    statements: list[str],
    claim_bindings: list[dict[str, Any]],
    evidence_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Turn one real visible elimination step into an explicit supervised target."""
    augmented = copy.deepcopy(candidate)
    stage = augmented["stages"][target_index]
    evidence_description = elimination_action_description(evidence_actions)
    evidence_observation_text = "；".join(
        f"{evidence['action_id']}回显“{evidence['observation_span']}”"
        for binding in claim_bindings
        for evidence in binding["evidence"]
    )
    grounded_claims = [binding["supervised_fact"] for binding in claim_bindings]
    candidate_phrase = evidence_rejection_scope(claim_bindings)
    stage["thinking"] = (
        f"先逐项核对已经返回的事实：{'；'.join(grounded_claims)}。"
        "排除只使用这些可定位的回显；没有被回显覆盖的状态不作推断。"
    )
    stage["conclusion"] = (
        f"候选排除：可见事实为{'；'.join(grounded_claims)}。"
        f"就这些已核对的子项而言，当前结果削弱“{candidate_phrase}”；"
        "未被回显覆盖的部分不作排除，因此这里只降低该候选的排查优先级。"
    )
    stage["source_hypothesis_elimination"] = True
    stage["rejected_candidate_statements"] = statements
    stage["supervised_elimination_claims"] = grounded_claims
    stage["elimination_claim_bindings"] = claim_bindings
    stage["elimination_evidence_action_count"] = len(evidence_actions)
    stage["elimination_evidence_action_ids"] = [
        str(action["action_id"]) for action in evidence_actions
    ]
    stage["elimination_evidence_description"] = evidence_description
    stage["elimination_evidence_observation"] = evidence_observation_text
    stage["text_optimization"] = {
        **stage["text_optimization"],
        "thinking_source": "claim_bound_hypothesis_elimination",
        "conclusion_source": "source_grounded_hypothesis_elimination",
        "source_elimination_statements": statements,
        "elimination_evidence_action_ids": stage["elimination_evidence_action_ids"],
        "elimination_derived_from_visible_source": True,
    }
    augmented["path_endpoint_policy"] = {
        "evidence_summary_nodes": 1,
        "decision_ready_nodes": 1,
        "decision_nodes": 1,
        "sampling_unit": "single_multi_target_endpoint_bundle",
    }
    return augmented


def build_path_endpoint_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Add endpoints only when visible same-snapshot evidence passes the gate."""
    augmented = copy.deepcopy(candidate)
    source_stages = candidate["stages"]
    evidence_stages = [
        copy.deepcopy(stage) for stage in source_stages[:-1] if stage["actions"]
    ]
    if not evidence_stages:
        return None

    gate = endpoint_evidence_gate(
        evidence_stages, candidate["raw"], compact_history=False
    )
    if not gate["passed"]:
        return None
    selected_facts = gate["selected_facts"]
    selected_observations_by_action: dict[str, list[str]] = defaultdict(list)
    for fact in selected_facts:
        selected_observations_by_action[str(fact["action_id"])].append(
            str(fact["observation"])
        )
    for stage in evidence_stages:
        for action in stage["actions"]:
            action_id = str(action["action_id"])
            observations = selected_observations_by_action.get(action_id)
            if not observations:
                continue
            action["source_endpoint_excerpt_sha256_lf_normalized"] = str(
                action["excerpt_sha256_lf_normalized"]
            )
            action["output_excerpt"] = "\n".join(dict.fromkeys(observations))
            action["output_is_excerpt"] = True
            action["excerpt_sha256_lf_normalized"] = base.digest_text(
                action["output_excerpt"]
            )
    gate = endpoint_evidence_gate(
        evidence_stages, candidate["raw"], compact_history=True
    )
    if not gate["passed"]:
        raise ValueError(
            f"{candidate['raw']['id']}: endpoint evidence disappeared from visible history"
        )
    selected_facts = gate["selected_facts"]
    evidence_actions = [action for stage in evidence_stages for action in stage["actions"]]
    by_action_id = {str(action["action_id"]): action for action in evidence_actions}
    decisive_actions = [
        by_action_id[action_id]
        for action_id in dict.fromkeys(str(fact["action_id"]) for fact in selected_facts)
    ]
    evidence_message_indices = [
        int(stage["source_message_index"])
        for stage in evidence_stages
        if stage.get("source_message_index") is not None
    ]
    evidence_features = sorted(set().union(
        *(set(stage.get("causal_features", [])) for stage in evidence_stages)
    ))
    evidence_description = "、".join(
        dict.fromkeys(endpoint_fact_description(fact) for fact in selected_facts)
    )
    observation_description = "；".join(
        dict.fromkeys(str(fact["observation"]) for fact in selected_facts)
    )
    provenance_hash = base.digest_text("\n".join(
        [
            *(str(action["excerpt_sha256_lf_normalized"]) for action in decisive_actions),
            *(str(fact["observation"]) for fact in selected_facts),
        ]
    ))
    common = {
        "source_message_index": None,
        "source_message_event_line": None,
        "source_message_item_id": None,
        "is_final": False,
        "actions": [],
        "omitted_actions": [],
        "causal_features": evidence_features,
        "source_evidence_message_indices": evidence_message_indices,
        "source_evidence_action_count": len(evidence_actions),
        "source_decisive_action_count": len(decisive_actions),
        "source_decisive_action_ids": [str(action["action_id"]) for action in decisive_actions],
        "grounded_evidence_description": evidence_description,
        "grounded_evidence_observation": observation_description,
        "excluded_evidence_description": None,
        "excluded_evidence_observation": None,
        "endpoint_evidence_gate": gate,
        "derived_from_verified_final_answer": False,
    }
    evidence_summary = {
        **copy.deepcopy(common),
        "synthetic_stage_type": "evidence_summary",
        "thinking": (
            f"题目指定快照的直接结果已经返回：{evidence_description}。"
            "这里只归纳当前可见事实，不使用其他快照或未返回的信息补足。"
        ),
        "conclusion": (
            f"证据归纳：{evidence_description}。每项事实都能在前序工具回显中直接定位。"
        ),
        "text_optimization": {
            "thinking_source": "evidence_summary_bridge",
            "conclusion_source": "path_evidence_synthesis",
            "hidden_chain_of_thought_claimed": False,
            "source_message_sha256_lf_normalized": provenance_hash,
            "source_evidence_message_indices": evidence_message_indices,
            "derived_from_verified_final_answer": False,
            "verified_label_used_only_for_posthoc_gate": True,
        },
    }
    stop_judgment = {
        **copy.deepcopy(common),
        "synthetic_stage_type": "decision_ready",
        "thinking": (
            f"当前直接证据为：{evidence_description}。"
            "这些事实足以支持当前故障候选进入最终决策；"
            "这里不把已找到支持证据扩大为已经排尽所有其他故障。"
        ),
        "conclusion": (
            f"停止判断：{evidence_description}直接支持当前候选。"
            "现有证据已达到作答所需强度，进入最终决策并按题目格式输出。"
        ),
        "text_optimization": {
            "thinking_source": "stop_judgment_bridge",
            "conclusion_source": "verified_path_stop_judgment",
            "hidden_chain_of_thought_claimed": False,
            "source_message_sha256_lf_normalized": provenance_hash,
            "source_evidence_message_indices": evidence_message_indices,
            "derived_from_verified_final_answer": False,
            "verified_label_used_only_for_posthoc_gate": True,
        },
    }
    augmented["stages"] = [
        *evidence_stages,
        evidence_summary,
        stop_judgment,
        copy.deepcopy(source_stages[-1]),
    ]
    augmented["path_endpoint_policy"] = {
        "evidence_summary_nodes": 1,
        "decision_ready_nodes": 1,
        "decision_nodes": 1,
        "sampling_unit": "single_multi_target_endpoint_bundle",
        "evidence_gate_rule": gate["gate_rule"],
        "verified_label_in_summary_or_stop": False,
        "original_no_action_prefix_nodes_replaced": sum(
            not stage["actions"] for stage in source_stages[:-1]
        ),
    }
    return augmented


def stage_target_type(stage: dict[str, Any], index: int) -> str:
    if stage.get("source_hypothesis_elimination"):
        return "hypothesis_elimination"
    if stage.get("synthetic_stage_type") == "evidence_summary":
        return "evidence_summary"
    if stage.get("synthetic_stage_type") == "decision_ready":
        return "decision_ready"
    if stage["is_final"]:
        return "decision"
    if not stage["actions"]:
        return "source_decision_ready"
    return "planning" if index == 0 else "reasoning"


def stage_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["target_type"] == "decision_ready" or right["target_type"] == "decision_ready":
        return left["target_type"] == right["target_type"]
    left_features = set(left["stage"]["causal_features"])
    right_features = set(right["stage"]["causal_features"])
    return jaccard(left_features, right_features) >= NODE_DUPLICATE_THRESHOLD


def build_row(
    candidate: dict[str, Any],
    cluster: dict[str, Any],
    target_index: int,
    target_type: str,
    merged_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    annotation = candidate["annotation"]
    raw = candidate["raw"]
    stage = candidate["stages"][target_index]
    endpoint_bundle = target_type == "endpoint_bundle"
    synthetic_stage_type = (
        "endpoint_bundle" if endpoint_bundle else stage.get("synthetic_stage_type")
    )
    evidence_metadata_stage = (
        candidate["stages"][target_index - 2] if endpoint_bundle else stage
    )
    if synthetic_stage_type:
        row_id = (
            f"q{int(raw['case_id']):04d}_{cluster['cluster_id']}_"
            f"success_{int(raw['success_slot']):02d}_{synthetic_stage_type}"
        )
    else:
        row_id = (
            f"q{int(raw['case_id']):04d}_{cluster['cluster_id']}_"
            f"success_{int(raw['success_slot']):02d}_step_"
            f"{int(stage['source_message_index']) + 1:02d}"
        )
    messages = base.build_messages(raw, candidate["stages"], target_index)
    if not messages or messages[0]["role"] != "system":
        raise ValueError(f"{row_id}: base message builder did not emit a system message")
    messages[0]["content"] = CODEX_CLI_SYSTEM_PROMPT
    rewrite_messages_to_codex_cli_protocol(messages, candidate["stages"], target_index)
    thinking_source = stage["text_optimization"]["thinking_source"]
    thinking_loss_scale = THINKING_LOSS_SCALE_BY_SOURCE[thinking_source]
    conclusion_source = stage["text_optimization"]["conclusion_source"]
    conclusion_loss_scale = CONCLUSION_LOSS_SCALE_BY_SOURCE[conclusion_source]
    for message in reversed(messages):
        if (
            message["role"] == "assistant"
            and "<think>" in message["content"]
            and message.get("loss_scale") == base.THINKING_LOSS_SCALE
        ):
            message["loss_scale"] = thinking_loss_scale
            break
    for message in reversed(messages):
        if (
            message["role"] == "assistant"
            and "<think>" not in message["content"]
            and message.get("loss_scale") == base.TARGET_LOSS_SCALE
        ):
            message["loss_scale"] = conclusion_loss_scale
            break
    for message in messages:
        if (
            message["role"] == "tool_call"
            and message.get("loss_scale") == base.TARGET_LOSS_SCALE
        ):
            message["loss_scale"] = TOOL_CALL_LOSS_SCALE
    endpoint_component_loss: dict[str, dict[str, float]] = {}
    if endpoint_bundle:
        component_stages = candidate["stages"][target_index - 2 : target_index + 1]
        if [
            component.get("synthetic_stage_type") or "decision"
            for component in component_stages
        ] != list(ENDPOINT_COMPONENT_TYPES):
            raise ValueError(f"{row_id}: invalid endpoint bundle stage order")
        for component_type, component in zip(ENDPOINT_COMPONENT_TYPES, component_stages):
            component_thinking_source = component["text_optimization"]["thinking_source"]
            component_conclusion_source = component["text_optimization"]["conclusion_source"]
            component_thinking_loss = THINKING_LOSS_SCALE_BY_SOURCE[
                component_thinking_source
            ]
            component_conclusion_loss = CONCLUSION_LOSS_SCALE_BY_SOURCE[
                component_conclusion_source
            ]
            thinking_matches = [
                message for message in messages
                if message["role"] == "assistant"
                and "<think>" in message["content"]
                and component["thinking"] in message["content"]
            ]
            conclusion_matches = [
                message for message in messages
                if message["role"] == "assistant"
                and "<think>" not in message["content"]
                and base.normalize_text(str(component["conclusion"]))
                in base.normalize_text(str(message["content"]))
            ]
            if len(thinking_matches) != 1 or len(conclusion_matches) != 1:
                raise ValueError(
                    f"{row_id}: endpoint component {component_type} message mapping failed"
                )
            thinking_matches[0]["loss_scale"] = component_thinking_loss
            conclusion_matches[0]["loss_scale"] = component_conclusion_loss
            endpoint_component_loss[component_type] = {
                "thinking": component_thinking_loss,
                "conclusion_or_result": component_conclusion_loss,
            }
    return {
        "id": row_id,
        "tools": json.dumps(CODEX_CLI_TOOLS, ensure_ascii=False, separators=(",", ":")),
        "messages": messages,
        "metadata": {
            "dataset_type": "reasoning_causal_path_step",
            "target_type": target_type,
            "review_status": "auto_clustered_draft",
            "split": annotation["split"],
            "case_id": raw["case_id"],
            "row_index": raw["row_index"],
            "trajectory_id": raw["id"],
            "success_slot": raw["success_slot"],
            "attempt_index": raw["attempt_index"],
            "path_cluster_id": cluster["cluster_id"],
            "path_cluster_size": len(cluster["members"]),
            "path_cluster_member_ids": sorted(member["raw"]["id"] for member in cluster["members"]),
            "retained_path_count_for_case": None,
            "step_index_in_compacted_path": target_index + 1,
            "step_count_in_compacted_path": len(candidate["stages"]),
            "source_message_index": stage["source_message_index"],
            "source_message_event_line": stage["source_message_event_line"],
            "source_message_item_id": stage["source_message_item_id"],
            "synthetic_stage": synthetic_stage_type is not None,
            "synthetic_stage_type": synthetic_stage_type,
            "source_evidence_message_indices": evidence_metadata_stage.get(
                "source_evidence_message_indices", []
            ),
            "source_evidence_action_count": evidence_metadata_stage.get(
                "source_evidence_action_count", 0
            ),
            "source_decisive_action_count": evidence_metadata_stage.get(
                "source_decisive_action_count", 0
            ),
            "source_decisive_action_ids": evidence_metadata_stage.get(
                "source_decisive_action_ids", []
            ),
            "grounded_evidence_description": evidence_metadata_stage.get(
                "grounded_evidence_description"
            ),
            "grounded_evidence_observation": evidence_metadata_stage.get(
                "grounded_evidence_observation"
            ),
            "excluded_evidence_description": stage.get(
                "excluded_evidence_description"
            ),
            "excluded_evidence_observation": stage.get(
                "excluded_evidence_observation"
            ),
            "endpoint_evidence_gate": evidence_metadata_stage.get(
                "endpoint_evidence_gate"
            ),
            "endpoint_supervision_components": (
                list(ENDPOINT_COMPONENT_TYPES) if endpoint_bundle else []
            ),
            "endpoint_component_loss": endpoint_component_loss,
            "rejected_candidate_statements": stage.get(
                "rejected_candidate_statements", []
            ),
            "supervised_elimination_claims": stage.get(
                "supervised_elimination_claims", []
            ),
            "elimination_claim_bindings": stage.get(
                "elimination_claim_bindings", []
            ),
            "elimination_evidence_action_count": stage.get(
                "elimination_evidence_action_count", 0
            ),
            "elimination_evidence_action_ids": stage.get(
                "elimination_evidence_action_ids", []
            ),
            "elimination_evidence_description": stage.get(
                "elimination_evidence_description"
            ),
            "elimination_evidence_observation": stage.get(
                "elimination_evidence_observation"
            ),
            "elimination_derived_from_visible_source": bool(
                stage.get("source_hypothesis_elimination")
            ),
            "derived_from_verified_final_answer": stage.get(
                "derived_from_verified_final_answer", False
            ),
            "history_contains_evidence_summary": any(
                prior.get("synthetic_stage_type") == "evidence_summary"
                for prior in candidate["stages"][:target_index]
            ),
            "history_contains_stop_judgment": any(
                prior.get("synthetic_stage_type") == "decision_ready"
                for prior in candidate["stages"][:target_index]
            ),
            "source_file": annotation["raw_file"],
            "source_event_file": annotation["events_file"],
            "source_event_sha256_lf_normalized": annotation["events_sha256_lf_normalized"],
            "thinking_source": stage["text_optimization"]["thinking_source"],
            "thinking_is_original_hidden_chain_of_thought": False,
            "system_prompt_source": CODEX_MODEL_CATALOG.relative_to(ROOT).as_posix(),
            "system_prompt_model_slug": CODEX_CLI_MODEL_SLUG,
            "system_prompt_sha256": CODEX_CLI_SYSTEM_PROMPT_SHA256,
            "conclusion_source": conclusion_source,
            "text_optimization": stage["text_optimization"],
            "causal_features": stage.get("causal_features", []),
            "current_actions": base.public_actions(stage["actions"]),
            "current_action_count": len(stage["actions"]),
            "action_selection": stage.get("action_selection", {
                "policy": "no_current_action",
                "original_action_count": 0,
                "eligible_action_count": 0,
                "kept_action_count": 0,
                "required_claim_units": [],
                "covered_claim_units_before": [],
                "covered_claim_units_after": [],
                "claim_coverage_before": 1.0,
                "claim_coverage_after": 1.0,
            }),
            "merged_equivalent_nodes": merged_sources,
            "merged_equivalent_node_count": len(merged_sources),
            "evidence_converged_without_next_tool_call": endpoint_bundle,
            "future_event_leakage_checked": True,
            "final_answer_visible": stage["is_final"],
            "actual_result_items": raw["actual_result_items"] if stage["is_final"] else None,
            "reference_answer_match": bool(raw["answer_matches_reference"]) if stage["is_final"] else None,
            "loss_policy": {
                "thinking": thinking_loss_scale,
                "conclusion_or_result": conclusion_loss_scale,
                "tool_calls": TOOL_CALL_LOSS_SCALE,
                "history": base.HISTORY_LOSS_SCALE,
                "tool_responses": "context_only",
            },
            "training_sampling_role": "primary",
            "sampling_source_row_id": None,
            "endpoint_group_exposures_per_query_per_epoch": (
                ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
                if target_type in ENDPOINT_TARGET_TYPES
                else 0
            ),
            "endpoint_schedule_epoch": None,
            "endpoint_schedule_slot": None,
            "core_schedule_epoch": None,
            "core_schedule_slot": None,
            "tool_protocol": {
                "name": "exec_command",
                "command_argument": "cmd",
                "operating_system": "linux",
                "path_style": "repository_relative_saved_configs",
                "source_command_audit": "exact PowerShell retained in current_actions.source_command",
                "tool_response_transport": "Codex CLI shape with Linux-normalized discovery/search output; token count is heuristic and context-only",
            },
            "path_endpoint_policy": candidate.get("path_endpoint_policy"),
            "cross_trajectory_evidence_recovery": bool(
                candidate.get("cross_trajectory_evidence_recovery")
            ),
            "inclusive_or_singleton_selected_by_evidence": bool(
                candidate.get("inclusive_or_singleton_selected_by_evidence")
            ),
        },
    }


def candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_id": candidate["raw"]["id"],
        "success_slot": candidate["raw"]["success_slot"],
        "attempt_index": candidate["raw"]["attempt_index"],
        "quality": candidate["quality"],
        "path_features": sorted(candidate["path_features"]),
        "path_transitions": sorted(candidate["path_transitions"]),
        "stage_preparation": candidate["stage_preparation"],
    }


def recovery_source_action(
    candidate: dict[str, Any], command: dict[str, Any]
) -> dict[str, Any] | None:
    if int(command.get("exit_code", 1)) != 0 or command.get("status") != "completed":
        return None
    source_command = str(command["command"])
    semantics = action_semantics({"command": source_command})
    if (
        semantics.get("has_path_glob")
        or set(semantics.get("snapshots", [])) != {question_snapshot(candidate["raw"])}
        or semantics.get("is_discovery")
    ):
        return None
    direct_markers = (
        "protocol status", "bpdu-filter", "preempt", "state", "master",
        "ip route-static", "static-lsp", " alte ", "discarding", "ip address",
        " vlanif", "instance ", " inet ",
    )
    searchable = f" {source_command}\n{command.get('output', '')} ".lower()
    if not any(marker in searchable for marker in direct_markers):
        return None
    source_powershell_command = normalize_supervised_command(source_command)
    try:
        translation = translate_powershell_to_exec_command(source_powershell_command)
    except ValueError:
        return None
    trajectory_id = str(candidate["raw"]["id"])
    item_id = str(command["item_id"])
    output = str(command.get("output", ""))
    action = {
        "action_id": f"REC-{trajectory_id}-{item_id}",
        "item_id": item_id,
        "event_line": int(command["event_line"]),
        "command": translation["cmd"],
        "output_excerpt": output,
        "output_is_excerpt": False,
        "exit_code": int(command["exit_code"]),
        "status": str(command["status"]),
        "selection_score": 0.0,
        "target": base.command_target(source_command),
        "command_sha256_lf_normalized": base.digest_text(source_command),
        "full_output_sha256_lf_normalized": base.digest_text(output),
        "excerpt_sha256_lf_normalized": base.digest_text(output),
        "source_command": source_command,
        "source_powershell_command": source_powershell_command,
        "source_powershell_command_sha256_lf_normalized": base.digest_text(
            source_powershell_command
        ),
        "justification": translation["justification"],
        "command_translation": translation["kind"],
        "command_protocol": "codex_cli.exec_command.arguments.cmd",
        "command_was_translated": translation["cmd"] != source_powershell_command,
        "result_protocol": "codex_cli.function_call_output.linux_normalized",
        "supervised_cmd_sha256_lf_normalized": base.digest_text(translation["cmd"]),
        "causal_priority": 0.0,
        "causal_semantics": semantics,
        "cross_trajectory_recovery_source": True,
        "source_trajectory_id": trajectory_id,
    }
    return action


def build_case_recovery_endpoint_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compose minimal same-query real actions when no single run has full coverage."""
    actions: list[dict[str, Any]] = []
    for candidate in candidates:
        for command in candidate["commands"]:
            action = recovery_source_action(candidate, command)
            if action is not None:
                actions.append(action)
    if not actions:
        return None
    unique_actions: dict[str, dict[str, Any]] = {}
    for action in actions:
        key = f"{action['command']}\n{action['full_output_sha256_lf_normalized']}"
        unique_actions.setdefault(key, action)
    recovery_stage = {
        "source_message_index": None,
        "source_message_event_line": None,
        "source_message_item_id": None,
        "thinking": (
            "单条成功运行没有同时保留全部原子证据；这里仅合并同一题目、同一配置快照中"
            "已经真实执行并成功返回的只读检查。"
        ),
        "conclusion": "复核形成闭环所需的设备级直接证据，不使用其他题目或其他网络快照。",
        "is_final": False,
        "actions": list(unique_actions.values()),
        "omitted_actions": [],
        "text_optimization": {
            "thinking_source": "fixed_bridge_template",
            "conclusion_source": "evidence_aligned_reconstruction",
            "hidden_chain_of_thought_claimed": False,
            "source_message_sha256_lf_normalized": base.digest_text(
                "0807 same-query cross-trajectory evidence recovery"
            ),
            "cross_trajectory_evidence_recovery": True,
        },
    }
    recovery_stage["causal_features"] = sorted(stage_feature_set(recovery_stage))
    # The inclusive-OR evaluator accepts either core (or both), but SFT should
    # prefer the existing correct singleton whose device-specific VLAN/instance
    # closure is actually visible.  Evidence actions are collected before this
    # post-hoc target choice and remain label-independent.
    base_candidate: dict[str, Any] | None = None
    provisional: dict[str, Any] | None = None
    for candidate in sorted(
        candidates,
        key=lambda item: float(item["quality"]["score"]),
        reverse=True,
    ):
        candidate_gate = endpoint_evidence_gate(
            [recovery_stage], candidate["raw"], compact_history=False
        )
        if candidate_gate["passed"]:
            base_candidate = candidate
            provisional = candidate_gate
            break
    if base_candidate is None or provisional is None:
        return None
    selected_action_ids = {
        str(fact["action_id"]) for fact in provisional["selected_facts"]
    }
    recovery_stage["actions"] = [
        action for action in recovery_stage["actions"]
        if str(action["action_id"]) in selected_action_ids
    ]
    recovery_stage["causal_features"] = sorted(stage_feature_set(recovery_stage))
    source = copy.deepcopy(base_candidate)
    source["stages"] = [recovery_stage, copy.deepcopy(base_candidate["stages"][-1])]
    endpoint = build_path_endpoint_candidate(source)
    if endpoint is not None:
        endpoint["cross_trajectory_evidence_recovery"] = True
        endpoint["inclusive_or_singleton_selected_by_evidence"] = (
            73 <= int(base_candidate["raw"]["case_id"]) <= 86
        )
    return endpoint


def label_independent_candidate_quality(
    raw: dict[str, Any],
    messages: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [
        command for command in commands
        if command["status"] == "completed" and command["exit_code"] in (0, None)
    ]
    unique_successful = {str(command["command"]) for command in successful}
    retained_actions = [action for stage in stages[:-1] for action in stage["actions"]]
    devices = {
        device
        for action in retained_actions
        for device in action["causal_semantics"].get("devices", [])
    }
    families = {
        family
        for action in retained_actions
        for family in action["causal_semantics"].get("families", [])
        if family not in {"other", "discovery", "config"}
    }
    duplicate_count = len(successful) - len(unique_successful)
    failed_count = len(commands) - len(successful)
    message_count = len(messages)
    duration = float(raw.get("source", {}).get("duration_seconds") or 0.0)
    score = (
        10.0 * min(1.0, len(successful) / max(1, len(commands)))
        + 2.0 * min(6, len(devices))
        + 2.0 * min(8, len(families))
        + 6.0 * float(3 <= message_count <= 7)
        + 0.25 * min(24, len(retained_actions))
        - 0.12 * duplicate_count
        - 0.60 * failed_count
        - 0.0015 * duration
    )
    return {
        "score": round(score, 6),
        "label_features_used": False,
        "message_count": message_count,
        "source_command_count": len(commands),
        "successful_command_count": len(successful),
        "duplicate_successful_command_count": duplicate_count,
        "failed_command_count": failed_count,
        "retained_action_count": len(retained_actions),
        "retained_device_count": len(devices),
        "retained_family_count": len(families),
        "duration_seconds": duration,
    }


def process_case(
    case_id: int, annotations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for annotation in annotations:
        raw = base.load_json(ROOT / annotation["raw_file"])
        messages, commands = base.parse_events(ROOT / annotation["events_file"])
        base.validate_raw(annotation, raw, messages)
        stages, stage_preparation = prepare_stages(raw, messages, commands)
        path_features, path_transitions = trajectory_path_features(stages)
        candidates.append(
            {
                "annotation": annotation,
                "raw": raw,
                "messages": messages,
                "commands": commands,
                "stages": stages,
                "stage_preparation": stage_preparation,
                "path_features": path_features,
                "path_transitions": path_transitions,
                "quality": label_independent_candidate_quality(
                    raw, messages, commands, stages
                ),
            }
        )
    if len(candidates) != 10:
        raise ValueError(f"q{case_id}: expected 10 accepted trajectories, found {len(candidates)}")

    clusters = cluster_trajectories(candidates)
    retained_clusters = retain_path_clusters(clusters, candidates)
    endpoint_candidates_by_cluster: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        qualified: list[tuple[float, int, dict[str, Any]]] = []
        for member in cluster["members"]:
            endpoint_candidate = build_path_endpoint_candidate(member)
            if endpoint_candidate is None:
                continue
            qualified.append(
                (
                    float(member["quality"]["score"]),
                    -len(endpoint_candidate["stages"]),
                    endpoint_candidate,
                )
            )
        if qualified:
            endpoint_candidates_by_cluster[cluster["cluster_id"]] = max(
                qualified, key=lambda item: (item[0], item[1])
            )[2]
    if not any(
        cluster["cluster_id"] in endpoint_candidates_by_cluster
        for cluster in retained_clusters
    ):
        recovery_cluster = next(
            (
                cluster for cluster in clusters
                if cluster["cluster_id"] in endpoint_candidates_by_cluster
            ),
            None,
        )
        if recovery_cluster is not None:
            retained_clusters = [*retained_clusters[:-1], recovery_cluster]
    initially_retained_ids = {cluster["cluster_id"] for cluster in retained_clusters}

    path_endpoint_candidates = {
        cluster["cluster_id"]: endpoint_candidates_by_cluster[cluster["cluster_id"]]
        for cluster in retained_clusters
        if cluster["cluster_id"] in endpoint_candidates_by_cluster
    }
    if not path_endpoint_candidates:
        recovery_endpoint = build_case_recovery_endpoint_candidate(candidates)
        if recovery_endpoint is not None:
            path_endpoint_candidates[retained_clusters[0]["cluster_id"]] = recovery_endpoint
    # A retained training path must have a trustworthy ending. Early nodes from
    # paths that fail the direct-evidence gate are excluded instead of teaching
    # an investigation pattern with no supported stop/answer transition.
    endpoint_qualified_cluster_ids = set(path_endpoint_candidates)
    retained_clusters = [
        cluster
        for cluster in retained_clusters
        if cluster["cluster_id"] in endpoint_qualified_cluster_ids
    ]
    if not retained_clusters:
        raise ValueError(f"q{case_id}: no retained path passed the endpoint evidence gate")
    path_candidates = {
        cluster["cluster_id"]: cluster["representative"]
        for cluster in retained_clusters
    }
    # Preserve at most one strongest real candidate-elimination example per
    # query. It may come from a cluster that lacks a trustworthy final endpoint:
    # the elimination node is independently grounded in its own earlier visible
    # evidence and is treated as auxiliary supervision, not as a retained path.
    elimination_options: list[tuple[tuple[float, ...], dict[str, Any], Any]] = []
    for cluster in clusters:
        source = choose_path_elimination_source(cluster)
        if source is None:
            continue
        candidate, _, statements, claim_bindings, evidence_actions = source
        score = (
            float(len(claim_bindings)),
            float(sum(len(binding["claim"]) for binding in claim_bindings)),
            float(candidate["quality"]["score"]),
            float(len(cluster["members"])),
        )
        elimination_options.append((score, cluster, source))
    path_elimination_sources: dict[
        str, tuple[
            dict[str, Any], int, list[str], list[dict[str, Any]], list[dict[str, Any]]
        ]
    ] = {}
    elimination_cluster_by_id: dict[str, dict[str, Any]] = {}
    if elimination_options:
        _, elimination_cluster, elimination_source = max(
            elimination_options,
            key=lambda item: (item[0], str(item[1]["cluster_id"])),
        )
        elimination_cluster_id = str(elimination_cluster["cluster_id"])
        path_elimination_sources[elimination_cluster_id] = elimination_source
        elimination_cluster_by_id[elimination_cluster_id] = elimination_cluster
    chosen_representative_elimination_steps: set[tuple[str, int]] = set()
    for cluster_id, source in path_elimination_sources.items():
        if str(source[0]["raw"]["id"]) == str(
            elimination_cluster_by_id[cluster_id]["representative"]["raw"]["id"]
        ):
            chosen_representative_elimination_steps.add(
                (str(source[0]["raw"]["id"]), int(source[1]))
            )
    selected_nodes: list[dict[str, Any]] = []
    node_sources: list[dict[str, Any]] = []
    for cluster in retained_clusters:
        representative = path_candidates[cluster["cluster_id"]]
        for target_index, stage in enumerate(representative["stages"][:-1]):
            if (
                str(representative["raw"]["id"]),
                target_index,
            ) in chosen_representative_elimination_steps:
                continue
            target_type = stage_target_type(stage, target_index)
            node = {
                "candidate": representative,
                "cluster": cluster,
                "target_index": target_index,
                "target_type": target_type,
                "stage": stage,
                "merged": [],
            }
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(selected_nodes)
                    if stage_duplicate(node, existing)
                ),
                None,
            )
            source = {
                "trajectory_id": representative["raw"]["id"],
                "path_cluster_id": cluster["cluster_id"],
                "source_message_index": stage["source_message_index"],
            }
            if duplicate_index is None:
                selected_nodes.append(node)
                node_sources.append(source)
            else:
                selected_nodes[duplicate_index]["merged"].append(source)

    # Add at most one exact visible-source, evidence-backed candidate
    # elimination per retained path. A non-representative cluster member may
    # supply the node when it contains the clearer real elimination example.
    for cluster_id, source in path_elimination_sources.items():
        cluster = elimination_cluster_by_id[cluster_id]
        candidate, target_index, statements, claim_bindings, evidence_actions = source
        elimination_candidate = build_hypothesis_elimination_candidate(
            candidate, target_index, statements, claim_bindings, evidence_actions
        )
        stage = elimination_candidate["stages"][target_index]
        selected_nodes.append(
            {
                "candidate": elimination_candidate,
                "cluster": cluster,
                "target_index": target_index,
                "target_type": "hypothesis_elimination",
                "stage": stage,
                "merged": [],
            }
        )

    # One endpoint row jointly supervises summary, stop judgment, and final
    # answer.  This avoids serializing the same long context three times.
    for cluster in retained_clusters:
        representative = path_endpoint_candidates.get(cluster["cluster_id"])
        if representative is None:
            continue
        target_index = len(representative["stages"]) - 1
        selected_nodes.append(
            {
                "candidate": representative,
                "cluster": cluster,
                "target_index": target_index,
                "target_type": "endpoint_bundle",
                "stage": representative["stages"][target_index],
                "merged": [],
            }
        )

    represented_cluster_ids = {
        node["cluster"]["cluster_id"] for node in selected_nodes
    }
    retained_clusters = [
        cluster
        for cluster in retained_clusters
        if cluster["cluster_id"] in represented_cluster_ids
    ]
    if not retained_clusters:
        raise ValueError(f"q{case_id}: all retained paths disappeared after node deduplication")
    retained_ids = {cluster["cluster_id"] for cluster in retained_clusters}

    rows = [
        build_row(
            node["candidate"],
            node["cluster"],
            node["target_index"],
            node["target_type"],
            node["merged"],
        )
        for node in selected_nodes
    ]
    for row in rows:
        row["metadata"]["retained_path_count_for_case"] = len(retained_clusters)
        row["metadata"]["auxiliary_elimination_from_non_endpoint_path"] = (
            row["metadata"]["target_type"] == "hypothesis_elimination"
            and str(row["metadata"]["path_cluster_id"])
            not in endpoint_qualified_cluster_ids
        )

    raw_checkpoint_count = sum(len(candidate["messages"]) for candidate in candidates)
    compacted_checkpoint_count = sum(len(candidate["stages"]) for candidate in candidates)
    selection_case = {
        "case_id": case_id,
        "split": annotations[0]["split"],
        "source_trajectory_count": len(candidates),
        "raw_visible_checkpoint_count": raw_checkpoint_count,
        "causal_prefix_checkpoint_count": compacted_checkpoint_count,
        "path_cluster_count": len(clusters),
        "retained_path_cluster_count": len(retained_clusters),
        "selected_sft_node_count": len(rows),
        "evidence_summary_node_count": sum(
            row["metadata"]["target_type"] == "endpoint_bundle" for row in rows
        ),
        "hypothesis_elimination_node_count": sum(
            row["metadata"]["target_type"] == "hypothesis_elimination"
            for row in rows
        ),
        "decision_ready_node_count": sum(
            row["metadata"]["target_type"] == "endpoint_bundle" for row in rows
        ),
        "decision_node_count": sum(
            row["metadata"]["target_type"] == "endpoint_bundle" for row in rows
        ),
        "synthetic_path_endpoint_node_count": sum(
            row["metadata"]["synthetic_stage"] for row in rows
        ),
        "reduction_from_raw_visible_checkpoints": raw_checkpoint_count - len(rows),
        "clusters": [
            {
                "cluster_id": cluster["cluster_id"],
                "member_count": len(cluster["members"]),
                "member_trajectory_ids": sorted(member["raw"]["id"] for member in cluster["members"]),
                "representative_trajectory_id": cluster["representative"]["raw"]["id"],
                "representative_quality": cluster["representative"]["quality"],
                "representative_path_features": sorted(cluster["representative"]["path_features"]),
                "retained": cluster["cluster_id"] in retained_ids,
                "auxiliary_elimination_selected": (
                    cluster["cluster_id"] in path_elimination_sources
                    and cluster["cluster_id"] not in retained_ids
                ),
                "omission_reason": (
                    None
                    if cluster["cluster_id"] in retained_ids
                    else (
                        "endpoint_gate_failed_but_one_grounded_elimination_was_retained"
                        if cluster["cluster_id"] in path_elimination_sources
                        else "failed_same_snapshot_direct_evidence_gate"
                        if cluster["cluster_id"] in initially_retained_ids
                        and cluster["cluster_id"] not in endpoint_qualified_cluster_ids
                        else "all_nodes_merged_into_other_retained_paths"
                        if cluster["cluster_id"] in initially_retained_ids
                        else "low_support_or_redundant_path"
                    )
                ),
            }
            for cluster in clusters
        ],
        "selected_nodes": [
            {
                "row_id": row["id"],
                "target_type": row["metadata"]["target_type"],
                "trajectory_id": row["metadata"]["trajectory_id"],
                "path_cluster_id": row["metadata"]["path_cluster_id"],
                "source_message_index": row["metadata"]["source_message_index"],
                "synthetic_stage": row["metadata"]["synthetic_stage"],
                "synthetic_stage_type": row["metadata"]["synthetic_stage_type"],
                "causal_features": row["metadata"]["causal_features"],
                "merged_equivalent_node_count": row["metadata"]["merged_equivalent_node_count"],
            }
            for row in rows
        ],
        "candidates": [candidate_public(candidate) for candidate in candidates],
    }
    return rows, selection_case


def build_endpoint_schedule_rows(
    endpoint_pool: list[dict[str, Any]], epoch: int
) -> list[dict[str, Any]]:
    """Select complete summary/stop/decision path groups, balanced per query."""
    if not 1 <= epoch <= ENDPOINT_SCHEDULE_EPOCHS:
        raise ValueError(f"invalid endpoint schedule epoch: {epoch}")
    by_case_path: dict[int, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in endpoint_pool:
        metadata = row["metadata"]
        case_id = int(metadata["case_id"])
        path_id = str(metadata["path_cluster_id"])
        target_type = str(metadata["target_type"])
        if target_type not in ENDPOINT_TARGET_TYPES:
            raise ValueError(f"{row['id']}: non-endpoint row in endpoint pool")
        if target_type in by_case_path[case_id][path_id]:
            raise ValueError(
                f"q{case_id}/{path_id}: duplicate {target_type} endpoint"
            )
        by_case_path[case_id][path_id][target_type] = row

    scheduled: list[dict[str, Any]] = []
    for case_id in sorted(by_case_path):
        paths = sorted(by_case_path[case_id])
        for path_id in paths:
            missing = set(ENDPOINT_TARGET_TYPES) - set(
                by_case_path[case_id][path_id]
            )
            if missing:
                raise ValueError(
                    f"q{case_id}/{path_id}: incomplete endpoint group {sorted(missing)}"
                )
        offset = (epoch - 1) * ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH
        for slot in range(ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH):
            path_id = paths[(offset + slot) % len(paths)]
            for target_type in ENDPOINT_TARGET_TYPES:
                source = by_case_path[case_id][path_id][target_type]
                row = copy.deepcopy(source)
                row["id"] = (
                    f"{source['id']}_endpoint_epoch_{epoch:02d}_"
                    f"slot_{slot + 1:02d}"
                )
                row["metadata"]["training_sampling_role"] = (
                    "endpoint_group_epoch_schedule"
                )
                row["metadata"]["sampling_source_row_id"] = source["id"]
                row["metadata"]["endpoint_schedule_epoch"] = epoch
                row["metadata"]["endpoint_schedule_slot"] = slot + 1
                scheduled.append(row)
    return scheduled


def build_core_schedule_rows(
    core_pool: list[dict[str, Any]], epoch: int
) -> list[dict[str, Any]]:
    """Give every training query exactly the same number of core exposures."""
    if not 1 <= epoch <= ENDPOINT_SCHEDULE_EPOCHS:
        raise ValueError(f"invalid core schedule epoch: {epoch}")
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in core_pool:
        by_case[int(row["metadata"]["case_id"])].append(row)
    scheduled: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        target_priority = {
            "hypothesis_elimination": 0,
            "reasoning": 1,
            "planning": 2,
            "source_decision_ready": 3,
        }
        rows = sorted(
            by_case[case_id],
            key=lambda row: (
                target_priority.get(str(row["metadata"]["target_type"]), 9),
                str(row["id"]),
            ),
        )
        if not rows:
            raise ValueError(f"q{case_id}: empty core pool")
        # A deterministic per-query phase prevents the preferred elimination
        # row from concentrating in epoch 1 while preserving round-robin
        # coverage and equal query weight.
        case_phase = (case_id * 7) % len(rows)
        offset = case_phase + (epoch - 1) * CORE_EXPOSURES_PER_QUERY_PER_EPOCH
        for slot in range(CORE_EXPOSURES_PER_QUERY_PER_EPOCH):
            source = rows[(offset + slot) % len(rows)]
            row = copy.deepcopy(source)
            row["id"] = f"{source['id']}_core_epoch_{epoch:02d}_slot_{slot + 1:02d}"
            row["metadata"]["training_sampling_role"] = "case_balanced_core_epoch_schedule"
            row["metadata"]["sampling_source_row_id"] = source["id"]
            row["metadata"]["core_schedule_epoch"] = epoch
            row["metadata"]["core_schedule_slot"] = slot + 1
            scheduled.append(row)
    return scheduled


def heuristic_training_signal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate target-token and weighted-loss shares without a model tokenizer."""
    row_counts: Counter[str] = Counter()
    raw_tokens_by_target: Counter[str] = Counter()
    weighted_tokens_by_target: Counter[str] = Counter()
    weighted_tokens_by_message_kind: Counter[str] = Counter()
    auto_endpoint_weight = 0.0
    for row in rows:
        target_type = str(row["metadata"]["target_type"])
        row_counts[target_type] += 1
        for message in row["messages"]:
            loss_scale = float(message.get("loss_scale", 0) or 0)
            if loss_scale <= 0:
                continue
            token_count = estimate_context_token_count(str(message["content"]))
            raw_tokens_by_target[target_type] += token_count
            weighted_tokens_by_target[target_type] += token_count * loss_scale
            if (
                target_type == "endpoint_bundle"
                and message["role"] == "assistant"
                and "<result>" not in str(message["content"])
            ):
                auto_endpoint_weight += token_count * loss_scale
            if message["role"] == "tool_call":
                message_kind = "tool_call"
            elif message["role"] == "assistant" and "<think>" in message["content"]:
                message_kind = "thinking"
            else:
                message_kind = "conclusion_or_answer"
            weighted_tokens_by_message_kind[message_kind] += token_count * loss_scale
    raw_total = sum(raw_tokens_by_target.values())
    weighted_total = sum(weighted_tokens_by_target.values())
    if raw_total <= 0 or weighted_total <= 0:
        raise ValueError("heuristic training signal audit has no supervised tokens")
    return {
        "method": "one CJK code point is approximately one token; other text is approximately four characters per token",
        "row_count": len(rows),
        "target_type_rows": dict(sorted(row_counts.items())),
        "raw_target_token_estimate": raw_total,
        "weighted_token_estimate": round(weighted_total, 6),
        "weighted_message_kind_percent": {
            key: round(100.0 * value / weighted_total, 6)
            for key, value in sorted(weighted_tokens_by_message_kind.items())
        },
        "weighted_target_type_percent": {
            key: round(100.0 * value / weighted_total, 6)
            for key, value in sorted(weighted_tokens_by_target.items())
        },
        "auto_summary_stop_weighted_percent": round(
            100.0 * auto_endpoint_weight / weighted_total, 6
        ),
    }


def main() -> None:
    if infer_families("saved_configs/CampusNetwork_01/PE1/display_lldp_neighbor_brief.txt") != {"lldp"}:
        raise ValueError("LLDP family regression: LLDP must not be inferred as MPLS/LDP")
    if "mpls" not in infer_families(
        "saved_configs/CampusNetwork_01/PE1/display_mpls_ldp_lsp.txt"
    ):
        raise ValueError("MPLS/LDP family regression: standalone LDP was not detected")
    if is_safe_procedural_sentence("路径核对发现关键局部异常") or is_safe_procedural_sentence(
        "我已把候选范围收敛到路径接口，下一步继续检查"
    ):
        raise ValueError("mixed factual/procedural regression was accepted")
    release_status, tokenizer_report = target_tokenizer_release_status()
    source = base.load_json(SOURCE_CURATION)
    frozen = base.load_json(FROZEN_0804_CURATION)
    source_train = set(source["split"]["train_case_ids"])
    source_validation = set(source["split"]["validation_case_ids"])
    frozen_train = set(frozen["split"]["train_case_ids"])
    frozen_validation = set(frozen["split"]["validation_case_ids"])
    if (source_train, source_validation) != (frozen_train, frozen_validation):
        raise ValueError("0807 source split differs from the frozen 0804 split")

    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in source["trajectories"]:
        if annotation.get("selected"):
            by_case[int(annotation["case_id"])].append(annotation)
    if set(by_case) != source_train | source_validation:
        raise ValueError("eligible case inventory differs from the frozen split")

    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    case_documents: list[dict[str, Any]] = []
    target_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    for case_id in sorted(by_case):
        rows, case_document = process_case(case_id, by_case[case_id])
        split = case_document["split"]
        if split not in target_counts:
            raise ValueError(f"q{case_id}: invalid split {split}")
        target_counts[split].update(row["metadata"]["target_type"] for row in rows)
        (train_rows if split == "train" else validation_rows).extend(rows)
        case_documents.append(case_document)

    selection_document = {
        "schema_version": "0807-evidence-gated-cluster-selection.v6",
        "status": release_status,
        "source_curation": SOURCE_CURATION.relative_to(ROOT).as_posix(),
        "source_curation_sha256_lf_normalized": base.digest_file(SOURCE_CURATION),
        "frozen_0804_curation": FROZEN_0804_CURATION.relative_to(ROOT).as_posix(),
        "frozen_0804_curation_sha256_lf_normalized": base.digest_file(FROZEN_0804_CURATION),
        "policy": {
            "trajectory_cluster_basis": "ordered exact snapshot/device/file-query/family evidence path",
            "trajectory_cluster_threshold": TRAJECTORY_CLUSTER_THRESHOLD,
            "max_retained_paths_per_case": MAX_RETAINED_PATHS_PER_CASE,
            "singleton_policy": "retain only at-or-above median quality when nonredundant",
            "action_policy": "successful source commands only; label-independent same-snapshot ranking; exact prefix/interface/VRID/file signatures prevent semantically different queries from collapsing",
            "supervised_command_normalization": "strictly translate admitted read-only PowerShell to Codex CLI exec_command Linux commands with repository-relative saved_configs paths; retain exact archived source and inner PowerShell for audit",
            "action_selection": "minimal label-independent next-action-intent cover; no fixed numerical action cap; publish source and final-supervision full/partial/zero coverage; all snapshot/device/filename globs are ineligible",
            "node_duplicate_threshold": NODE_DUPLICATE_THRESHOLD,
            "hypothesis_elimination_policy": "retain at most one source rejection per query after splitting it into factual clauses; every supervised clause binds exact earlier non-header observation spans and action IDs; unsupported clauses are deleted",
            "path_endpoint_policy": "generate the summary/stop/decision group only after label-independent direct facts from visible same-snapshot history pass a domain gate; VRRP role-misalignment additionally requires source-host IPv4/VLAN -> matching Vlanif Master -> explicit MST VLAN/instance mapping -> Alternate/Discarding on that same instance for every target device; summary/stop must not contain the verified answer",
            "final_decision_policy": "retain only evidence-qualified path decisions; guarantee at least one endpoint per query, allowing minimal same-query cross-trajectory evidence recovery with exact action provenance",
            "case_balancing_policy": "sample exactly two nonduplicated core rows and one single-row multi-target endpoint bundle per training query per epoch; rotate both pools deterministically across five epochs",
            "infrastructure_failures_and_interruptions": "not archived, clustered, counted, or trained",
        },
        "counts": {
            "cases": len(case_documents),
            "source_trajectories": sum(case["source_trajectory_count"] for case in case_documents),
            "raw_visible_checkpoints": sum(case["raw_visible_checkpoint_count"] for case in case_documents),
            "causal_prefix_checkpoints": sum(case["causal_prefix_checkpoint_count"] for case in case_documents),
            "path_clusters": sum(case["path_cluster_count"] for case in case_documents),
            "retained_path_clusters": sum(case["retained_path_cluster_count"] for case in case_documents),
            "selected_sft_nodes": sum(case["selected_sft_node_count"] for case in case_documents),
            "evidence_summary_nodes": sum(case["evidence_summary_node_count"] for case in case_documents),
            "hypothesis_elimination_nodes": sum(
                case["hypothesis_elimination_node_count"]
                for case in case_documents
            ),
            "decision_ready_nodes": sum(case["decision_ready_node_count"] for case in case_documents),
            "decision_nodes": sum(case["decision_node_count"] for case in case_documents),
            "synthetic_path_endpoint_nodes": sum(
                case["synthetic_path_endpoint_node_count"] for case in case_documents
            ),
        },
        "cases": case_documents,
    }
    base.write_json(CLUSTER_SELECTION, selection_document)

    for row in train_rows:
        row["metadata"]["training_sampling_role"] = "semantic_pool_not_for_direct_training"
    train_core_rows = [
        copy.deepcopy(row)
        for row in train_rows
        if row["metadata"]["target_type"] not in ENDPOINT_TARGET_TYPES
    ]
    for row in train_core_rows:
        row["metadata"]["training_sampling_role"] = "core_pool_not_for_direct_training"
    train_endpoint_pool_rows = [
        copy.deepcopy(row)
        for row in train_rows
        if row["metadata"]["target_type"] in ENDPOINT_TARGET_TYPES
    ]
    for row in train_endpoint_pool_rows:
        row["metadata"]["training_sampling_role"] = "endpoint_pool_not_for_direct_training"
    core_schedules = {
        epoch: build_core_schedule_rows(train_core_rows, epoch)
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    endpoint_schedules = {
        epoch: build_endpoint_schedule_rows(train_endpoint_pool_rows, epoch)
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    training_signal_audits = {
        epoch: heuristic_training_signal_audit(
            [*core_schedules[epoch], *endpoint_schedules[epoch]]
        )
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    train_output = base.write_jsonl(TRAIN_OUTPUT, train_rows)
    train_core_output = base.write_jsonl(TRAIN_CORE_OUTPUT, train_core_rows)
    train_endpoint_pool_output = base.write_jsonl(
        TRAIN_ENDPOINT_POOL_OUTPUT, train_endpoint_pool_rows
    )
    core_schedule_outputs = {
        epoch: base.write_jsonl(core_schedule_output(epoch), rows)
        for epoch, rows in core_schedules.items()
    }
    endpoint_schedule_outputs = {
        epoch: base.write_jsonl(endpoint_schedule_output(epoch), rows)
        for epoch, rows in endpoint_schedules.items()
    }
    for obsolete_path in OBSOLETE_OUTPUTS:
        if obsolete_path.exists():
            obsolete_path.unlink()
    validation_output = base.write_jsonl(VALIDATION_OUTPUT, validation_rows)
    manifest = {
        "schema_version": "qwen36-0807-evidence-gated-case-balanced-sft.v7",
        "status": release_status,
        "scope": "data/2026-08-07 only",
        "reproducibility": {
            "document": REPRODUCIBILITY_DOC.relative_to(ROOT).as_posix(),
            "document_sha256_lf_normalized": base.digest_file(REPRODUCIBILITY_DOC),
            "change_policy": "every source, curation, split, conversion, loss, sampling, prompt, tool protocol, tokenizer, training-entry, validator, or generated-data change must update documentation, regenerate all artifacts, and pass independent validation",
            "tracked_files": {
                name: {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256_lf_normalized": base.digest_file(path),
                }
                for name, path in REPRODUCIBILITY_FILES.items()
            },
        },
        "source_curation": SOURCE_CURATION.relative_to(ROOT).as_posix(),
        "source_curation_sha256_lf_normalized": base.digest_file(SOURCE_CURATION),
        "reference_answer_policy": {
            "source_dataset": SOURCE_DATASET.relative_to(ROOT).as_posix(),
            "source_dataset_sha256_lf_normalized": base.digest_file(SOURCE_DATASET),
            "q73_q86": "inclusive OR accepts singleton Core_SW_01, singleton Core_SW_02, or the two-device set; current accepted trajectories remain singleton",
            "sft_endpoint_preference": "supervise the evidence-strongest singleton target; admit a dual target only when both devices independently satisfy the full VLAN/instance closure",
        },
        "cluster_selection": CLUSTER_SELECTION.relative_to(ROOT).as_posix(),
        "cluster_selection_sha256_lf_normalized": base.digest_file(CLUSTER_SELECTION),
        "system_prompt": {
            "role": "system",
            "source": CODEX_MODEL_CATALOG.relative_to(ROOT).as_posix(),
            "source_sha256_lf_normalized": base.digest_file(CODEX_MODEL_CATALOG),
            "model_slug": CODEX_CLI_MODEL_SLUG,
            "field": "base_instructions",
            "content_sha256": CODEX_CLI_SYSTEM_PROMPT_SHA256,
            "content": CODEX_CLI_SYSTEM_PROMPT,
        },
        "split": {
            "source": "frozen data/2026-08-04 case split",
            "group_key": "case_id",
            "train_case_count": len(source_train),
            "validation_case_count": len(source_validation),
            "train_validation_case_intersection": [],
            "validation_case_ids": sorted(source_validation),
        },
        "counts": {
            **selection_document["counts"],
            "train_sft_rows": len(train_rows),
            "train_core_rows": len(train_core_rows),
            "train_core_schedule_rows_per_epoch": len(core_schedules[1]),
            "train_endpoint_pool_rows": len(train_endpoint_pool_rows),
            "train_endpoint_schedule_rows_per_epoch": len(endpoint_schedules[1]),
            "train_endpoint_groups_per_epoch": (
                len(endpoint_schedules[1]) // len(ENDPOINT_TARGET_TYPES)
            ),
            "effective_train_row_exposures_per_epoch": (
                len(core_schedules[1]) + len(endpoint_schedules[1])
            ),
            "validation_sft_rows": len(validation_rows),
            "target_types": {
                split: dict(sorted(counts.items())) for split, counts in target_counts.items()
            },
        },
        "conversion": {
            "ten_successful_trajectories_per_case": True,
            "trajectory_path_clustering": True,
            "cross_trajectory_node_deduplication": True,
            "one_sample_per_retained_valuable_checkpoint": True,
            "at_most_one_source_grounded_hypothesis_elimination_per_query": True,
            "evidence_gated_endpoints_only": True,
            "all_training_queries_have_at_least_one_evidence_gated_endpoint": True,
            "verified_label_present_in_summary_or_stop": False,
            "verified_label_used_only_for_posthoc_endpoint_gate": True,
            "core_exposures_per_query_per_epoch": CORE_EXPOSURES_PER_QUERY_PER_EPOCH,
            "endpoint_group_exposures_per_query_per_epoch": ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH,
            "endpoint_schedule_epochs": ENDPOINT_SCHEDULE_EPOCHS,
            "endpoint_sampling_policy": "retain one path-specific multi-target bundle per qualified path; each bundle jointly supervises evidence summary, cautious stop judgment, and final decision without repeating the context; each epoch chooses one bundle per query by deterministic round-robin",
            "hypothesis_elimination_policy": "use at most one source rejection per query only after each retained factual clause binds exact earlier non-header observation spans and action IDs; delete every unsupported clause; allow an endpoint-unqualified source cluster only as an explicitly auxiliary elimination",
            "investigative_action_selection": "greedy minimal label-independent next-action-intent coverage with no fixed action count cap; scope device/family pairs to local clauses to avoid Cartesian false positives; separately archive source-intent and final-supervised-intent full/partial/zero coverage plus explicit action bindings; 0-to-0 is never called a success",
            "action_family_policy": "derive LLDP as an independent topology family; infer standalone LDP/MPLS tokens with alphanumeric boundaries so LLDP can never satisfy MPLS intent",
            "original_message_policy": "retain only pure future procedural clauses verbatim; split mixed factual/procedural sentences at clause punctuation, rewrite each bindable fact into exact earlier non-header observation atoms, and delete every unbound factual clause; keep source text only in metadata; empty/header/help/truncated-config omission text cannot support a positive or absence claim",
            "path_endpoint_text_policy": "extract label-independent atomic facts from same-snapshot visible history; compare them with the verified label only as a final gate; summary/stop use natural evidence-specific language, do not expose internal gate IDs, and state support for the current candidate rather than proof that every alternative was exhausted",
            "endpoint_gate_policy": "STP disabled, BPDU filter, VRRP non-preempt, strict source-host VLAN -> Vlanif Master -> MST VLAN/instance -> same-instance Alternate/Discarding role misalignment, or a three-device IP/MPLS cycle whose next-hop ownership is derived from visible interface-address facts; every device in a dual VRRP target needs its own full closure; cross-snapshot and snapshot/device/filename glob evidence is ineligible",
            "same_query_recovery_policy": "when no single successful run contains every atom, compose only real successful read-only actions from the same query and same snapshot, retain full provenance, and reduce their visible result to the selected exact lines",
            "command_selection": "label-independent ranking of exact successful source commands by target snapshot, concrete target, information family, and source score; retain the minimal claim-cover set without a fixed numerical cap and strictly translate the admitted read-only subset",
            "supervised_command_normalization": "exec_command + cmd + Linux read-only utilities + repository-relative saved_configs paths; unsupported source commands fail generation instead of receiving tool loss",
            "source_command_audit": "retain exact archived command, canonical inner PowerShell, translation kind, and pre/post hashes in metadata",
            "command_results": "relevant source excerpts with source hashes; PowerShell discovery/search presentation is normalized to Linux find/grep paths and all responses use the Codex CLI function-call-output shape; context-only transport token counts are heuristic",
            "future_results_in_current_target": False,
            "infrastructure_failures_and_interruptions": "not present and never recorded",
        },
        "loss_policy": {
            "thinking_by_source": THINKING_LOSS_SCALE_BY_SOURCE,
            "conclusion_by_source": CONCLUSION_LOSS_SCALE_BY_SOURCE,
            "current_tool_calls": TOOL_CALL_LOSS_SCALE,
            "historical_assistant_and_tool_calls": base.HISTORY_LOSS_SCALE,
            "tool_responses": "context_only",
            "required_cli": "--loss_scale default --is_binary_loss_scale false",
        },
        "heuristic_training_signal_by_epoch": {
            f"epoch_{epoch:02d}": audit
            for epoch, audit in training_signal_audits.items()
        },
        "training_profile": {
            "model": "Qwen3.6-27B",
            "tuner_type": "lora",
            "distributed_strategy": "ddp",
            "world_size": 2,
            "cuda_visible_devices": "0,1",
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_length": 16384,
            "truncation_strategy": "delete",
            "preserve_thinking": True,
            "add_non_thinking_prefix": False,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "checkpoint_selection_strategy": "fixed_epoch",
            "fixed_validation_epoch": 3,
            "tokenizer_preflight_required": True,
            "semantic_train_pool": TRAIN_OUTPUT.relative_to(ROOT).as_posix(),
            "semantic_train_pool_not_for_direct_training": True,
            "core_pool_not_for_direct_training": True,
            "endpoint_pool_not_for_direct_training": True,
            "train_dataset_components_by_epoch": {
                f"epoch_{epoch:02d}": [
                    core_schedule_output(epoch).relative_to(ROOT).as_posix(),
                    endpoint_schedule_output(epoch).relative_to(ROOT).as_posix(),
                ]
                for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
            },
            "formal_entry": "scripts/train_qwen36_0807_evidence_gated_5epoch.sh; prepare, dry-run, and train modes; five sequential full-state resume stages",
        },
        "target_tokenizer_preflight": (
            {
                "report": TOKENIZER_PREFLIGHT_REPORT.relative_to(ROOT).as_posix(),
                "report_sha256_lf_normalized": base.digest_file(
                    TOKENIZER_PREFLIGHT_REPORT
                ),
                "schema_version": tokenizer_report["schema_version"],
                "status": tokenizer_report["status"],
                "model_identity_files": tokenizer_report["model_identity_files"],
                "template": tokenizer_report["template"],
                "totals": tokenizer_report["totals"],
            }
            if release_status == "rule_and_target_tokenizer_validated_release_candidate"
            and tokenizer_report is not None
            else {
                "status": "required_before_formal_training",
                "report": TOKENIZER_PREFLIGHT_REPORT.relative_to(ROOT).as_posix(),
            }
        ),
        "comparison_experiment_plan": {
            "reference": "config/qwen36_0805_formal_training.json",
            "epochs": 5,
            "distributed_strategy": "ddp",
            "world_size": 2,
            "cuda_visible_devices": "0,1",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
            "train_rows_per_stage": 216,
            "optimizer_steps_per_stage": 27,
            "expected_global_step_boundaries": [0, 27, 54, 81, 108, 135],
            "checkpoint_suffix_by_stage": [27, 54, 81, 108, 135],
            "training_postcondition": "each stage must audit exactly 27 consecutive step_begin events, exact start/end global steps, and checkpoint-<expected_end_step>",
            "seed": 42,
            "data_seed": 42,
            "shuffle_each_epoch": True,
            "fixed_learning_rate_by_epoch": [2e-5, 1.5e-5, 1e-5, 6e-6, 3e-6],
            "epoch_specific_endpoint_schedule_required": True,
            "epoch_specific_core_schedule_required": True,
            "sequential_one_epoch_resume_required": True,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0,
            "endpoint_group_exposures_per_query_per_epoch": ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH,
            "endpoint_path_rotation_coverage": "all retained train paths appear across the five schedules; summary, stop, and decision always share one selected path and within-query path exposure differs by at most one",
            "checkpoint_policy": "evaluate and save at every epoch; retain all five; eval loss is diagnostic only; always validate epoch 3",
            "checkpoint_selection_strategy": "fixed_epoch",
            "fixed_validation_epoch": 3,
            "fixed_validation_checkpoint_suffix": 81,
            "agent_checkpoint_selection": False,
            "final_agent_validation": {
                "selected_epoch": 3,
                "selected_checkpoint_suffix": 81,
                "case_ids": sorted(source_validation),
                "repeats_per_case": 5,
                "total_attempts": 60,
                "reuse_checkpoint_selection_attempts": 0,
                "new_attempts": 60
            },
            "infrastructure_failures_and_interruptions": "excluded from samples and denominators",
        },
        "outputs": {
            "train": train_output,
            "train_core": train_core_output,
            "train_endpoint_pool": train_endpoint_pool_output,
            **{
                f"train_core_epoch_{epoch:02d}": output
                for epoch, output in core_schedule_outputs.items()
            },
            **{
                f"train_endpoint_epoch_{epoch:02d}": output
                for epoch, output in endpoint_schedule_outputs.items()
            },
            "validation": validation_output,
        },
    }
    base.write_json(MANIFEST_OUTPUT, manifest)
    q1 = next(case for case in case_documents if case["case_id"] == 1)
    print(
        f"Clustered {selection_document['counts']['source_trajectories']} trajectories "
        f"across {len(case_documents)} cases"
    )
    print(
        f"Retained paths={selection_document['counts']['retained_path_clusters']}; "
        f"semantic rows: train={len(train_rows)}, validation={len(validation_rows)}; "
        f"per-epoch train exposures={len(core_schedules[1]) + len(endpoint_schedules[1])}"
    )
    print(
        f"q0001: raw checkpoints={q1['raw_visible_checkpoint_count']}, "
        f"clusters={q1['path_cluster_count']}, retained paths={q1['retained_path_cluster_count']}, "
        f"selected nodes={q1['selected_sft_node_count']}"
    )
    print(f"Selection: {CLUSTER_SELECTION.relative_to(ROOT).as_posix()}")
    print(f"Manifest: {MANIFEST_OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
