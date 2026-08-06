#!/usr/bin/env python3
"""Build the 0805 causal-path-clustered weighted multi-turn SFT dataset."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shlex
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import convert_0804_best_trajectory_reasoning_sft as base


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "2026-08-05"
SOURCE_CURATION = DATA_ROOT / "curation" / "accepted_trajectory_selection.json"
FROZEN_0804_CURATION = (
    ROOT / "data" / "2026-08-04" / "curation" / "accepted_trajectory_selection.json"
)
CLUSTER_SELECTION = DATA_ROOT / "curation" / "causal_path_clusters_per_case.json"
SFT_DIR = DATA_ROOT / "sft"
TRAIN_OUTPUT = SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train.jsonl"
TRAIN_CORE_OUTPUT = SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train_core.jsonl"
TRAIN_ENDPOINT_POOL_OUTPUT = (
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train_endpoint_pool.jsonl"
)
VALIDATION_OUTPUT = SFT_DIR / "qwen3_6_27b_reasoning_causal_path_validation.jsonl"
MANIFEST_OUTPUT = SFT_DIR / "reasoning_causal_path_manifest.json"
REPRODUCIBILITY_DOC = DATA_ROOT / "REPRODUCIBILITY.md"
CODEX_MODEL_CATALOG = ROOT / "config" / "codex_qwen_model_catalog.json"
CODEX_CLI_MODEL_SLUG = "Qwen3.6-27B-trained"

MAX_RETAINED_PATHS_PER_CASE = 4
MAX_ACTIONS_PER_STAGE = 2
TRAJECTORY_CLUSTER_THRESHOLD = 0.46
NODE_DUPLICATE_THRESHOLD = 0.76
TOOL_CALL_LOSS_SCALE = 0.10
THINKING_LOSS_SCALE_BY_SOURCE = {
    "pruned_original_visible_agent_message": 0.60,
    "evidence_grounded_bridge": 0.20,
    "fixed_bridge_template": 0.00,
    "minimal_final_bridge": 0.00,
}
CONCLUSION_LOSS_SCALE_BY_SOURCE = {
    "original_visible_conclusion": 1.00,
    "evidence_aligned_reconstruction": 0.40,
    "source_grounded_hypothesis_elimination": 0.60,
    "path_evidence_synthesis": 0.20,
    "verified_path_stop_judgment": 0.20,
    "verified_final_answer": 1.00,
}
ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH = 2
ENDPOINT_SCHEDULE_EPOCHS = 5
ENDPOINT_TARGET_TYPES = ("evidence_summary", "decision_ready", "decision")
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
    return SFT_DIR / f"qwen3_6_27b_reasoning_causal_path_train_endpoint_epoch_{epoch:02d}.jsonl"


OBSOLETE_DECISION_OUTPUTS = [
    SFT_DIR / "qwen3_6_27b_reasoning_causal_path_train_decision_pool.jsonl",
    *[
        SFT_DIR / f"qwen3_6_27b_reasoning_causal_path_train_decision_epoch_{epoch:02d}.jsonl"
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    ],
]


REPRODUCIBILITY_FILES = {
    "repository_rules": ROOT / "AGENTS.md",
    "root_readme": ROOT / "README.md",
    "date_readme": DATA_ROOT / "README.md",
    "reproducibility_document": REPRODUCIBILITY_DOC,
    "archive_converter": ROOT / "scripts" / "convert_accepted_only_100x10_to_sft.py",
    "base_reasoning_converter": ROOT / "scripts" / "convert_0804_best_trajectory_reasoning_sft.py",
    "causal_path_converter": Path(__file__).resolve(),
    "causal_path_validator": ROOT / "scripts" / "validate_0805_causal_path_reasoning_sft.py",
    "tokenizer_preflight": ROOT / "scripts" / "check_0804_best1_token_lengths.py",
    "training_entry": ROOT / "scripts" / "train_qwen36_0805_causal_path_quick.sh",
}

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
    r"CampusNetwork[^\\/'\"]*[\\/]+([^\\/'\"]+)[\\/]+([^\\/'\"]+\.txt)",
    re.IGNORECASE,
)
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*|\n+")

FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
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
    stem = re.sub(r"(?:\d{1,3}\.){3}\d{1,3}(?:_\d+)?", "<ip>", stem)
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
    lowered = text.lower()
    families = {
        family
        for family, patterns in FAMILY_PATTERNS
        if any(pattern in lowered for pattern in patterns)
    }
    if any(marker in lowered for marker in DISCOVERY_MARKERS):
        families.add("discovery")
    return families or {"other"}


def action_semantics(action: dict[str, Any]) -> dict[str, Any]:
    command = str(action["command"])
    matches = PATH_PATTERN.findall(command.replace("\\\\", "\\"))
    targets = sorted(
        {
            f"{device.lower()}/{normalize_filename(filename)}"
            for device, filename in matches
            if device not in {"*", "**"}
        }
    )
    devices = sorted({target.split("/", 1)[0] for target in targets})
    # The command/file name describes the requested evidence class.  Do not
    # derive clustering features from the returned configuration body: a full
    # config often mentions every protocol and would make unrelated paths look
    # artificially identical.
    families = sorted(infer_families(command))
    if targets:
        keys = sorted(
            {
                f"{device}/{family}"
                for device in devices
                for family in families
                if family not in {"discovery", "other"}
            }
        )
    else:
        keys = sorted(f"unknown/{family}" for family in families if family != "other")
    return {
        "targets": targets,
        "devices": devices,
        "families": families,
        "keys": keys,
        "is_discovery": "discovery" in families,
    }


def fault_context(raw: dict[str, Any]) -> tuple[set[str], str]:
    devices, reasons = base.fault_parts(raw)
    if len(set(reasons)) != 1:
        raise ValueError(f"{raw['id']}: expected one merged fault type")
    return {device.lower() for device in devices}, reasons[0]


def action_priority(
    action: dict[str, Any], semantics: dict[str, Any], raw: dict[str, Any]
) -> float:
    fault_devices, reason = fault_context(raw)
    families = set(semantics["families"])
    devices = set(semantics["devices"])
    score = float(action.get("selection_score", 0.0)) / 20.0
    score += 8.0 * bool(families & PRIMARY_FAMILIES[reason])
    score += 3.0 * bool(families & SUPPORT_FAMILIES[reason])
    score += 7.0 * bool(devices & fault_devices)
    score -= 7.0 * bool(families & LOW_VALUE_FAMILIES)
    score -= 4.0 * semantics["is_discovery"]
    return round(score, 6)


def prune_actions(
    actions: list[dict[str, Any]], raw: dict[str, Any], stage_index: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluated: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for action in actions:
        semantics = action_semantics(action)
        evaluated.append((action_priority(action, semantics, raw), action, semantics))

    non_discovery = [item for item in evaluated if not item[2]["is_discovery"]]
    pool = non_discovery if non_discovery else evaluated
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

    ranked = sorted(
        deduplicated.values(),
        key=lambda item: (-item[0], int(item[1]["event_line"])),
    )[:MAX_ACTIONS_PER_STAGE]
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
    return kept, omitted


def sentence_concepts(text: str) -> set[str]:
    concepts = infer_families(text) - {"other", "discovery"}
    concepts.update(value.lower() for value in re.findall(r"\b(?:PE\d|[A-Za-z]+_SW_\d+)\b", text))
    return concepts


def action_description(actions: list[dict[str, Any]]) -> str:
    pairs: list[str] = []
    for action in actions:
        semantics = action["causal_semantics"]
        devices = semantics["devices"] or ["相关节点"]
        families = [family for family in semantics["families"] if family != "other"]
        family = FAMILY_NAMES.get(families[0] if families else "other", "补充状态")
        for device in devices[:2]:
            value = f"{device} 的{family}"
            if value not in pairs:
                pairs.append(value)
    return "、".join(pairs[:4]) or "最有区分度的运行状态"


def evidence_observation(action: dict[str, Any], raw: dict[str, Any]) -> str:
    """Select one compact, path-specific observation from an admitted result."""
    _, fault_reason = fault_context(raw)
    semantics = action["causal_semantics"]
    keywords = {
        fault_reason.lower(),
        *(alias.lower() for alias in base.FAULT_ALIASES.get(fault_reason, ())),
        *(str(value).lower() for value in semantics["devices"]),
        *(str(value).lower() for value in semantics["families"]),
    }
    status_markers = {
        "disable", "disabled", "enable", "enabled", "down", "up", "error",
        "inactive", "active", "unreachable", "reachable", "not", " no ",
        "best", "valid", "select", "advertised", "cost", "preference", "localpref",
        "未", "无", "关闭", "开启", "故障", "异常", "正常", "丢包", "时延",
    }
    candidates: list[tuple[float, int, str]] = []
    for index, source_line in enumerate(str(action.get("output_excerpt", "")).splitlines()):
        line = re.sub(r"\s+", " ", source_line).strip()
        if (
            not line
            or len(line) < 3
            or re.fullmatch(r"[-=+|<>\s]+", line)
            or line.startswith(("(ed):", "...", "Legend:", "Flags:"))
            or re.match(r"^\([^)]+\):", line)
            or line.lower().startswith(("display ", "route flags:"))
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


def optimize_stage_text(
    original_text: str,
    original_thinking: str,
    original_conclusion: str,
    actions: list[dict[str, Any]],
    raw: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    fault_devices, reason = fault_context(raw)
    action_concepts = set()
    action_devices = set()
    for action in actions:
        action_concepts.update(action["causal_semantics"]["families"])
        action_devices.update(action["causal_semantics"]["devices"])
    aliases = {reason.lower(), *(alias.lower() for alias in base.FAULT_ALIASES.get(reason, ()))}
    relevant_terms = action_concepts | action_devices | fault_devices | aliases
    source_sentences = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(original_thinking)
        if sentence.strip()
    ]
    kept_sentences = [
        sentence
        for sentence in source_sentences
        if any(term in sentence.lower() for term in relevant_terms)
        or any(marker in sentence for marker in ("证据", "排除", "假设", "路径", "下一步"))
    ]
    kept_sentences = kept_sentences[:4]
    if kept_sentences:
        thinking = "".join(kept_sentences)
    else:
        thinking = (
            "当前证据尚不足以完成最小归因，需要沿已出现的业务路径继续取证，"
            "并优先检查能够区分剩余候选的状态。"
        )

    conclusion_overlap = sentence_concepts(original_conclusion) & action_concepts
    if actions and conclusion_overlap:
        conclusion = original_conclusion
        conclusion_source = "original_visible_conclusion"
    elif actions:
        conclusion = (
            f"下一步核对{action_description(actions)}，用于区分剩余候选；"
            "在结果返回前不提前输出最终答案。"
        )
        conclusion_source = "evidence_aligned_reconstruction"
    else:
        conclusion = original_conclusion
        conclusion_source = "original_visible_conclusion"

    return thinking, conclusion, {
        "source_message_sha256_lf_normalized": base.digest_text(original_text),
        "source_thinking_sentence_count": len(source_sentences),
        "retained_thinking_sentence_count": len(kept_sentences),
        "thinking_source": "pruned_original_visible_agent_message" if kept_sentences else "fixed_bridge_template",
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


def prepare_stages(
    raw: dict[str, Any], messages: list[dict[str, Any]], commands: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_stages = base.build_stage_data(raw, messages, commands)
    retained: list[dict[str, Any]] = []
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

        kept_actions, omitted_actions = prune_actions(stage["actions"], raw, source_index)
        stage["actions"] = kept_actions
        thinking, conclusion, optimization = optimize_stage_text(
            messages[source_index]["text"],
            stage["thinking"],
            stage["conclusion"],
            kept_actions,
            raw,
        )
        stage["thinking"] = thinking
        stage["conclusion"] = conclusion
        stage["text_optimization"] = optimization
        stage["omitted_actions"] = omitted_actions
        stage["causal_features"] = sorted(stage_feature_set(stage))

        source_text = messages[source_index]["text"]
        explicit_stop = base.reason_is_explicit(source_text, base.fault_parts(raw)[1]) or any(
            marker in messages[source_index]["text"] for marker in ("证据收敛", "最小根因", "最终只输出")
        )
        contradicts_verified_root = base.reason_is_explicit(
            source_text, base.fault_parts(raw)[1]
        ) and bool(
            re.search(
                r"(?:不能把|不可把|不应把).{0,60}(?:当作|视为).{0,20}根因"
                r"|(?:不是|并非).{0,30}根因"
                r"|不支持.{0,40}(?:根因|故障假设)",
                source_text,
            )
        )
        if contradicts_verified_root:
            omissions.append(
                {
                    "source_message_index": source_index,
                    "reason": "contradicts_verified_final_root_and_is_a_detour",
                    "omitted_action_count": len(omitted_actions),
                }
            )
            continue
        if not kept_actions and not explicit_stop:
            omissions.append(
                {
                    "source_message_index": source_index,
                    "reason": "no_retained_causal_action_or_stop_signal",
                    "omitted_action_count": len(omitted_actions),
                }
            )
            continue
        retained.append(stage)

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
    if stage["is_final"] or stage["text_optimization"]["thinking_source"] != (
        "pruned_original_visible_agent_message"
    ):
        return []
    statements = [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(stage["thinking"])
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


def elimination_statement_terms(statements: list[str]) -> set[str]:
    terms = {
        value.lower()
        for value in re.findall(
            r"[A-Za-z][A-Za-z0-9_./<>:-]{1,}|(?:\d+\.){1,3}\d+|\d+%",
            " ".join(statements),
        )
        if len(value) >= 2
    }
    return {
        value
        for value in terms
        if value not in {"ip", "wan", "lan"}
        and not re.fullmatch(r"pe\d|[a-z]+_sw_\d+", value)
    }


def elimination_evidence_actions(
    candidate: dict[str, Any], target_index: int, statements: list[str]
) -> list[dict[str, Any]]:
    """Select already-returned tool evidence that grounds an elimination statement."""
    statement_text = " ".join(statements).lower()
    statement_families = sentence_concepts(statement_text)
    statement_terms = elimination_statement_terms(statements)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for stage in candidate["stages"][:target_index]:
        for action in stage["actions"]:
            semantics = action["causal_semantics"]
            if semantics["is_discovery"]:
                continue
            observation = evidence_observation(action, candidate["raw"])
            if observation.endswith("已成功返回可复核结果"):
                continue
            families = set(semantics["families"])
            devices = {str(value).lower() for value in semantics["devices"]}
            family_overlap = len(families & statement_families)
            device_overlap = sum(device in statement_text for device in devices)
            target = str(action.get("target") or "").lower()
            target_overlap = bool(target and any(part in statement_text for part in target.split("/")))
            output_text = str(action.get("output_excerpt") or "").lower()
            output_overlap = sum(term in output_text for term in statement_terms)
            if family_overlap == 0 and output_overlap == 0:
                continue
            score = (
                float(action["causal_priority"])
                + 5.0 * family_overlap
                + 4.0 * device_overlap
                + 1.5 * target_overlap
                + 3.0 * output_overlap
            )
            ranked.append((score, action))
    ranked.sort(
        key=lambda item: (
            -item[0],
            str(item[1]["action_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, action in ranked:
        action_id = str(action["action_id"])
        if action_id in seen:
            continue
        selected.append(action)
        seen.add(action_id)
        if len(selected) == 2:
            break
    return selected


def elimination_observation_description(
    actions: list[dict[str, Any]], raw: dict[str, Any], statements: list[str]
) -> str:
    """Quote result lines that overlap the exact visible elimination statement."""
    statement_text = " ".join(statements).lower()
    statement_terms = elimination_statement_terms(statements)
    observations: list[str] = []
    for action in actions[:2]:
        candidates: list[tuple[float, int, str]] = []
        for index, source_line in enumerate(
            str(action.get("output_excerpt", "")).splitlines()
        ):
            line = re.sub(r"\s+", " ", source_line).strip()
            if not line or len(line) < 3 or re.fullmatch(r"[-=+|<>\s]+", line):
                continue
            lowered = line.lower()
            overlap = sum(term in lowered for term in statement_terms)
            if overlap == 0:
                continue
            status_bonus = sum(
                marker in lowered
                for marker in (
                    "up", "down", "enable", "disable", "error", "drop", "crc",
                    "cost", "best", "next", "route", "lsp", "mtu", "0",
                )
            )
            candidates.append((5.0 * overlap + status_bonus, -index, line[:180]))
        observation = (
            max(candidates)[2]
            if candidates
            else evidence_observation(action, raw)
        )
        observations.append(
            f"{elimination_action_description([action])}回显“{observation}”"
        )
    return "；".join(observations)


def choose_path_elimination_source(
    cluster: dict[str, Any],
) -> tuple[dict[str, Any], int, list[str], list[dict[str, Any]]] | None:
    """Choose one faithful, evidence-backed elimination node from a path cluster."""
    ranked: list[
        tuple[tuple[float, ...], dict[str, Any], int, list[str], list[dict[str, Any]]]
    ] = []
    representative_id = str(cluster["representative"]["raw"]["id"])
    for candidate in cluster["members"]:
        for target_index, stage in enumerate(candidate["stages"][:-1]):
            if target_index == 0:
                continue
            statements = source_elimination_statements(stage)
            if not statements:
                continue
            evidence_actions = elimination_evidence_actions(
                candidate, target_index, statements
            )
            if not evidence_actions:
                continue
            score = (
                float(len(evidence_actions)),
                float(sum(len(value) for value in statements)),
                float(candidate["quality"]["score"]),
                float(str(candidate["raw"]["id"]) == representative_id),
                -float(target_index),
                -float(candidate["raw"]["attempt_index"]),
            )
            ranked.append(
                (score, candidate, target_index, statements, evidence_actions)
            )
    if not ranked:
        return None
    _, candidate, target_index, statements, evidence_actions = max(
        ranked, key=lambda item: item[0]
    )
    return candidate, target_index, statements, evidence_actions


def build_hypothesis_elimination_candidate(
    candidate: dict[str, Any],
    target_index: int,
    statements: list[str],
    evidence_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Turn one real visible elimination step into an explicit supervised target."""
    augmented = copy.deepcopy(candidate)
    stage = augmented["stages"][target_index]
    evidence_description = elimination_action_description(evidence_actions)
    evidence_observation_text = elimination_observation_description(
        evidence_actions, candidate["raw"], statements
    )
    exact_statement = "".join(statements)
    stage["conclusion"] = f"候选排除：{exact_statement}"
    stage["source_hypothesis_elimination"] = True
    stage["rejected_candidate_statements"] = statements
    stage["elimination_evidence_action_count"] = len(evidence_actions)
    stage["elimination_evidence_action_ids"] = [
        str(action["action_id"]) for action in evidence_actions
    ]
    stage["elimination_evidence_description"] = evidence_description
    stage["elimination_evidence_observation"] = evidence_observation_text
    stage["text_optimization"] = {
        **stage["text_optimization"],
        "conclusion_source": "source_grounded_hypothesis_elimination",
        "source_elimination_statements": statements,
        "elimination_evidence_action_ids": stage["elimination_evidence_action_ids"],
        "elimination_derived_from_visible_source": True,
    }
    augmented["path_endpoint_policy"] = {
        "evidence_summary_nodes": 1,
        "decision_ready_nodes": 1,
        "decision_nodes": 1,
        "sampling_unit": "path_endpoint_group",
    }
    return augmented


def build_path_endpoint_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Add a grounded evidence summary and explicit stop judgment to one path."""
    augmented = copy.deepcopy(candidate)
    source_stages = candidate["stages"]
    evidence_stages = [
        copy.deepcopy(stage)
        for stage in source_stages[:-1]
        if stage["actions"]
    ]
    if not evidence_stages:
        raise ValueError(f"{candidate['raw']['id']}: retained path has no evidence action")

    evidence_actions = [
        action for stage in evidence_stages for action in stage["actions"]
    ]
    fault_devices, fault_reason = fault_context(candidate["raw"])
    device_decisive_actions = [
        action
        for action in evidence_actions
        if set(action["causal_semantics"]["devices"]) & fault_devices
    ]
    specific_primary_families = PRIMARY_FAMILIES[fault_reason] - {"config"}
    device_primary_actions = [
        action
        for action in device_decisive_actions
        if set(action["causal_semantics"]["families"]) & specific_primary_families
    ]
    family_decisive_actions = [
        action
        for action in evidence_actions
        if set(action["causal_semantics"]["families"]) & specific_primary_families
    ]
    decisive_actions = (
        device_primary_actions or device_decisive_actions or family_decisive_actions
    )
    if not decisive_actions:
        decisive_actions = sorted(
            evidence_actions,
            key=lambda action: -float(action["causal_priority"]),
        )[:2]
    else:
        decisive_actions = sorted(
            decisive_actions,
            key=lambda action: -float(action["causal_priority"]),
        )[:2]
    decisive_action_ids = {str(action["action_id"]) for action in decisive_actions}
    exclusion_candidates = sorted(
        (
            action
            for action in evidence_actions
            if str(action["action_id"]) not in decisive_action_ids
            and not action["causal_semantics"]["is_discovery"]
        ),
        key=lambda action: -float(action["causal_priority"]),
    )
    exclusion_actions = [
        action
        for action in exclusion_candidates
        if not evidence_observation(action, candidate["raw"]).endswith(
            "已成功返回可复核结果"
        )
    ][:2]
    evidence_message_indices = [
        int(stage["source_message_index"]) for stage in evidence_stages
    ]
    evidence_features = sorted(
        set().union(*(set(stage.get("causal_features", [])) for stage in evidence_stages))
    )
    evidence_description = action_description(decisive_actions)
    observation_description = grounded_observation_description(
        decisive_actions, candidate["raw"]
    )
    exclusion_description = (
        action_description(exclusion_actions)
        if exclusion_actions
        else "题目给定信息与该路径的其余候选故障点"
    )
    exclusion_observation = (
        grounded_observation_description(exclusion_actions, candidate["raw"])
        if exclusion_actions
        else exclusion_description
    )
    verified_items = [str(item) for item in candidate["raw"]["actual_result_items"]]
    if not verified_items:
        raise ValueError(f"{candidate['raw']['id']}: verified final result is empty")
    verified_root = "、".join(verified_items)
    provenance_hash = base.digest_text(
        "\n".join(
            [
                *(str(action["excerpt_sha256_lf_normalized"]) for action in evidence_actions),
                *verified_items,
            ]
        )
    )
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
        "source_decisive_action_ids": [
            str(action["action_id"]) for action in decisive_actions
        ],
        "grounded_evidence_description": evidence_description,
        "grounded_evidence_observation": observation_description,
        "excluded_evidence_description": exclusion_description,
        "excluded_evidence_observation": exclusion_observation,
        "derived_from_verified_final_answer": True,
    }
    evidence_summary = {
        **copy.deepcopy(common),
        "synthetic_stage_type": "evidence_summary",
        "thinking": (
            f"{evidence_description}的结果已经返回，其中{observation_description}。"
            f"现在需要核对排除项“{exclusion_observation}”是否给出相反证据，并把当前路径压缩为最小证据链。"
        ),
        "conclusion": (
            f"证据归纳：{observation_description}，与已验证根因“{verified_root}”直接一致。"
            f"作为排除项，{exclusion_observation}未形成能够推翻该归因的反证；"
            f"因此当前最小根因集合仍为“{verified_root}”。"
        ),
        "text_optimization": {
            "thinking_source": "evidence_grounded_bridge",
            "conclusion_source": "path_evidence_synthesis",
            "hidden_chain_of_thought_claimed": False,
            "source_message_sha256_lf_normalized": provenance_hash,
            "source_evidence_message_indices": evidence_message_indices,
            "derived_from_verified_final_answer": True,
        },
    }
    stop_judgment = {
        **copy.deepcopy(common),
        "synthetic_stage_type": "decision_ready",
        "thinking": (
            f"停止条件已经具体满足：{observation_description}直接支持“{verified_root}”，"
            f"而{exclusion_observation}没有提供相反证据。继续取证不会改变当前最小根因集合。"
        ),
        "conclusion": (
            f"停止判断：已由{evidence_description}确认“{verified_root}”，并核对"
            f"{exclusion_observation}而未发现反证；最小根因集合已经收敛。"
            "因此停止调用工具，下一步直接按题目要求输出最终答案。"
        ),
        "text_optimization": {
            "thinking_source": "evidence_grounded_bridge",
            "conclusion_source": "verified_path_stop_judgment",
            "hidden_chain_of_thought_claimed": False,
            "source_message_sha256_lf_normalized": provenance_hash,
            "source_evidence_message_indices": evidence_message_indices,
            "derived_from_verified_final_answer": True,
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
        "sampling_unit": "path_endpoint_group",
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
        return "decision_ready"
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
    synthetic_stage_type = stage.get("synthetic_stage_type")
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
            "source_evidence_message_indices": stage.get(
                "source_evidence_message_indices", []
            ),
            "source_evidence_action_count": stage.get(
                "source_evidence_action_count", 0
            ),
            "source_decisive_action_count": stage.get(
                "source_decisive_action_count", 0
            ),
            "source_decisive_action_ids": stage.get(
                "source_decisive_action_ids", []
            ),
            "grounded_evidence_description": stage.get(
                "grounded_evidence_description"
            ),
            "grounded_evidence_observation": stage.get(
                "grounded_evidence_observation"
            ),
            "excluded_evidence_description": stage.get(
                "excluded_evidence_description"
            ),
            "excluded_evidence_observation": stage.get(
                "excluded_evidence_observation"
            ),
            "rejected_candidate_statements": stage.get(
                "rejected_candidate_statements", []
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
            "merged_equivalent_nodes": merged_sources,
            "merged_equivalent_node_count": len(merged_sources),
            "evidence_converged_without_next_tool_call": target_type == "decision_ready",
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
            "tool_protocol": {
                "name": "exec_command",
                "command_argument": "cmd",
                "operating_system": "linux",
                "path_style": "repository_relative_saved_configs",
                "source_command_audit": "exact PowerShell retained in current_actions.source_command",
                "tool_response_transport": "Codex CLI shape with Linux-normalized discovery/search output; token count is heuristic and context-only",
            },
            "path_endpoint_policy": candidate.get("path_endpoint_policy"),
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
                "quality": base.candidate_quality(annotation, raw, messages, commands),
            }
        )
    if len(candidates) != 10:
        raise ValueError(f"q{case_id}: expected 10 accepted trajectories, found {len(candidates)}")

    clusters = cluster_trajectories(candidates)
    retained_clusters = retain_path_clusters(clusters, candidates)
    initially_retained_ids = {cluster["cluster_id"] for cluster in retained_clusters}

    path_candidates = {
        cluster["cluster_id"]: build_path_endpoint_candidate(cluster["representative"])
        for cluster in retained_clusters
    }
    path_elimination_sources = {
        cluster["cluster_id"]: source
        for cluster in retained_clusters
        if (source := choose_path_elimination_source(cluster)) is not None
    }
    chosen_representative_elimination_steps = {
        (str(source[0]["raw"]["id"]), int(source[1]))
        for cluster in retained_clusters
        if (source := path_elimination_sources.get(cluster["cluster_id"])) is not None
        and str(source[0]["raw"]["id"])
        == str(cluster["representative"]["raw"]["id"])
    }
    selected_nodes: list[dict[str, Any]] = []
    node_sources: list[dict[str, Any]] = []
    for cluster in retained_clusters:
        representative = path_candidates[cluster["cluster_id"]]
        for target_index, stage in enumerate(representative["stages"][:-3]):
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
    for cluster in retained_clusters:
        source = path_elimination_sources.get(cluster["cluster_id"])
        if source is None:
            continue
        candidate, target_index, statements, evidence_actions = source
        elimination_candidate = build_hypothesis_elimination_candidate(
            candidate, target_index, statements, evidence_actions
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

    # Endpoint nodes are intentionally not deduplicated across paths: each
    # retained causal history must teach its own evidence-to-stop transition.
    for cluster in retained_clusters:
        representative = path_candidates[cluster["cluster_id"]]
        for target_index in (len(representative["stages"]) - 3, len(representative["stages"]) - 2):
            stage = representative["stages"][target_index]
            selected_nodes.append(
                {
                    "candidate": representative,
                    "cluster": cluster,
                    "target_index": target_index,
                    "target_type": stage_target_type(stage, target_index),
                    "stage": stage,
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

    # Every retained causal path gets its own stop-to-answer transition. Query
    # weighting is handled later by an epoch-specific balanced sampler rather
    # than by dropping valid decision contexts here.
    for cluster in retained_clusters:
        final_candidate = path_candidates[cluster["cluster_id"]]
        final_index = len(final_candidate["stages"]) - 1
        selected_nodes.append(
            {
                "candidate": final_candidate,
                "cluster": cluster,
                "target_index": final_index,
                "target_type": "decision",
                "stage": final_candidate["stages"][final_index],
                "merged": [],
            }
        )

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
            row["metadata"]["target_type"] == "evidence_summary" for row in rows
        ),
        "hypothesis_elimination_node_count": sum(
            row["metadata"]["target_type"] == "hypothesis_elimination"
            for row in rows
        ),
        "decision_ready_node_count": sum(
            row["metadata"]["target_type"] == "decision_ready" for row in rows
        ),
        "decision_node_count": sum(
            row["metadata"]["target_type"] == "decision" for row in rows
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
                "omission_reason": (
                    None
                    if cluster["cluster_id"] in retained_ids
                    else (
                        "all_nodes_merged_into_other_retained_paths"
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


def heuristic_training_signal_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate target-token and weighted-loss shares without a model tokenizer."""
    row_counts: Counter[str] = Counter()
    raw_tokens_by_target: Counter[str] = Counter()
    weighted_tokens_by_target: Counter[str] = Counter()
    weighted_tokens_by_message_kind: Counter[str] = Counter()
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
    auto_endpoint_weight = sum(
        weighted_tokens_by_target[target_type]
        for target_type in ("evidence_summary", "decision_ready")
    )
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
    source = base.load_json(SOURCE_CURATION)
    frozen = base.load_json(FROZEN_0804_CURATION)
    source_train = set(source["split"]["train_case_ids"])
    source_validation = set(source["split"]["validation_case_ids"])
    frozen_train = set(frozen["split"]["train_case_ids"])
    frozen_validation = set(frozen["split"]["validation_case_ids"])
    if (source_train, source_validation) != (frozen_train, frozen_validation):
        raise ValueError("0805 source split differs from the frozen 0804 split")

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
        "schema_version": "0805-causal-path-cluster-selection.v4",
        "status": "auto_clustered_draft_requires_domain_review",
        "source_curation": SOURCE_CURATION.relative_to(ROOT).as_posix(),
        "source_curation_sha256_lf_normalized": base.digest_file(SOURCE_CURATION),
        "frozen_0804_curation": FROZEN_0804_CURATION.relative_to(ROOT).as_posix(),
        "frozen_0804_curation_sha256_lf_normalized": base.digest_file(FROZEN_0804_CURATION),
        "policy": {
            "trajectory_cluster_basis": "ordered device-by-protocol causal evidence path",
            "trajectory_cluster_threshold": TRAJECTORY_CLUSTER_THRESHOLD,
            "max_retained_paths_per_case": MAX_RETAINED_PATHS_PER_CASE,
            "singleton_policy": "retain only at-or-above median quality when nonredundant",
            "action_policy": "successful source commands only; remove duplicate, housekeeping, low-value, and causally weak actions",
            "supervised_command_normalization": "strictly translate admitted read-only PowerShell to Codex CLI exec_command Linux commands with repository-relative saved_configs paths; retain exact archived source and inner PowerShell for audit",
            "max_actions_per_stage": MAX_ACTIONS_PER_STAGE,
            "node_duplicate_threshold": NODE_DUPLICATE_THRESHOLD,
            "hypothesis_elimination_policy": "mine at most one exact visible-source elimination statement per retained path when already-returned successful tool evidence can be attached; never synthesize a missing rejection",
            "path_endpoint_policy": "one grounded evidence_summary, one explicit decision_ready, and one verified decision per retained path; sample the three nodes as an indivisible endpoint group",
            "final_decision_policy": "retain every verified path decision in the semantic pool; equal query/path-group exposure is enforced by epoch schedules",
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

    train_core_rows = [
        row
        for row in train_rows
        if row["metadata"]["target_type"] not in ENDPOINT_TARGET_TYPES
    ]
    train_endpoint_pool_rows = [
        row
        for row in train_rows
        if row["metadata"]["target_type"] in ENDPOINT_TARGET_TYPES
    ]
    endpoint_schedules = {
        epoch: build_endpoint_schedule_rows(train_endpoint_pool_rows, epoch)
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    training_signal_audits = {
        epoch: heuristic_training_signal_audit(
            [*train_core_rows, *endpoint_schedules[epoch]]
        )
        for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
    }
    train_output = base.write_jsonl(TRAIN_OUTPUT, train_rows)
    train_core_output = base.write_jsonl(TRAIN_CORE_OUTPUT, train_core_rows)
    train_endpoint_pool_output = base.write_jsonl(
        TRAIN_ENDPOINT_POOL_OUTPUT, train_endpoint_pool_rows
    )
    endpoint_schedule_outputs = {
        epoch: base.write_jsonl(endpoint_schedule_output(epoch), rows)
        for epoch, rows in endpoint_schedules.items()
    }
    for obsolete_path in OBSOLETE_DECISION_OUTPUTS:
        if obsolete_path.exists():
            obsolete_path.unlink()
    validation_output = base.write_jsonl(VALIDATION_OUTPUT, validation_rows)
    manifest = {
        "schema_version": "qwen36-0805-causal-path-reasoning-sft.v8",
        "status": "auto_clustered_draft_requires_domain_review",
        "scope": "data/2026-08-05 only",
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
            "train_endpoint_pool_rows": len(train_endpoint_pool_rows),
            "train_endpoint_schedule_rows_per_epoch": len(endpoint_schedules[1]),
            "train_endpoint_groups_per_epoch": (
                len(endpoint_schedules[1]) // len(ENDPOINT_TARGET_TYPES)
            ),
            "effective_train_row_exposures_per_epoch": (
                len(train_core_rows) + len(endpoint_schedules[1])
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
            "one_source_grounded_hypothesis_elimination_per_eligible_retained_path": True,
            "one_evidence_summary_per_retained_path": True,
            "one_stop_judgment_per_retained_path": True,
            "one_final_decision_per_retained_path": True,
            "endpoint_group_exposures_per_query_per_epoch": ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH,
            "endpoint_schedule_epochs": ENDPOINT_SCHEDULE_EPOCHS,
            "endpoint_sampling_policy": "retain every path-specific evidence_summary, decision_ready, and decision in the semantic pool; for each epoch choose exactly two paths per query by deterministic round-robin and expose all three endpoint nodes from each chosen path",
            "hypothesis_elimination_policy": "use exact visible-source rejection statements only when one or two already-returned successful source actions provide concrete evidence; retain source statement, action IDs, observations, and hashes in metadata",
            "max_actions_per_investigative_stage": MAX_ACTIONS_PER_STAGE,
            "original_message_policy": "prune visible source reasoning to evidence-aligned sentences; fixed bridge templates remain context-only; never claim hidden chain-of-thought",
            "path_endpoint_text_policy": "name path-specific device/config evidence, quote a compact observed result, state checked exclusions, and explain why further tools cannot change the minimum root-cause set; synthetic summary/stop thinking and conclusions use reduced loss 0.20",
            "command_selection": "causally rank exact successful source commands, cap every investigative stage at two discriminative actions, then strictly translate the admitted read-only subset to Codex CLI exec_command calls",
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
            "lora_rank": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_length": 16384,
            "truncation_strategy": "delete",
            "preserve_thinking": True,
            "add_non_thinking_prefix": False,
            "tokenizer_preflight_required": True,
            "semantic_train_pool": TRAIN_OUTPUT.relative_to(ROOT).as_posix(),
            "train_dataset_components_by_epoch": {
                f"epoch_{epoch:02d}": [
                    TRAIN_CORE_OUTPUT.relative_to(ROOT).as_posix(),
                    endpoint_schedule_output(epoch).relative_to(ROOT).as_posix(),
                ]
                for epoch in range(1, ENDPOINT_SCHEDULE_EPOCHS + 1)
            },
            "quick_smoke_entry": "one epoch at a time; select TRAIN_EPOCH_INDEX=1..5 so each run uses the matching balanced path-endpoint-group schedule",
        },
        "comparison_experiment_plan": {
            "reference": "experiments/2026-08-04-qwen36-27b-best1-agent-validation/README.md#下一轮5-epoch实验方案已确认未执行",
            "epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "effective_batch_size": 8,
            "seed": 42,
            "data_seed": 42,
            "shuffle_each_epoch": True,
            "fixed_learning_rate_by_epoch": [2e-5, 1.5e-5, 1e-5, 6e-6, 3e-6],
            "epoch_specific_endpoint_schedule_required": True,
            "sequential_one_epoch_resume_required": True,
            "endpoint_group_exposures_per_query_per_epoch": ENDPOINT_EXPOSURES_PER_QUERY_PER_EPOCH,
            "endpoint_path_rotation_coverage": "all retained train paths appear across the five schedules; summary, stop, and decision always share one selected path and within-query path exposure differs by at most one",
            "checkpoint_policy": "evaluate and save at every epoch; retain all five; do not auto-select solely by eval loss",
            "checkpoint_agent_selection": {
                "case_ids": [12, 20, 38, 71, 86, 100],
                "repeats_per_case_per_checkpoint": 2,
                "reasoning_effort": "high",
                "ranking": [
                    "strict_accuracy_desc",
                    "model_hard_timeout_asc",
                    "average_duration_asc",
                    "sft_eval_loss_asc",
                    "epoch_asc"
                ]
            },
            "final_agent_validation": {
                "case_ids": sorted(source_validation),
                "repeats_per_case": 5,
                "total_attempts": 60,
                "reuse_selected_checkpoint_selection_attempts": 12,
                "new_attempts_after_checkpoint_selection": 48
            },
            "infrastructure_failures_and_interruptions": "excluded from samples and denominators",
        },
        "outputs": {
            "train": train_output,
            "train_core": train_core_output,
            "train_endpoint_pool": train_endpoint_pool_output,
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
        f"per-epoch train exposures={len(train_core_rows) + len(endpoint_schedules[1])}"
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
