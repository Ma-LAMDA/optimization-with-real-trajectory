#!/usr/bin/env python3
"""Extract only generator-safe fields from the source JSONL.

This process is the sole pre-generation reader of the source JSONL.  It decodes
only allow-listed values and skips every other JSON value without materializing
it, so the answer field never enters a generator-visible object or artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_TOP_LEVEL = {"id", "question", "output_format"}
PROMPT_KEYS = {"user_prompt", "user prompt", "prompt", "userPrompt"}


class ParseError(ValueError):
    pass


def skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t\r\n":
        index += 1
    return index


def skip_string(text: str, index: int) -> int:
    if index >= len(text) or text[index] != '"':
        raise ParseError(f"expected string at offset {index}")
    index += 1
    while index < len(text):
        char = text[index]
        if char == '"':
            return index + 1
        if char == "\\":
            index += 2
        else:
            index += 1
    raise ParseError("unterminated string")


def skip_value(text: str, index: int) -> int:
    index = skip_ws(text, index)
    if index >= len(text):
        raise ParseError("missing value")
    char = text[index]
    if char == '"':
        return skip_string(text, index)
    if char in "[{":
        opener = char
        closer = "]" if opener == "[" else "}"
        depth = 1
        index += 1
        while index < len(text) and depth:
            char = text[index]
            if char == '"':
                index = skip_string(text, index)
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
            elif opener == "[" and char == "{":
                index = skip_value(text, index)
                continue
            elif opener == "{" and char == "[":
                index = skip_value(text, index)
                continue
            index += 1
        if depth:
            raise ParseError("unterminated compound value")
        return index
    start = index
    while index < len(text) and text[index] not in ",}] \t\r\n":
        index += 1
    token = text[start:index]
    if token in {"true", "false", "null"}:
        return index
    try:
        float(token)
    except ValueError as exc:
        raise ParseError(f"invalid primitive at offset {start}") from exc
    return index


def decode_value(text: str, index: int) -> tuple[Any, int]:
    index = skip_ws(text, index)
    value, end = json.JSONDecoder().raw_decode(text, index)
    return value, end


def extract_metadata_prompt(text: str, index: int) -> tuple[str | None, int]:
    index = skip_ws(text, index)
    if index >= len(text) or text[index] != "{":
        return None, skip_value(text, index)
    index += 1
    found: str | None = None
    while True:
        index = skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            return found, index + 1
        key, index = decode_value(text, index)
        if not isinstance(key, str):
            raise ParseError("metadata key is not a string")
        index = skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise ParseError("missing metadata colon")
        index += 1
        if key in PROMPT_KEYS:
            value, index = decode_value(text, index)
            if not isinstance(value, str) or not value.strip():
                raise ParseError(f"metadata {key!r} is not a non-empty string")
            if found is not None and value != found:
                raise ParseError("metadata contains conflicting user prompts")
            found = value
        else:
            index = skip_value(text, index)
        index = skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return found, index + 1
        raise ParseError("invalid metadata object separator")


def extract_safe_record(text: str) -> tuple[dict[str, Any], str | None]:
    index = skip_ws(text, 0)
    if index >= len(text) or text[index] != "{":
        raise ParseError("record is not an object")
    index += 1
    result: dict[str, Any] = {}
    metadata_prompt: str | None = None
    seen: set[str] = set()
    while True:
        index = skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            index += 1
            break
        key, index = decode_value(text, index)
        if not isinstance(key, str):
            raise ParseError("record key is not a string")
        if key in seen:
            raise ParseError(f"duplicate top-level key {key!r}")
        seen.add(key)
        index = skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise ParseError("missing record colon")
        index += 1
        if key in ALLOWED_TOP_LEVEL:
            result[key], index = decode_value(text, index)
        elif key == "metadata":
            metadata_prompt, index = extract_metadata_prompt(text, index)
        else:
            index = skip_value(text, index)
        index = skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            index += 1
            break
        raise ParseError("invalid record object separator")
    if skip_ws(text, index) != len(text):
        raise ParseError("trailing data after record")
    missing = ALLOWED_TOP_LEVEL.difference(result)
    if missing:
        raise ParseError(f"missing safe fields: {sorted(missing)}")
    if not isinstance(result["id"], int) or isinstance(result["id"], bool):
        raise ParseError("id is not an integer")
    for field in ("question", "output_format"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ParseError(f"{field} is not a non-empty string")
    return result, metadata_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--localized-template", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    prompts: list[str | None] = []
    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ParseError(f"blank source line {row_index}")
            safe, prompt = extract_safe_record(raw_line.rstrip("\r\n"))
            safe["row_index"] = row_index
            records.append(safe)
            prompts.append(prompt)
    if len(records) != 100:
        raise ParseError(f"expected 100 records, found {len(records)}")

    ids = [record["id"] for record in records]
    duplicated_ids = {value for value in ids if ids.count(value) > 1}
    source_template = args.template.read_text(encoding="utf-8-sig")
    metadata_present = [value is not None for value in prompts]
    if any(metadata_present) and not all(metadata_present):
        raise ParseError("metadata user prompt is present for only some rows")
    if all(metadata_present):
        unique_prompts = set(prompts)
        if len(unique_prompts) != 1:
            raise ParseError("per-row metadata prompts differ; unsupported safe batching")
    prompt_source = "experiment inputs/IP user prompt by text.txt"

    documented_url = "http://127.0.0.1:3080"
    from urllib.parse import urlsplit

    parsed_local = urlsplit(args.base_url)
    if parsed_local.scheme != "http" or parsed_local.hostname != "127.0.0.1" or not parsed_local.port:
        raise ParseError("local service base must be an explicit http://127.0.0.1:PORT URL")
    if source_template.count(documented_url) != 1:
        raise ParseError("prompt must contain the documented local service URL exactly once")
    if "saved_configs/" not in source_template:
        raise ParseError("prompt must name the configuration root as saved_configs/")
    localized = source_template.replace(documented_url, args.base_url)
    replacement = {
        "operation": "local service port adaptation only",
        "from": documented_url,
        "to": args.base_url,
    }
    for placeholder in ("{original_query}", "{output_format}"):
        if localized.count(placeholder) != 1:
            raise ParseError(f"prompt must contain {placeholder!r} exactly once")

    args.localized_template.parent.mkdir(parents=True, exist_ok=True)
    if args.localized_template.exists():
        if args.localized_template.read_text(encoding="utf-8") != localized:
            raise ParseError("existing localized template conflicts with current task")
    else:
        args.localized_template.write_text(localized, encoding="utf-8", newline="\n")

    index_records = []
    for safe in records:
        original_id = safe["id"]
        row_index = safe["row_index"]
        unique_key = (
            f"row{row_index:03d}"
            if original_id in duplicated_ids
            else f"q{original_id:04d}"
        )
        index_records.append(
            {
                "row_index": row_index,
                "original_id": original_id,
                "sample_id": original_id,
                "sample_key": unique_key,
                "question": safe["question"],
                "output_format": safe["output_format"],
            }
        )
    payload = {
        "schema_version": "ip-distill-safe-input-index.v1",
        "source_sha256": sha256(args.dataset),
        "record_count": len(index_records),
        "duplicate_original_ids": sorted(duplicated_ids),
        "prompt_source": prompt_source,
        "dataset_metadata_prompt_present": all(metadata_present),
        "dataset_metadata_prompt_used": False,
        "prompt_localization": replacement,
        "localized_template_sha256": hashlib.sha256(localized.encode("utf-8")).hexdigest(),
        "records": index_records,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"record_count": len(index_records), "source_sha256": payload["source_sha256"], "prompt_source": prompt_source}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
