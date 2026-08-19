#!/usr/bin/env python3
"""Insert raw Codex rollout reasoning into a ``codex exec --json`` event stream.

Codex persists raw reasoning in its session rollout even when ``codex exec
--json`` omits reasoning items from stdout.  This utility locates the rollout
by the stream's ``thread.started`` id, aligns each reasoning response with the
next visible model action, and atomically enriches ``events.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class ReasoningRecord:
    source_item_id: str
    text: str
    anchor_kind: str | None
    anchor_value: str | None
    insert_index: int | None = None
    insert_before_item_id: str | None = None
    alignment: str = "unmatched_append"


@dataclass
class ActionRecord:
    kind: str
    value: str
    item_id: str
    insert_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")),
    )
    parser.add_argument("--require-reasoning", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").strip()


def compact_space(value: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(value))


def command_fingerprint(value: str) -> str:
    # Codex renders shell tool arguments as ``/bin/bash -lc`` commands and
    # escapes nested quotes.  Removing only shell quoting/escape characters
    # preserves paths, operators, and command order while making the rollout
    # argument comparable with the emitted command_execution item.
    return compact_space(value).replace("\\", "").replace('"', "").replace("'", "")


def message_text(payload: dict[str, Any]) -> str | None:
    direct = payload.get("text")
    if isinstance(direct, str) and direct.strip():
        return normalize_text(direct)
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict)
        and part.get("type") in {"output_text", "text"}
        and isinstance(part.get("text"), str)
    ]
    text = "".join(parts)
    return normalize_text(text) if text.strip() else None


def function_command(payload: dict[str, Any]) -> str | None:
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return compact_space(arguments) if arguments.strip() else None
    if isinstance(arguments, dict):
        for key in ("command", "cmd"):
            command = arguments.get(key)
            if isinstance(command, str) and command.strip():
                return compact_space(command)
        return compact_space(json.dumps(arguments, ensure_ascii=False, sort_keys=True))
    return None


def reasoning_text(payload: dict[str, Any]) -> str | None:
    content = payload.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "reasoning_text"
        and isinstance(part.get("text"), str)
    ]
    text = "".join(parts)
    return text if text.strip() else None


def thread_id_from_events(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("type") == "thread.started" and isinstance(
            event.get("thread_id"), str
        ):
            return event["thread_id"]
    raise ValueError("events stream has no thread.started id")


def find_rollout(sessions_root: Path, thread_id: str, events_path: Path) -> Path:
    # Session directories are date-partitioned.  Search a bounded window first
    # so large long-lived CODEX_HOME directories remain cheap to inspect.
    event_time = datetime.fromtimestamp(events_path.stat().st_mtime, timezone.utc)
    matches: list[Path] = []
    for offset in range(-2, 3):
        candidate_date = event_time + timedelta(days=offset)
        directory = (
            sessions_root
            / f"{candidate_date.year:04d}"
            / f"{candidate_date.month:02d}"
            / f"{candidate_date.day:02d}"
        )
        matches.extend(directory.glob(f"rollout-*-{thread_id}.jsonl"))
    if not matches:
        matches = list(sessions_root.rglob(f"rollout-*-{thread_id}.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one rollout for thread {thread_id}, found {len(matches)}"
        )
    return matches[0]


def extract_reasoning(rollout: list[dict[str, Any]]) -> list[ReasoningRecord]:
    records: list[ReasoningRecord] = []
    for index, record in enumerate(rollout):
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "reasoning":
            continue
        text = reasoning_text(payload)
        if not text:
            continue

        anchor_kind: str | None = None
        anchor_value: str | None = None
        for following in rollout[index + 1 :]:
            following_payload = following.get("payload")
            if following.get("type") != "response_item" or not isinstance(
                following_payload, dict
            ):
                continue
            following_type = following_payload.get("type")
            if following_type == "reasoning":
                break
            if following_type == "message":
                value = message_text(following_payload)
                if value:
                    anchor_kind, anchor_value = "message", value
                    break
            if following_type == "function_call":
                value = function_command(following_payload)
                if value:
                    anchor_kind, anchor_value = "command", value
                    break

        source_item_id = payload.get("id")
        if not isinstance(source_item_id, str) or not source_item_id:
            source_item_id = digest_bytes(text.encode("utf-8"))[:16]
        records.append(
            ReasoningRecord(
                source_item_id=source_item_id,
                text=text,
                anchor_kind=anchor_kind,
                anchor_value=anchor_value,
            )
        )
    return records


def existing_actions(events: list[dict[str, Any]]) -> list[ActionRecord]:
    first_index_by_id: dict[str, int] = {}
    for index, event in enumerate(events):
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            first_index_by_id.setdefault(item["id"], index)

    actions: list[ActionRecord] = []
    for index, event in enumerate(events):
        if event.get("type") != "item.completed":
            continue
        item = event.get("item")
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        item_type = item.get("type")
        if item_type == "agent_message" and isinstance(item.get("text"), str):
            actions.append(
                ActionRecord("message", normalize_text(item["text"]), item["id"], index)
            )
        elif item_type == "command_execution" and isinstance(
            item.get("command"), str
        ):
            actions.append(
                ActionRecord(
                    "command",
                    compact_space(item["command"]),
                    item["id"],
                    first_index_by_id.get(item["id"], index),
                )
            )
    return actions


def action_matches(reasoning: ReasoningRecord, action: ActionRecord) -> bool:
    if reasoning.anchor_kind != action.kind or not reasoning.anchor_value:
        return False
    if action.kind == "message":
        return normalize_text(reasoning.anchor_value) == normalize_text(action.value)
    expected = command_fingerprint(reasoning.anchor_value)
    actual = command_fingerprint(action.value)
    return expected == actual or expected in actual or actual in expected


def align_reasoning(
    reasoning_records: list[ReasoningRecord], actions: list[ActionRecord]
) -> None:
    action_cursor = 0
    for reasoning in reasoning_records:
        match_index: int | None = None
        if reasoning.anchor_kind and reasoning.anchor_value:
            for index in range(action_cursor, len(actions)):
                if action_matches(reasoning, actions[index]):
                    match_index = index
                    reasoning.alignment = f"exact_{reasoning.anchor_kind}"
                    break
        if match_index is None and action_cursor < len(actions):
            match_index = action_cursor
            reasoning.alignment = "ordinal_fallback"
        if match_index is None:
            continue
        action = actions[match_index]
        reasoning.insert_index = action.insert_index
        reasoning.insert_before_item_id = action.item_id
        action_cursor = match_index + 1


def synthetic_event(
    reasoning: ReasoningRecord, thread_id: str, rollout_name: str
) -> dict[str, Any]:
    return {
        "type": "item.completed",
        "item": {
            "id": f"reasoning_{reasoning.source_item_id}",
            "type": "reasoning",
            "text": reasoning.text,
            "source": "codex_session_rollout",
            "source_thread_id": thread_id,
            "source_rollout": rollout_name,
            "source_item_id": reasoning.source_item_id,
            "text_sha256": digest_bytes(reasoning.text.encode("utf-8")),
            "alignment": reasoning.alignment,
            "insert_before_item_id": reasoning.insert_before_item_id,
        },
    }


def enrich_events(
    events: list[dict[str, Any]],
    reasoning_records: list[ReasoningRecord],
    thread_id: str,
    rollout_name: str,
) -> list[dict[str, Any]]:
    insertions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    append_events: list[dict[str, Any]] = []
    for reasoning in reasoning_records:
        event = synthetic_event(reasoning, thread_id, rollout_name)
        if reasoning.insert_index is None:
            append_events.append(event)
        else:
            insertions[reasoning.insert_index].append(event)

    enriched: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        enriched.extend(insertions.get(index, []))
        enriched.append(event)
    enriched.extend(append_events)
    return enriched


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def update_metadata(
    events_path: Path,
    events: list[dict[str, Any]],
    thread_id: str,
    rollout: Path,
    reasoning_records: list[ReasoningRecord],
    before_sha256: str,
) -> None:
    metadata_path = events_path.with_name("metadata.json")
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    event_counts: Counter[str] = Counter()
    item_counts: Counter[str] = Counter()
    for event in events:
        event_counts[str(event.get("type", "unknown"))] += 1
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("type"), str):
            item_counts[item["type"]] += 1
    metadata["event_type_counts"] = dict(sorted(event_counts.items()))
    metadata["item_type_counts"] = dict(sorted(item_counts.items()))
    metadata.setdefault("sha256", {})["events"] = digest_file(events_path)
    metadata["reasoning_capture"] = {
        "status": "captured",
        "source": "codex_session_rollout",
        "thread_id": thread_id,
        "source_rollout": rollout.name,
        "source_rollout_sha256": digest_file(rollout),
        "events_sha256_before_enrichment": before_sha256,
        "reasoning_items": len(reasoning_records),
        "reasoning_characters": sum(len(record.text) for record in reasoning_records),
        "alignment_counts": dict(
            sorted(Counter(record.alignment for record in reasoning_records).items())
        ),
        "reasoning_text_sha256": digest_bytes(
            "\0".join(record.text for record in reasoning_records).encode("utf-8")
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    events_path = args.events.resolve()
    events = load_jsonl(events_path)
    native_reasoning = [
        event
        for event in events
        if event.get("type") == "item.completed"
        and isinstance(event.get("item"), dict)
        and event["item"].get("type") == "reasoning"
        and isinstance(event["item"].get("text"), str)
        and event["item"]["text"].strip()
    ]
    if native_reasoning:
        print(
            json.dumps(
                {
                    "events": str(events_path),
                    "status": "already_enriched_or_native",
                    "reasoning_items": len(native_reasoning),
                },
                ensure_ascii=False,
            )
        )
        return 0

    thread_id = thread_id_from_events(events)
    rollout = find_rollout(args.codex_home.resolve() / "sessions", thread_id, events_path)
    rollout_records = load_jsonl(rollout)
    reasoning_records = extract_reasoning(rollout_records)
    if not reasoning_records:
        message = f"no non-empty reasoning records found in {rollout}"
        if args.require_reasoning:
            raise ValueError(message)
        print(json.dumps({"events": str(events_path), "status": message}))
        return 0

    align_reasoning(reasoning_records, existing_actions(events))
    before_sha256 = digest_file(events_path)
    enriched = enrich_events(events, reasoning_records, thread_id, rollout.name)
    write_jsonl_atomic(events_path, enriched)
    update_metadata(
        events_path,
        enriched,
        thread_id,
        rollout,
        reasoning_records,
        before_sha256,
    )
    summary = {
        "events": str(events_path),
        "status": "enriched",
        "thread_id": thread_id,
        "rollout": str(rollout),
        "reasoning_items": len(reasoning_records),
        "reasoning_characters": sum(len(record.text) for record in reasoning_records),
        "alignment_counts": dict(
            sorted(Counter(record.alignment for record in reasoning_records).items())
        ),
        "events_sha256": digest_file(events_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
