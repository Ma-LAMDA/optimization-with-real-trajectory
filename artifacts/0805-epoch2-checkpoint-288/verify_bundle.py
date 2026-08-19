#!/usr/bin/env python3
"""Verify the portable 0805 epoch-2 LoRA checkpoint and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_PAYLOAD = {
    "adapter_model.safetensors": "f39313e8f28160a58fcdd8a35916382f8959e12c335ccb5c26252885647ad06b",
    "adapter_config.json": "660828605d3031bfb63e166f079d1e7e2a01254d207ed201338a562e88bef4a5",
    "additional_config.json": "c7799462ebedae6557ffad31566029e2d2f958b7b40e46e972cf901bcaf45733",
    "args.json": "3ce5c1f6aa58233fdb46e84fa7779858258b3cbff79f04223962ff665fd63cd9",
    "trainer_state.json": "ddb278031f80e4c9fc1ca2dd0345d29543ad80b3db89087fa7427d4f44023fca",
}
EXPECTED_SOURCE_COMMIT = "34fa0dbff027e7ab1241f229042441e7412e1223"
EXPECTED_EVAL_LOSS = 0.15015004575252533
EXPECTED_EVAL_TOKEN_ACC = 0.9293184783699688


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_normalized(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_sum_file(path: Path) -> list[tuple[str, str]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, source = line.split(maxsplit=1)
        records.append((digest.lower(), source.strip().lstrip("*")))
    return records


def verify_recorded_files(records: list[tuple[str, str]], root: Path, repo_mode: bool) -> int:
    checked = 0
    for expected, source in records:
        if repo_mode:
            marker = "optimization-with-real-trajectory/"
            relative = source.split(marker, 1)[1] if marker in source else source
            target = root / relative
        else:
            target = root / Path(source).name
        if not target.is_file():
            raise SystemExit(f"missing recorded file: {target}")
        actual = sha256_lf_normalized(target) if repo_mode else sha256(target)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {target}: {actual} != {expected}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()

    bundle = Path(__file__).resolve().parent
    for name, expected in EXPECTED_PAYLOAD.items():
        path = bundle / name
        if not path.is_file():
            raise SystemExit(f"missing checkpoint payload: {path}")
        if name.endswith(".safetensors"):
            with path.open("rb") as handle:
                prefix = handle.read(64)
            if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise SystemExit("adapter_model.safetensors is an unresolved Git LFS pointer; run git lfs pull")
        # Git may materialize tracked text files with CRLF on Windows.  Keep
        # the adapter byte-exact, but compare JSON payloads in canonical LF
        # form so a valid cross-platform checkout is not rejected.
        actual = sha256(path) if name.endswith(".safetensors") else sha256_lf_normalized(path)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {name}: {actual} != {expected}")

    source_commit = (bundle / "source_git_commit.txt").read_text(encoding="utf-8").strip()
    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise SystemExit(f"unexpected source commit: {source_commit}")

    adapter = json.loads((bundle / "adapter_config.json").read_text(encoding="utf-8"))
    expected_adapter = {"r": 8, "lora_alpha": 32, "lora_dropout": 0.05, "peft_type": "LORA"}
    for key, expected in expected_adapter.items():
        if adapter.get(key) != expected:
            raise SystemExit(f"unexpected adapter setting {key}: {adapter.get(key)!r} != {expected!r}")

    state = json.loads((bundle / "trainer_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != 288 or not math.isclose(float(state.get("epoch", -1)), 2.0):
        raise SystemExit(f"not the epoch-2 boundary: step={state.get('global_step')} epoch={state.get('epoch')}")
    eval_rows = [row for row in state.get("log_history", []) if row.get("step") == 288 and "eval_loss" in row]
    if not eval_rows:
        raise SystemExit("epoch-2 eval record is missing")
    eval_row = eval_rows[-1]
    if not math.isclose(float(eval_row["eval_loss"]), EXPECTED_EVAL_LOSS, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"unexpected eval loss: {eval_row['eval_loss']}")
    if not math.isclose(float(eval_row["eval_token_acc"]), EXPECTED_EVAL_TOKEN_ACC, rel_tol=0, abs_tol=1e-12):
        raise SystemExit(f"unexpected eval token accuracy: {eval_row['eval_token_acc']}")

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("safetensors is required for structural verification") from exc
    with safe_open(bundle / "adapter_model.safetensors", framework="numpy") as handle:
        keys = list(handle.keys())
        shapes = [tuple(handle.get_tensor(key).shape) for key in keys]
        dtypes = {str(handle.get_tensor(key).dtype).removeprefix("torch.") for key in keys}
    if len(keys) != 992 or sum(math.prod(shape) for shape in shapes) != 58363904:
        raise SystemExit("unexpected Safetensors structure")
    if dtypes != {"float32"}:
        raise SystemExit(f"unexpected adapter dtypes: {sorted(dtypes)}")

    base_checked = 0
    if args.base_model:
        base_checked = verify_recorded_files(
            read_sum_file(bundle / "source_base_model_files_sha256.txt"),
            args.base_model.resolve(),
            repo_mode=False,
        )

    repo_checked = 0
    if args.repo_root:
        repo_checked = verify_recorded_files(
            read_sum_file(bundle / "source_input_sha256.txt"),
            args.repo_root.resolve(),
            repo_mode=True,
        )

    print(json.dumps({
        "status": "ok",
        "artifact": "0805-epoch2-checkpoint-288",
        "epoch": 2.0,
        "global_step": 288,
        "eval_loss": EXPECTED_EVAL_LOSS,
        "eval_token_acc": EXPECTED_EVAL_TOKEN_ACC,
        "adapter_sha256": EXPECTED_PAYLOAD["adapter_model.safetensors"],
        "tensor_count": len(keys),
        "base_model_files_checked": base_checked,
        "repository_inputs_checked": repo_checked,
    }, indent=2))


if __name__ == "__main__":
    main()
