#!/usr/bin/env python3
"""Create a per-run Codex model catalog with metadata for the served model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata-source", default="Qwen3.6-27B-trained")
    return parser.parse_args()


def load_catalog(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    models = catalog.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path} must contain a non-empty models list")
    slugs = [model.get("slug") for model in models if isinstance(model, dict)]
    if len(slugs) != len(models) or any(not isinstance(slug, str) or not slug for slug in slugs):
        raise ValueError(f"{path} contains an invalid model slug")
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"{path} contains duplicate model slugs")
    return catalog


def ensure_model(catalog: dict, model_name: str, metadata_source: str) -> bool:
    models = catalog["models"]
    by_slug = {model["slug"]: model for model in models}
    if model_name in by_slug:
        return False
    if metadata_source not in by_slug:
        raise ValueError(f"metadata source {metadata_source!r} is not present in the template")

    model = copy.deepcopy(by_slug[metadata_source])
    model["slug"] = model_name
    model["display_name"] = model_name
    model["description"] = (
        f"Local Qwen3.6-27B model {model_name}; Codex metadata cloned from "
        f"{metadata_source} for an isolated evaluation run."
    )
    models.append(model)
    return True


def write_catalog(path: Path, catalog: dict) -> str:
    payload = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary_path = Path(stream.name)
    os.replace(temporary_path, path)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.template)
    added = ensure_model(catalog, args.model, args.metadata_source)
    digest = write_catalog(args.output, catalog)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": args.model,
                "metadata_source": args.metadata_source,
                "added": added,
                "sha256": digest,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
