"""Evaluation fingerprints that make resumable prediction runs safe.

Used by collect_predictions.py: before consuming any prior .progress/
resume state, build a fingerprint covering every input path (weights,
checkpoint, config, genome, bigwigs, ...), the CLI settings, and the
relevant source files, and check it against what a previous partial run
recorded. A mismatch means the output directory's resume state was left
by a different run and must not be reused.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


_FULL_HASH_LIMIT = 64 * 1024 * 1024
_SAMPLE_BYTES = 4 * 1024 * 1024


def _hash_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    if stat.st_size <= _FULL_HASH_LIMIT:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hash_kind = "full_sha256"
    else:
        with path.open("rb") as handle:
            digest.update(handle.read(_SAMPLE_BYTES))
            handle.seek(max(0, stat.st_size - _SAMPLE_BYTES))
            digest.update(handle.read(_SAMPLE_BYTES))
        digest.update(str(stat.st_size).encode())
        hash_kind = "first_last_4MiB_plus_size_sha256"
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        hash_kind: digest.hexdigest(),
    }


def path_identity(raw_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a stable, content-sensitive identity for a file or directory."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return {"kind": "file", **_hash_file(path)}
    if not path.is_dir():
        raise ValueError("Unsupported evaluation input path: {}".format(path))

    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        identity = _hash_file(child)
        identity["relative_path"] = str(child.relative_to(path))
        identity.pop("path")
        entries.append(identity)
    tree_json = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "kind": "directory",
        "path": str(path.resolve()),
        "files": entries,
        "tree_sha256": hashlib.sha256(tree_json.encode()).hexdigest(),
    }


def build_fingerprint(
    *,
    framework: str,
    settings: Mapping[str, Any],
    paths: Mapping[str, str | os.PathLike[str] | list[str]],
    code_paths: list[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Build a canonical fingerprint covering inference inputs and code."""
    path_manifest: dict[str, Any] = {}
    for name, value in paths.items():
        if isinstance(value, list):
            path_manifest[name] = [path_identity(item) for item in value]
        else:
            path_manifest[name] = path_identity(value)
    payload = {
        "schema_version": 1,
        "framework": framework,
        "settings": dict(settings),
        "paths": path_manifest,
        "code": [path_identity(path) for path in code_paths],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "fingerprint_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "payload": payload,
    }


def ensure_provenance(output_dir: str, fingerprint: Mapping[str, Any]) -> str:
    """Write or validate provenance before any resumable state is consumed."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    progress_state = output / ".progress" / "state.pkl"
    provenance_path = output / "provenance.json"
    expected = str(fingerprint["fingerprint_sha256"])

    if progress_state.exists() and not provenance_path.exists():
        raise RuntimeError(
            "Unsafe legacy resume state at {}: no provenance.json. "
            "Use a new output directory.".format(progress_state)
        )
    if provenance_path.exists():
        with provenance_path.open() as handle:
            existing = json.load(handle)
        observed = existing.get("fingerprint_sha256")
        if observed != expected:
            raise RuntimeError(
                "Evaluation output directory belongs to a different run: "
                "stored fingerprint={}, requested={}. Use a new directory.".format(
                    observed, expected
                )
            )
    else:
        tmp = provenance_path.with_suffix(".json.tmp")
        with tmp.open("w") as handle:
            json.dump(fingerprint, handle, indent=2, sort_keys=True)
        os.replace(tmp, provenance_path)
    return expected
