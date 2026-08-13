from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "configs" / "external_assets.lock.json"


def load_lock(path: str | Path = DEFAULT_LOCK) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(root).as_posix().encode()
        digest.update(relative + b"\0" + sha256_file(file_path).encode() + b"\n")
    return digest.hexdigest()


def verify_files(root: str | Path, expected: dict[str, str]) -> list[str]:
    root = Path(root)
    errors: list[str] = []
    for relative, expected_hash in expected.items():
        file_path = root / relative
        if not file_path.is_file():
            errors.append(f"missing: {file_path}")
        elif sha256_file(file_path) != expected_hash:
            errors.append(f"checksum mismatch: {file_path}")
    return errors
