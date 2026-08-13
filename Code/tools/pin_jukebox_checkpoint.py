#!/usr/bin/env python3
"""Pin an exact local JukeboxModel checkpoint in the release lock file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asset_lock import DEFAULT_LOCK, sha256_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write immutable Jukebox checkpoint metadata into the asset lock.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--model-id-or-path", required=True, help="Published repository ID or immutable storage URI.")
    parser.add_argument("--revision", required=True, help="Immutable model commit, tag, or storage generation ID.")
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint_dir.is_dir():
        raise NotADirectoryError(args.checkpoint_dir)
    with args.lock_file.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    lock["jukebox"].update({
        "status": "pinned",
        "model_id_or_path": args.model_id_or_path,
        "revision": args.revision,
        "checkpoint_tree_sha256": sha256_tree(args.checkpoint_dir),
    })
    with args.lock_file.open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Pinned Jukebox checkpoint in {args.lock_file}")


if __name__ == "__main__":
    main()
