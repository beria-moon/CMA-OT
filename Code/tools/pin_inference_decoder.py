#!/usr/bin/env python3
"""Pin the exact TorchScript decoder consumed by ``cma_ot.infer``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asset_lock import DEFAULT_LOCK, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the inference decoder SHA-256 into the asset lock.")
    parser.add_argument("--decoder-file", required=True, type=Path)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    if not args.decoder_file.is_file():
        raise FileNotFoundError(args.decoder_file)
    with args.lock_file.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    lock["inference_decoder"].update({"status": "pinned", "file_sha256": sha256_file(args.decoder_file)})
    with args.lock_file.open("w", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Pinned inference decoder in {args.lock_file}")


if __name__ == "__main__":
    main()
