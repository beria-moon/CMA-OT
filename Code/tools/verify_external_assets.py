#!/usr/bin/env python3
"""Verify external model files against ``configs/external_assets.lock.json``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from asset_lock import DEFAULT_LOCK, load_lock, sha256_file, sha256_tree, verify_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify pinned MERT, VAE, decoder, and Jukebox assets.")
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--mert-dir", type=Path)
    parser.add_argument("--vae-dir", type=Path, help="Directory containing vae_config.json and vae_model.safetensors.")
    parser.add_argument("--decoder-file", type=Path, help="Exact TorchScript decoder used by cma_ot.infer.")
    parser.add_argument("--jukebox-dir", type=Path, help="Exact local JukeboxModel checkpoint directory.")
    parser.add_argument("--require-inference-decoder", action="store_true")
    parser.add_argument("--require-jukebox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock = load_lock(args.lock_file)
    errors: list[str] = []
    if args.mert_dir:
        errors.extend(verify_files(args.mert_dir, lock["mert"]["files"]))
    if args.vae_dir:
        errors.extend(verify_files(args.vae_dir, lock["diffrhythm_vae"]["files"]))

    decoder = lock["inference_decoder"]
    if args.decoder_file:
        expected = decoder.get("file_sha256")
        if not expected:
            errors.append("Inference decoder is not pinned; run tools/pin_inference_decoder.py and commit the updated lock file.")
        elif not args.decoder_file.is_file() or sha256_file(args.decoder_file) != expected:
            errors.append(f"checksum mismatch: {args.decoder_file}")
    elif args.require_inference_decoder and not decoder.get("file_sha256"):
        errors.append("Inference decoder is not pinned; the release is incomplete.")

    jukebox = lock["jukebox"]
    if args.jukebox_dir:
        expected = jukebox.get("checkpoint_tree_sha256")
        if not expected:
            errors.append("Jukebox checkpoint is not pinned; run tools/pin_jukebox_checkpoint.py and commit the updated lock file.")
        elif sha256_tree(args.jukebox_dir) != expected:
            errors.append(f"checksum mismatch: {args.jukebox_dir}")
    elif args.require_jukebox and not jukebox.get("checkpoint_tree_sha256"):
        errors.append("Jukebox checkpoint is not pinned; the release is incomplete.")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("External asset verification passed.")


if __name__ == "__main__":
    main()
