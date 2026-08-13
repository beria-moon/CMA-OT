#!/usr/bin/env python3
"""Extract locked Jukebox VQ-VAE codebook embeddings from WAV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asset_lock import DEFAULT_LOCK, load_lock


LEVELS = ("bottom", "middle", "top")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Jukebox VQ-VAE features from WAV files.")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id-or-path", default=None, help="Must equal the pinned Jukebox artifact in --lock-file.")
    parser.add_argument("--revision", default=None, help="Must equal the pinned immutable Jukebox revision in --lock-file.")
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--device", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def wav_files(root: Path, recursive: bool) -> list[Path]:
    paths = root.rglob("*.wav") if recursive else root.glob("*.wav")
    files = sorted(path for path in paths if path.is_file())
    if not files:
        raise FileNotFoundError(f"No WAV files found in: {root}")
    return files


def load_mono_44k1(path: Path):
    import torch
    import torchaudio
    import torchaudio.functional as audio_functional

    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.mean(axis=1)).float()
    except Exception:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float().mean(dim=0)
    if sample_rate != 44100:
        waveform = audio_functional.resample(waveform, sample_rate, 44100)
    return waveform


def codebook_embedding(codebook: object, tokens):
    import torch

    token_ids = tokens.squeeze(0).long()
    if hasattr(codebook, "embed"):
        return codebook.embed[token_ids]
    if torch.is_tensor(codebook):
        return codebook[token_ids]
    raise TypeError("Unsupported Jukebox codebook representation; expected tensor or object with .embed.")


def main() -> None:
    args = parse_args()
    try:
        import numpy as np
        import torch
        import transformers
        from tqdm import tqdm
        from transformers import JukeboxModel
    except ImportError as exc:
        raise ImportError("Install requirements-features-jukebox.txt before extracting Jukebox features.") from exc

    spec = load_lock(args.lock_file)["jukebox"]
    if spec["status"] != "pinned" or not spec["checkpoint_tree_sha256"]:
        raise RuntimeError("Jukebox is not pinned. Run tools/pin_jukebox_checkpoint.py on the exact checkpoint and commit the updated lock file.")
    model_id = args.model_id_or_path or spec["model_id_or_path"]
    revision = args.revision or spec["revision"]
    if model_id != spec["model_id_or_path"] or revision != spec["revision"]:
        raise ValueError("Jukebox model ID and revision must match configs/external_assets.lock.json.")
    if transformers.__version__ != spec["transformers"]:
        raise RuntimeError(f"Expected transformers=={spec['transformers']}, found {transformers.__version__}.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = JukeboxModel.from_pretrained(model_id, revision=revision).to(device).eval()
    if not hasattr(model, "vqvae"):
        raise AttributeError("The pinned checkpoint does not expose model.vqvae.")
    output_root = args.output_dir / "jukebox"
    level_dirs = {level: output_root / level for level in LEVELS}
    for directory in level_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"

    with metadata_path.open("a", encoding="utf-8") as metadata:
        for wav_path in tqdm(wav_files(args.audio_dir, args.recursive), desc="Extracting Jukebox"):
            destinations = {level: level_dirs[level] / f"{wav_path.stem}.npy" for level in LEVELS}
            if all(path.exists() for path in destinations.values()) and not args.overwrite:
                continue
            waveform = load_mono_44k1(wav_path).to(device)[None, :, None]
            with torch.inference_mode():
                token_levels = model.vqvae.encode(waveform)
                if len(token_levels) != 3:
                    raise ValueError(f"Expected three VQ-VAE levels, got {len(token_levels)}.")
                features = []
                for index, tokens in enumerate(token_levels):
                    codebook = getattr(model.vqvae.bottleneck.level_blocks[index], "codebook", None)
                    if codebook is None:
                        raise AttributeError(f"No codebook found for VQ-VAE level {index}.")
                    features.append(codebook_embedding(codebook, tokens).squeeze(0).transpose(0, 1).float().cpu().numpy())
            for level, feature in zip(LEVELS, features):
                np.save(destinations[level], feature)
            metadata.write(json.dumps({
                "id": wav_path.stem, "source": str(wav_path), "model": model_id, "revision": revision,
                "checkpoint_tree_sha256": spec["checkpoint_tree_sha256"], "sample_rate": spec["sample_rate"],
                "asset_lock": str(args.lock_file),
                "features": {level: {"path": str(destinations[level]), "shape": list(feature.shape), "dtype": str(feature.dtype)} for level, feature in zip(LEVELS, features)},
            }) + "\n")


if __name__ == "__main__":
    main()
