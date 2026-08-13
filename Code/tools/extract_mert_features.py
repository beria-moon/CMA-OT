#!/usr/bin/env python3
"""Extract locked MERT temporal features from WAV files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from asset_lock import DEFAULT_LOCK, load_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract temporal MERT features from WAV files.")
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id-or-path", default=None, help="Must equal the MERT repository fixed in --lock-file.")
    parser.add_argument("--revision", default=None, help="Must equal the immutable MERT revision fixed in --lock-file.")
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--layer", type=int, default=None, help="Must equal the hidden-state index fixed in --lock-file.")
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


def load_mono(path: Path, sample_rate: int):
    import torch
    import torchaudio
    import torchaudio.functional as audio_functional

    try:
        import soundfile as sf

        waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.mean(axis=1)).float()
    except Exception:
        waveform, source_rate = torchaudio.load(path)
        waveform = waveform.float().mean(dim=0)
    if source_rate != sample_rate:
        waveform = audio_functional.resample(waveform, source_rate, sample_rate)
    return waveform


def main() -> None:
    args = parse_args()
    try:
        import torch
        import transformers
        from tqdm import tqdm
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as exc:
        raise ImportError("Install requirements-features-mert.txt before extracting MERT features.") from exc

    spec = load_lock(args.lock_file)["mert"]
    model_id = args.model_id_or_path or spec["repo_id"]
    revision = args.revision or spec["revision"]
    layer = spec["hidden_state"] if args.layer is None else args.layer
    if model_id != spec["repo_id"] or revision != spec["revision"] or layer != spec["hidden_state"]:
        raise ValueError("MERT model ID, revision, and hidden-state index must match configs/external_assets.lock.json.")
    if transformers.__version__ != spec["transformers"]:
        raise RuntimeError(f"Expected transformers=={spec['transformers']}, found {transformers.__version__}.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True).to(device).eval()
    if processor.sampling_rate != spec["sample_rate"] or model.config.hidden_size != spec["hidden_size"]:
        raise RuntimeError("Loaded MERT checkpoint does not match the locked sampling rate or hidden size.")

    destination = args.output_dir / "mert"
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "metadata.jsonl"
    with metadata_path.open("a", encoding="utf-8") as metadata:
        for wav_path in tqdm(wav_files(args.audio_dir, args.recursive), desc="Extracting MERT"):
            output_path = destination / f"{wav_path.stem}.pt"
            if output_path.exists() and not args.overwrite:
                continue
            waveform = load_mono(wav_path, processor.sampling_rate)
            inputs = processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states
            if not 0 <= layer < len(hidden_states):
                raise ValueError(f"--layer={layer} is unavailable; model returned {len(hidden_states)} hidden states.")
            feature = hidden_states[layer].squeeze(0).float().cpu()
            torch.save(feature, output_path)
            metadata.write(json.dumps({
                "id": wav_path.stem, "source": str(wav_path), "feature": str(output_path),
                "model": model_id, "revision": revision, "hidden_state": layer,
                "sample_rate": processor.sampling_rate, "shape": list(feature.shape), "dtype": str(feature.dtype),
                "asset_lock": str(args.lock_file),
            }) + "\n")


if __name__ == "__main__":
    main()
