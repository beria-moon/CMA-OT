from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .config import build_model, load_config, seed_everything


DEFAULT_ASSET_LOCK = Path(__file__).resolve().parents[1] / "configs" / "external_assets.lock.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_decoder(decoder_path: str | Path, lock_path: str | Path) -> None:
    with Path(lock_path).open(encoding="utf-8") as handle:
        spec = json.load(handle)["inference_decoder"]
    expected = spec.get("file_sha256")
    if spec.get("status") != "pinned" or not expected:
        raise RuntimeError("Inference decoder is not pinned. Run tools/pin_inference_decoder.py and commit the updated asset lock.")
    if sha256_file(decoder_path) != expected:
        raise RuntimeError("The supplied TorchScript decoder does not match configs/external_assets.lock.json.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate music from a dance pose sequence.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pose", required=True, help="Numpy pose file [T, 17, 3].")
    parser.add_argument("--output", required=True, help="Output .wav path.")
    parser.add_argument("--vae", required=True, help="Pinned TorchScript VAE decoder with decode_export; not distributed here.")
    parser.add_argument("--asset-lock", type=Path, default=DEFAULT_ASSET_LOCK)
    parser.add_argument("--style-i3d", default=None, help="Optional I3D array [T, 1024].")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    verify_decoder(args.vae, args.asset_lock)
    config = load_config(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    pose = torch.from_numpy(np.load(args.pose)).float().unsqueeze(0).to(device)
    style = None if args.style_i3d is None else torch.from_numpy(np.load(args.style_i3d)).float().unsqueeze(0).to(device)
    latents, _ = model.sample(pose, style_prompt=style, steps=args.steps, cfg_strength=args.cfg_scale, seed=args.seed)
    vae = torch.jit.load(args.vae, map_location=device).eval()
    waveform = vae.decode_export(latents[0].transpose(1, 2)).squeeze(0).cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output, waveform, sample_rate=44100)


if __name__ == "__main__":
    main()
