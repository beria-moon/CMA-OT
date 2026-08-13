# CMA-OT Dance-to-Music


## Installation

```bash
cd CMA-OT
python -m pip install -r requirements.txt
```

The requirement files pin direct package versions and the CUDA 12.4 PyTorch wheel index. Use the separate MERT environment specified in `docs/external_versions.md`.

## Expert feature extraction

CMA-OT training requires Jukebox bottom/middle/top VQ-VAE codebook embeddings. The parameterized scripts are in `tools/`; use the matching command templates:

```bash
python -m pip install -r requirements-features-jukebox.txt
bash scripts/extract_jukebox_features.sh
```

MERT is provided for expert-feature baselines and ablations, not for the default CMA-OT training loop:

```bash
python -m pip install -r requirements-features-mert.txt
bash scripts/extract_mert_features.sh
```

See `docs/feature_extraction.md` for audio preprocessing, output shapes, and JSONL manifest creation. See `docs/external_versions.md` for immutable model revisions, exact dependency environments, SHA-256 verification, and the mandatory Jukebox checkpoint pinning procedure.

## Training

Each JSONL line requires these fields:

```json
{"id":"sample_0001","pose":"/data/pose.npy","latent":"/data/latent.npy","style_i3d":"/data/i3d.npy","jukebox_bottom":"/data/jukebox_bottom.npy","jukebox_middle":"/data/jukebox_middle.npy","jukebox_top":"/data/jukebox_top.npy"}
```

- `pose`: float array `[T, 17, 3]`.
- `latent`: 64-channel audio VAE latent, either `[64, T_latent]` or `[T_latent, 64]`.
- `style_i3d`: optional `[T_video, 1024]`; omit it to train with the null style condition.
- teacher features: each 64-channel Jukebox feature, either `[64, T]` or `[T, 64]`.


```bash
bash scripts/train_aistpp.sh
```

## inference

```bash
bash scripts/infer_dance_only.sh
```

## Assets and licenses

Do **not** commit or redistribute AIST++, TikTok data, raw media, derived latents, I3D features, Jukebox/MERT features, Jukebox/MERT weights, or VAE weights unless their licenses explicitly allow it. Download them from their official sources and follow their terms. If releasing a trained checkpoint, publish it separately with a model card, dataset provenance, checksum, and usage limits.

The source code in this directory is Apache-2.0; third-party packages and external weights retain their own licenses. Before publishing, pin both the Jukebox checkpoint and the TorchScript inference decoder as required by `docs/external_versions.md`.

