# Expert feature extraction

CMA-OT trains with three continuous Jukebox VQ-VAE codebook embeddings. MERT features are included for the repository's expert-feature baselines and ablations; the default CMA-OT trainer does not consume MERT files.

## Common rules

- Run extraction on the same paired 5-second music clips used to create VAE latents.
- Use stable sample IDs: every output uses the input WAV stem as its file name.
- The scripts write `metadata.jsonl` alongside features. Retain it with experiment artifacts.
- Do not add external weights or extracted features to this repository without confirming their licenses.

## Jukebox: required for CMA-OT training

Install the dedicated environment dependencies:

```bash
python -m pip install -r requirements-features-jukebox.txt
```

Before extraction, pin the exact compatible Hugging Face `JukeboxModel` checkpoint that exposes `model.vqvae` using `tools/pin_jukebox_checkpoint.py`; the extractor intentionally rejects an unpinned checkpoint:

```bash
bash scripts/extract_jukebox_features.sh
```

Output layout:

```text
<output-dir>/jukebox/
├── bottom/<id>.npy
├── middle/<id>.npy
├── top/<id>.npy
└── metadata.jsonl
```

Each `.npy` is `float32 [T, 64]`, created from the corresponding VQ-VAE codebook. Input is mixed to mono and resampled to 44.1 kHz. For the original 5-second protocol, expected lengths are approximately bottom `27562`, middle `6890`, top `1722`; validate these against the exact external checkpoint before training.

Create the trainer manifest after extracting all three levels:

```bash
python tools/build_train_manifest.py \
  --pose-dir /path/to/poses \
  --latent-dir /path/to/vae_latents \
  --jukebox-dir /path/to/features/jukebox \
  --style-i3d-dir /path/to/i3d_features \
  --output /path/to/aistpp_train.jsonl \
  --strict
```

## MERT: optional baseline / ablation feature

MERT has an independent dependency pin because the locked `MERT-v1-330M` configuration declares `transformers==4.27.1`:

```bash
python -m pip install -r requirements-features-mert.txt
bash scripts/extract_mert_features.sh
```

The default uses `m-a-p/MERT-v1-330M`, converts audio to mono at the processor's 24 kHz rate, and saves hidden state index 12 as a CPU `float32 [T, 1024]` tensor:

```text
<output-dir>/mert/<id>.pt
```

MERT is not an inference dependency. The dance-only inference command uses no Jukebox or MERT feature.

## License reminder

The scripts are adapters, not model distributions. In particular, the upstream Jukebox implementation/model and the MERT checkpoint may carry non-commercial or other restrictions. Obtain weights from official publishers and review their current license before training, releasing a checkpoint, or distributing derived features.
