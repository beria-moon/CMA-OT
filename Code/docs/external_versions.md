# External version lock

`configs/external_assets.lock.json` is the release authority for external assets. Do not update a model, revision, runtime, or checksum without regenerating all dependent features and reporting the change.

## Fixed runtime

Use Python `3.11.6` and the CUDA 12.4 PyTorch wheels declared in the exact requirement files:

- core training/inference: `requirements.txt`
- MERT extraction: `requirements-features-mert.txt`
- Jukebox extraction: `requirements-features-jukebox.txt`

Do not install these three environments together: MERT is fixed to `transformers==4.27.1`, while the core/Jukebox adapter uses `transformers==4.49.0`.

## Fixed assets

| Asset | Immutable reference |
| --- | --- |
| MERT | `m-a-p/MERT-v1-330M@5240c2708a5acaee1007f43fb9735c7dcd0b78c9` |
| MERT custom source | `yizhilll/MERT@391062e4c384aaad4e5a992be339ef70769dbd6f` |
| DiffRhythm VAE | `ASLP-lab/DiffRhythm-vae@74e2afacfd91dd1b96662c96dcef763c1258768b` |
| Jukebox source | `openai/jukebox@08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3` |

MERT and DiffRhythm VAE source files are locked by SHA-256 in `external_assets.lock.json`. Verify downloaded assets before feature extraction:

```bash
python tools/verify_external_assets.py \
  --mert-dir /path/to/MERT-v1-330M \
  --vae-dir /path/to/DiffRhythm-vae
```

The published DiffRhythm VAE uses `safetensors`, while `cma_ot.infer` requires a TorchScript decoder exposing `decode_export`. The decoder is a separate release asset and is deliberately blocked until it is pinned:

```bash
python tools/pin_inference_decoder.py --decoder-file /path/to/authorized_decoder.pt
python tools/verify_external_assets.py --decoder-file /path/to/authorized_decoder.pt --require-inference-decoder
```

## Required Jukebox release step

The exact Jukebox VQ-VAE checkpoint used for the original experiment is not present in this repository. The checked-in lock deliberately blocks Jukebox extraction until the release owner pins that checkpoint. This prevents silently generating non-comparable teacher features.

After obtaining the exact compatible checkpoint, run:

```bash
python tools/pin_jukebox_checkpoint.py \
  --checkpoint-dir /path/to/exact_jukebox_checkpoint \
  --model-id-or-path <immutable-repository-or-storage-uri> \
  --revision <immutable-commit-tag-or-generation-id>
python tools/verify_external_assets.py \
  --jukebox-dir /path/to/exact_jukebox_checkpoint \
  --require-jukebox
```

Commit the updated `configs/external_assets.lock.json`; then use the same repository ID and revision in every feature-extraction run. The generated `metadata.jsonl` records both fields and the checkpoint tree digest.
