#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m cma_ot.infer \
  --config "${ROOT}/configs/cma_ot_aistpp.yaml" \
  --checkpoint "/path/to/cma_ot_last.pt" \
  --vae "/path/to/pinned_authorized_vae_decoder.pt" \
  --asset-lock "${ROOT}/configs/external_assets.lock.json" \
  --pose "/path/to/dance.npy" \
  --output "${ROOT}/outputs/dance.wav" \
  --steps 32 --cfg-scale 4.0 --seed 42
