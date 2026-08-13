#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python "${ROOT}/tools/extract_jukebox_features.py" \
  --audio-dir "/path/to/5s_wav" \
  --output-dir "/path/to/features" \
  --lock-file "${ROOT}/configs/external_assets.lock.json" \
  --device cuda
