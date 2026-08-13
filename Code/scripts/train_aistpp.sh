#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

accelerate launch -m cma_ot.train \
  --config "${ROOT}/configs/cma_ot_aistpp.yaml" \
  --train-manifest "/path/to/aistpp/train.jsonl" \
  --output-dir "${ROOT}/runs/cma_ot_aistpp"
