# Reproducibility checklist

Before publishing results, record:

1. Dataset split identifiers and licenses, without redistributing restricted files.
2. The committed `configs/external_assets.lock.json` revision and successful `tools/verify_external_assets.py` output for VAE, MERT, and Jukebox.
3. Hardware, PyTorch/CUDA versions, seed, effective batch size, and gradient accumulation.
4. Full 200-epoch configuration and `last.pt` checksum.
5. The saved per-epoch curriculum weights and CMA-OT alpha statistics.
6. Inference settings: Euler solver, 32 steps, CFG 4.0, seed, and VAE decoder.
7. Exact evaluation protocol and generated audio identifiers for BCS/CSD/BHS/HSD/F1, PANNs-FAD, CLAP-FAD, and Audiobox aesthetics.
8. Exact dependency files used (`requirements*.txt`) and the corresponding `pip freeze` output.

The current source package intentionally leaves evaluator implementations out until their metric implementations and required pretrained assets have been audited against the paper protocol.
