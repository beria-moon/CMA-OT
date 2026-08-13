# Data format

The trainer reads UTF-8 JSONL, one sample per line. Paths may be absolute or relative to the invocation directory. Arrays are floating-point `.npy` files, except `.pt`/`.pth` tensors are also accepted.

Required fields:

- `pose`: `[T_pose, 17, 3]` root-relative or global 3D joint positions.
- `latent`: `[64, T_latent]` or `[T_latent, 64]` VAE latent for the paired music clip.
- `jukebox_bottom`, `jukebox_middle`, `jukebox_top`: `[64, T]` or `[T, 64]` teacher features from the same music clip.

Optional fields:

- `id`: string used in logs.
- `style_i3d`: `[T_video, 1024]` motion/video style feature. Omit for the null style condition.

For the paper setup, clips are 5 s at 44.1 kHz. The VAE downsampling ratio is 2048, so a 5 s segment has approximately 108 latent frames. Do not mix features generated with different audio preprocessing, VAE, or Jukebox checkpoints in one run.

Generate the required teacher features with `tools/extract_jukebox_features.py`. Its output root is `jukebox/{bottom,middle,top}` and can be passed directly to `tools/build_train_manifest.py`. MERT outputs are optional baseline features and are intentionally not part of the default CMA-OT manifest schema.
