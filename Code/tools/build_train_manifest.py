#!/usr/bin/env python3
"""Build the JSONL manifest consumed by ``cma_ot.train`` from aligned feature directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CMA-OT training JSONL manifest.")
    parser.add_argument("--pose-dir", required=True, type=Path)
    parser.add_argument("--latent-dir", required=True, type=Path)
    parser.add_argument("--jukebox-dir", required=True, type=Path, help="Directory containing bottom/, middle/, top/.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--style-i3d-dir", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping incomplete samples.")
    return parser.parse_args()


def resolve(directory: Path, sample_id: str, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        candidate = directory / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    pose_files = sorted(args.pose_dir.glob("*.npy"))
    if not pose_files:
        raise FileNotFoundError(f"No pose .npy files found in: {args.pose_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: list[str] = []
    with args.output.open("w", encoding="utf-8") as handle:
        for pose_path in pose_files:
            sample_id = pose_path.stem
            latent = resolve(args.latent_dir, sample_id, (".npy", ".pt", ".pth"))
            teachers = {level: args.jukebox_dir / level / f"{sample_id}.npy" for level in ("bottom", "middle", "top")}
            missing = [name for name, path in {"latent": latent, **teachers}.items() if path is None or not path.exists()]
            if missing:
                message = f"{sample_id}: missing {', '.join(missing)}"
                if args.strict:
                    raise FileNotFoundError(message)
                skipped.append(message)
                continue
            record = {
                "id": sample_id,
                "pose": str(pose_path),
                "latent": str(latent),
                **{f"jukebox_{level}": str(path) for level, path in teachers.items()},
            }
            if args.style_i3d_dir:
                style = resolve(args.style_i3d_dir, sample_id, (".npy", ".pt", ".pth"))
                if style:
                    record["style_i3d"] = str(style)
            handle.write(json.dumps(record) + "\n")
            written += 1
    print(f"Wrote {written} records to {args.output}.")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete samples; first: {skipped[0]}")


if __name__ == "__main__":
    main()
