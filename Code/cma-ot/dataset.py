from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


REQUIRED_FIELDS = ("pose", "latent", "jukebox_bottom", "jukebox_middle", "jukebox_top")
OPTIONAL_FIELDS = ("style_i3d", "id")


def _load_array(path: str | Path) -> torch.Tensor:
    path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=True).float()
    raise ValueError(f"Unsupported feature file: {path}")


def _time_major(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D feature, got {tuple(x.shape)}")
    return x.transpose(0, 1).contiguous() if x.shape[0] == channels else x.contiguous()


class CMAOTDataset(Dataset):
    """JSONL manifest dataset; teacher features are loaded only during training."""

    def __init__(self, manifest: str | Path, require_teacher: bool = True):
        self.manifest = Path(manifest)
        self.require_teacher = require_teacher
        with self.manifest.open(encoding="utf-8") as handle:
            self.records: list[dict[str, Any]] = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest is empty: {self.manifest}")
        if require_teacher:
            missing = [key for key in REQUIRED_FIELDS if key not in self.records[0]]
            if missing:
                raise ValueError(f"Manifest lacks required fields: {missing}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | None]:
        record = self.records[index]
        item: dict[str, torch.Tensor | str | None] = {
            "id": str(record.get("id", index)),
            "pose": _load_array(record["pose"]),
            "latent": _time_major(_load_array(record["latent"]), 64),
            "style_i3d": _load_array(record["style_i3d"]) if record.get("style_i3d") else None,
        }
        if self.require_teacher:
            for level in ("bottom", "middle", "top"):
                item[f"jukebox_{level}"] = _time_major(_load_array(record[f"jukebox_{level}"]), 64)
        return item


def collate_cma_ot(batch: list[dict[str, torch.Tensor | str | None]]) -> dict[str, torch.Tensor | list[str] | None]:
    result: dict[str, torch.Tensor | list[str] | None] = {"id": [str(x["id"]) for x in batch]}
    for key in ("pose", "latent", "jukebox_bottom", "jukebox_middle", "jukebox_top"):
        values = [x[key] for x in batch if key in x]
        if values:
            result[key] = pad_sequence(values, batch_first=True)
    style_values = [x["style_i3d"] for x in batch]
    result["style_i3d"] = None if any(x is None for x in style_values) else pad_sequence(style_values, batch_first=True)
    return result
