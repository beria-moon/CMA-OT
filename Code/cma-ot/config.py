from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .Aligncfm import AlignCFM
from .Aligndit import AlignDiT
from .dance_encoder import DanceRhythmEncoder


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any]) -> AlignCFM:
    model_cfg = config["model"]
    dance_cfg = config["dance"]
    transformer = AlignDiT(
        dim=model_cfg["dim"],
        depth=model_cfg["depth"],
        heads=model_cfg["heads"],
        dim_head=model_cfg["dim_head"],
        ff_mult=model_cfg["ff_mult"],
        mel_dim=model_cfg["latent_dim"],
        text_dim=dance_cfg["output_dim"],
        style_dim=model_cfg["style_dim"],
        max_frames=model_cfg["max_frames"],
        use_multilayer_cma_ot=True,
        adaptive_alpha=True,
    )
    return AlignCFM(
        transformer=transformer,
        dance_feature_extractor=DanceRhythmEncoder(
            num_joints=dance_cfg["num_joints"], output_dim=dance_cfg["output_dim"]
        ),
        num_channels=model_cfg["latent_dim"],
        max_frames=model_cfg["max_frames"],
        style_feature_dim=model_cfg["style_dim"],
    )
