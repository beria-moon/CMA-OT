from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .afm import CFM
from .dance_encoder import DanceRhythmEncoder


class AlignCFM(CFM):
    """Dance-conditioned flow matching model used by CMA-OT.

    Jukebox teacher features are accepted only by ``forward`` during training;
    ``sample`` never consumes teacher features or reference audio.
    """

    def __init__(
        self,
        transformer: nn.Module,
        dance_feature_extractor: DanceRhythmEncoder,
        *,
        num_channels: int,
        max_frames: int = 108,
        style_feature_dim: int = 512,
        i3d_feature_dim: int = 1024,
        **kwargs,
    ):
        super().__init__(transformer=transformer, num_channels=num_channels, max_frames=max_frames, **kwargs)
        self.dance_feature_extractor = dance_feature_extractor
        self.style_feature_dim = style_feature_dim
        self.i3d_projection = nn.Sequential(
            nn.Linear(i3d_feature_dim, style_feature_dim),
            nn.ReLU(),
            nn.Linear(style_feature_dim, style_feature_dim),
            nn.Tanh(),
        )

    def extract_dance_features(self, dance_pose: torch.Tensor, target_frames: int) -> torch.Tensor:
        return self.dance_feature_extractor(dance_pose, target_frames)

    def prepare_i3d_style_prompt(
        self, style_i3d: torch.Tensor | None, *, batch_size: int, target_frames: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if style_i3d is None:
            return torch.zeros(batch_size, target_frames, self.style_feature_dim, device=device, dtype=dtype)
        if style_i3d.ndim != 3:
            raise ValueError(f"Expected style I3D [B, T, C], got {tuple(style_i3d.shape)}")
        style_i3d = style_i3d.to(device=device, dtype=dtype)
        style = self.i3d_projection(style_i3d)
        if style.shape[1] != target_frames:
            style = torch.nn.functional.interpolate(style.transpose(1, 2), size=target_frames, mode="linear", align_corners=False).transpose(1, 2)
        return style

    def forward(
        self,
        inp: torch.Tensor,
        dance_pose: torch.Tensor,
        style_prompt: torch.Tensor | None = None,
        *,
        jukebox_features: dict[str, torch.Tensor] | None = None,
        curriculum_weights: dict[str, float] | None = None,
        lens: torch.Tensor | None = None,
    ):
        target_frames = inp.shape[1]
        dance_features = self.extract_dance_features(dance_pose.to(inp.dtype), target_frames)
        style = self.prepare_i3d_style_prompt(
            style_prompt, batch_size=inp.shape[0], target_frames=target_frames, device=inp.device, dtype=inp.dtype
        )
        return super().forward(
            inp=inp,
            text=dance_features,
            style_prompt=style,
            lens=lens,
            jukebox_features=jukebox_features,
            curriculum_weights=curriculum_weights,
        )

    @torch.no_grad()
    def sample(
        self,
        dance_pose: torch.Tensor,
        *,
        style_prompt: torch.Tensor | None = None,
        steps: int = 32,
        cfg_strength: float = 4.0,
        seed: int | None = None,
        duration: int | None = None,
    ):
        duration = duration or self.max_frames
        device = dance_pose.device
        dtype = next(self.parameters()).dtype
        dance_features = self.extract_dance_features(dance_pose.to(dtype), duration)
        style = self.prepare_i3d_style_prompt(
            style_prompt, batch_size=dance_pose.shape[0], target_frames=duration, device=device, dtype=dtype
        )
        condition = torch.zeros(dance_pose.shape[0], duration, self.num_channels, device=device, dtype=dtype)
        return super().sample(
            cond=condition,
            text=dance_features,
            duration=duration,
            style_prompt=style,
            steps=steps,
            cfg_strength=cfg_strength,
            seed=seed,
        )
