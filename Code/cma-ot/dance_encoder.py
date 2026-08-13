from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DanceRhythmEncoder(nn.Module):
    """Encode root-relative 3D poses ``[B, T, J, 3]`` into 512-D motion tokens."""

    def __init__(self, num_joints: int = 17, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.num_joints = num_joints
        self.pose_proj = nn.Sequential(
            nn.Linear(num_joints * 3 * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1, groups=output_dim),
            nn.GELU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
        )

    def forward(self, pose: torch.Tensor, target_frames: int) -> torch.Tensor:
        if pose.ndim != 4 or pose.shape[-1] != 3:
            raise ValueError(f"Expected pose [B, T, J, 3], got {tuple(pose.shape)}")
        if pose.shape[2] != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joints, got {pose.shape[2]}")

        pose = pose - pose[:, :, :1]
        velocity = F.pad(pose[:, 1:] - pose[:, :-1], (0, 0, 0, 0, 0, 1))
        features = torch.cat((pose, velocity), dim=-1).flatten(start_dim=2)
        features = self.pose_proj(features)
        features = features + self.temporal(features.transpose(1, 2)).transpose(1, 2)
        return F.interpolate(features.transpose(1, 2), size=target_frames, mode="linear", align_corners=False).transpose(1, 2)
