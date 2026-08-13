from __future__ import annotations

from random import random
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint


class CFM(nn.Module):
    """Conditional flow matching with Euler ODE sampling."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        sigma: float = 0.0,
        odeint_kwargs: dict | None = None,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.0,
        style_drop_prob: float = 0.0,
        lrc_drop_prob: float = 0.0,
        num_channels: int,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        max_frames: int = 108,
    ):
        super().__init__()
        self.transformer = transformer
        self.sigma = sigma
        self.num_channels = num_channels
        self.max_frames = max_frames
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.style_drop_prob = style_drop_prob
        self.lrc_drop_prob = lrc_drop_prob
        self.frac_lengths_mask = frac_lengths_mask
        self.odeint_kwargs = odeint_kwargs or {"method": "euler"}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _velocity(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        text: torch.Tensor,
        duration: int,
        *,
        style_prompt: torch.Tensor,
        steps: int = 32,
        cfg_strength: float = 4.0,
        seed: int | None = None,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        self.eval()
        if cond.shape[1] != duration:
            cond = F.interpolate(cond.transpose(1, 2), size=duration, mode="linear", align_corners=False).transpose(1, 2)
        if seed is not None:
            generator = torch.Generator(device=cond.device).manual_seed(seed)
            noise = torch.randn(cond.shape, device=cond.device, dtype=cond.dtype, generator=generator)
        else:
            noise = torch.randn_like(cond)
        zero_cond = torch.zeros_like(cond)
        t_grid = torch.linspace(0, 1, steps, device=cond.device, dtype=cond.dtype)

        def fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            pred = self._velocity(self.transformer(
                x=x, cond=zero_cond, text=text, time=t, style_prompt=style_prompt,
                drop_audio_cond=False, drop_text=False, drop_prompt=False, jukebox_features=None,
            ))
            if cfg_strength == 0:
                return pred
            null_pred = self._velocity(self.transformer(
                x=x, cond=zero_cond, text=text, time=t, style_prompt=style_prompt,
                drop_audio_cond=True, drop_text=True, drop_prompt=True, jukebox_features=None,
            ))
            return pred + cfg_strength * (pred - null_pred)

        trajectory = odeint(fn, noise, t_grid, **self.odeint_kwargs)
        output = trajectory[-1]
        if vocoder is not None:
            output = vocoder(output.transpose(1, 2))
        return (output,), trajectory

    def forward(
        self,
        inp: torch.Tensor,
        text: torch.Tensor,
        *,
        style_prompt: torch.Tensor,
        lens: torch.Tensor | None = None,
        jukebox_features: dict[str, torch.Tensor] | None = None,
        curriculum_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len = inp.shape[:2]
        if lens is None:
            lens = torch.full((batch,), seq_len, device=inp.device, dtype=torch.long)
        mask = torch.arange(seq_len, device=inp.device)[None, :] < lens[:, None]
        noise = torch.randn_like(inp)
        time = torch.sigmoid(torch.randn(batch, device=inp.device, dtype=inp.dtype))
        t = time[:, None, None]
        interpolant = (1 - t) * noise + t * inp
        target_velocity = inp - noise
        condition = torch.where(mask[..., None], torch.zeros_like(inp), inp)
        predicted, alignment_loss = self.transformer(
            x=interpolant,
            cond=condition,
            text=text,
            time=time,
            style_prompt=style_prompt,
            drop_audio_cond=random() < self.audio_drop_prob,
            drop_text=random() < self.lrc_drop_prob,
            drop_prompt=random() < self.style_drop_prob,
            jukebox_features=jukebox_features,
            curriculum_weights=curriculum_weights,
        )
        flow_loss = F.mse_loss(predicted[mask], target_velocity[mask])
        return flow_loss + alignment_loss, flow_loss.detach(), alignment_loss.detach()
