from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseStudent(nn.Module, ABC):
    """Role wrapper for ODE students."""

    def __init__(
        self,
        model: nn.Module,
        *,
        prediction_type: str = "flow",
        ode_denoise_step_indices: tuple[int, ...] | list[int] = (0, 1, 2, 4, 8, 12, 16),
        default_action_inference_steps: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.prediction_type = str(prediction_type)
        self.ode_denoise_step_indices = tuple(int(x) for x in ode_denoise_step_indices)
        self.default_action_inference_steps = int(default_action_inference_steps)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattr__("model")
            return getattr(model, name)

    @abstractmethod
    @torch.no_grad()
    def prepare_action_conditioning(
        self,
        *,
        prompt,
        input_image: torch.Tensor,
        action_horizon: int,
        start_latents: torch.Tensor,
        proprio=None,
        context=None,
        context_mask=None,
        tiled: bool = False,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict_action_chunk(
        self,
        *,
        noisy_action_latents: torch.Tensor,
        timestep_action: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def initialize_from_teacher(self, teacher) -> None:
        self.load_model_state_dict(teacher.export_model_state_dict(), strict=False)

    def export_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict)

    def flow_pred_to_action(
        self,
        *,
        noisy_action_latents: torch.Tensor,
        flow_pred: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        sigma = (timestep_action / float(self.infer_action_scheduler.num_train_timesteps)).to(
            device=noisy_action_latents.device,
            dtype=noisy_action_latents.dtype,
        )
        if sigma.ndim == 0:
            return noisy_action_latents - sigma * flow_pred
        if sigma.ndim == 1:
            if sigma.shape[0] != noisy_action_latents.shape[0]:
                raise ValueError(
                    f"`timestep_action` batch mismatch: {sigma.shape[0]} vs {noisy_action_latents.shape[0]}"
                )
            sigma = sigma.view(-1, *([1] * (noisy_action_latents.ndim - 1)))
            return noisy_action_latents - sigma * flow_pred

        if sigma.ndim == noisy_action_latents.ndim - 1:
            target_shape = noisy_action_latents.shape[:-1]
            if tuple(sigma.shape) == tuple(target_shape):
                sigma = sigma.unsqueeze(-1)
                return noisy_action_latents - sigma * flow_pred
            if sigma.ndim == 2 and sigma.shape[0] == target_shape[0]:
                if sigma.shape[1] == 1:
                    sigma = sigma.expand(target_shape[0], target_shape[1]).unsqueeze(-1)
                    return noisy_action_latents - sigma * flow_pred
                if target_shape[1] % sigma.shape[1] == 0:
                    chunk_size = target_shape[1] // sigma.shape[1]
                    sigma = sigma.repeat_interleave(chunk_size, dim=1).unsqueeze(-1)
                    return noisy_action_latents - sigma * flow_pred
            raise ValueError(
                "`timestep_action` shape is incompatible with action latents: "
                f"got {tuple(sigma.shape)} vs expected [B], [B,T], or [B,num_chunks] for {tuple(noisy_action_latents.shape)}"
            )

        while sigma.ndim < noisy_action_latents.ndim:
            sigma = sigma.unsqueeze(-1)
        return noisy_action_latents - sigma * flow_pred

    def _resolve_ode_forward_step_indices(self, num_denoise_steps: int) -> tuple[int, ...]:
        num_denoise_steps = int(num_denoise_steps)
        available = tuple(int(x) for x in self.ode_denoise_step_indices)
        if not available or available[0] != 0:
            raise ValueError(f"`ode_denoise_step_indices[0]` must be 0, got {available}")
        final_step = int(available[-1])
        if num_denoise_steps == 1:
            return (0, final_step)
        if num_denoise_steps == 2:
            path = (0, 8, final_step)
            if any(step not in available for step in path):
                raise ValueError(f"`ode_denoise_step_indices` must contain {path}, got {available}")
            return path
        if num_denoise_steps == 4:
            path = (0, 4, 8, 12, final_step)
            if any(step not in available for step in path):
                raise ValueError(f"`ode_denoise_step_indices` must contain {path}, got {available}")
            return path
        raise ValueError(f"Unsupported ODE denoise steps: {num_denoise_steps}. Expected one of [1, 2, 4].")

    def _build_ode_sigma_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        final_step = int(self.ode_denoise_step_indices[-1])
        timesteps, _ = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=final_step,
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=None,
        )
        sigma_steps = torch.zeros((final_step + 1,), device=self.device, dtype=self.torch_dtype)
        sigma_steps[:-1] = timesteps / float(self.infer_action_scheduler.num_train_timesteps)
        return sigma_steps, timesteps

    @abstractmethod
    @torch.no_grad()
    def infer_action(
        self,
        prompt,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int | None = None,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        start_latents: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        return self.model.infer(*args, **kwargs)

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        return self.model.infer_joint(*args, **kwargs)

    def save_checkpoint(self, path, optimizer=None, step=None):
        return self.model.save_checkpoint(path, optimizer=optimizer, step=step)

    def load_checkpoint(self, path, optimizer=None):
        return self.model.load_checkpoint(path, optimizer=optimizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
