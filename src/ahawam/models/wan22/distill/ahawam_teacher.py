from __future__ import annotations

from typing import Any, Optional

import torch

from .base_teacher import BaseTeacher


class AHAWAMTeacher(BaseTeacher):
    """AHAWAM teacher for prior-only ODE distillation."""

    @torch.no_grad()
    def rollout_action_latent_states(
        self,
        *,
        sample: dict[str, Any],
        num_inference_steps: int = 16,
        sigma_shift: Optional[float] = None,
        capture_step_indices: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16),
        tiled: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        self.eval()
        capture_steps = tuple(sorted({int(step) for step in capture_step_indices}))
        if not capture_steps or capture_steps[0] != 0:
            raise ValueError("`capture_step_indices[0]` must be 0.")
        if capture_steps[-1] != int(num_inference_steps):
            raise ValueError("`capture_step_indices[-1]` must equal `num_inference_steps`.")

        return self.model.rollout_action_prior_only(
            sample=sample,
            num_steps=num_inference_steps,
            capture_indices=capture_steps,
            sigma_shift=sigma_shift,
            tiled=tiled,
        )
