from __future__ import annotations

from typing import Any, Optional

import torch

from .base_teacher import BaseTeacher


class FastWAMTeacher(BaseTeacher):
    """FastWAM teacher role adapter for ODE distillation."""

    @torch.no_grad()
    def rollout_action_latent_states(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 16,
        sigma_shift: Optional[float] = None,
        capture_step_indices: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16),
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        capture_steps = tuple(sorted({int(x) for x in capture_step_indices}))
        if not capture_steps or capture_steps[0] != 0:
            raise NotImplementedError(
                "`capture_step_indices[0]` must be 0. Teacher rollout currently requires noise as the start state."
            )
        if capture_steps[-1] != int(num_inference_steps):
            raise NotImplementedError(
                "`capture_step_indices[-1]` must equal `num_inference_steps`. Teacher rollout currently requires the final state."
            )
        prepared = self.model._prepare_action_inference(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        rollout = self.model._rollout_action_latents_with_cache(
            start_latents=prepared["start_latents"],
            context=prepared["context"],
            context_mask=prepared["context_mask"],
            video_kv_cache=prepared["video_kv_cache"],
            attention_mask=prepared["attention_mask"],
            video_seq_len=prepared["video_seq_len"],
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            capture_step_indices=capture_step_indices,
        )
        return {
            "initial_latents": rollout["captured_states"][0].detach().clone(),
            "captured_states": rollout["captured_states"],
            "capture_step_indices": rollout["capture_step_indices"],
            "timesteps": rollout["timesteps"],
            "deltas": rollout["deltas"],
            "final_latents": rollout["final_latents"].detach().clone(),
        }
