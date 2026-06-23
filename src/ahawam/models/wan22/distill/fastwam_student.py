from __future__ import annotations

from typing import Any

import torch

from .base_student import BaseStudent


class FastWAMStudent(BaseStudent):
    """FastWAM student role adapter for ODE distillation."""

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
        return self.model._prepare_action_inference(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=start_latents,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=None,
            rand_device="cpu",
            tiled=tiled,
        )

    def predict_action_chunk(
        self,
        *,
        noisy_action_latents: torch.Tensor,
        timestep_action: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        flow_pred = self.model._predict_action_noise_with_cache_trainable(
            latents_action=noisy_action_latents,
            timestep_action=timestep_action,
            context=conditioning["context"],
            context_mask=conditioning["context_mask"],
            video_kv_cache=conditioning["video_kv_cache"],
            attention_mask=conditioning["attention_mask"],
            video_seq_len=conditioning["video_seq_len"],
        )
        pred_action = self.flow_pred_to_action(
            noisy_action_latents=noisy_action_latents,
            flow_pred=flow_pred,
            timestep_action=timestep_action,
        )
        return {
            "flow_pred": flow_pred,
            "pred_action": pred_action,
        }

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
        del negative_prompt, text_cfg_scale, sigma_shift
        self.eval()
        if num_inference_steps is None:
            num_inference_steps = int(self.default_action_inference_steps)
        prepared = self.model._prepare_action_inference(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=start_latents,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )
        current_latents = prepared["start_latents"]
        step_indices = self._resolve_ode_forward_step_indices(int(num_inference_steps))
        sigma_steps, timestep_steps = self._build_ode_sigma_grid()

        for current_step, next_step in zip(step_indices[:-1], step_indices[1:]):
            timestep_action = (
                timestep_steps[current_step]
                .expand(current_latents.shape[0])
                .to(
                    device=self.device,
                    dtype=self.torch_dtype,
                )
            )
            pred = self.predict_action_chunk(
                noisy_action_latents=current_latents,
                timestep_action=timestep_action,
                conditioning=prepared,
            )
            if next_step == step_indices[-1]:
                current_latents = pred["pred_action"]
            else:
                sigma_current = sigma_steps[current_step]
                sigma_next = sigma_steps[next_step]
                delta = sigma_next - sigma_current
                current_latents = self.infer_action_scheduler.step(
                    pred["flow_pred"], delta, current_latents
                )

        return {
            "action": current_latents[0].detach().to(device="cpu", dtype=torch.float32),
            "final_latents": current_latents.detach().clone(),
            "step_indices": step_indices,
        }
