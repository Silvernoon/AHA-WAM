from __future__ import annotations

from typing import Any, Optional

import torch

from .base_student import BaseStudent


class AHAWAMStudent(BaseStudent):
    """AHAWAM student for prior-only ODE distillation."""

    @torch.no_grad()
    def prepare_action_conditioning(
        self,
        *,
        prompt=None,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: int = 0,
        start_latents: Optional[torch.Tensor] = None,
        proprio=None,
        context=None,
        context_mask=None,
        tiled: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        sample = kwargs.get("sample")
        if sample is None:
            raise ValueError("`sample` is required for AHAWAMStudent.")
        return self.model._prepare_distill_video_state(sample, tiled=tiled)

    def predict_action_chunk(
        self,
        *,
        noisy_action_latents: torch.Tensor,
        timestep_action: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        flow_pred = self.model._predict_action_flow_with_video_state(
            noisy_action=noisy_action_latents,
            timestep_action=timestep_action,
            video_state=conditioning,
        )
        pred_action = self.flow_pred_to_action(
            noisy_action_latents=noisy_action_latents,
            flow_pred=flow_pred,
            timestep_action=timestep_action,
        )
        return {"flow_pred": flow_pred, "pred_action": pred_action}

    @torch.no_grad()
    def infer_action(
        self,
        prompt=None,
        input_image: Optional[torch.Tensor] = None,
        action_horizon: int = 0,
        proprio=None,
        context=None,
        context_mask=None,
        negative_prompt=None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift=None,
        seed=None,
        rand_device: str = "cpu",
        tiled: bool = False,
        start_latents: Optional[torch.Tensor] = None,
        sample: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> dict[str, Any]:
        self.eval()
        if num_inference_steps is None:
            num_inference_steps = int(self.default_action_inference_steps)

        phase = kwargs.get("phase")
        if phase is not None:
            return self.model.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                **kwargs,
            )

        if sample is None:
            raise ValueError(
                "`sample` or `phase` is required for AHAWAMStudent.infer_action."
            )

        video_state = self.model._prepare_distill_video_state(sample, tiled=tiled)
        action = video_state["action"]
        batch_size = video_state["batch_size"]

        if start_latents is not None:
            current_latents = start_latents.to(device=self.device, dtype=self.torch_dtype)
        else:
            current_latents = torch.randn_like(action)
        step_indices = self._resolve_ode_forward_step_indices(int(num_inference_steps))
        sigma_steps, timestep_steps = self._build_ode_sigma_grid()

        for current_step, next_step in zip(step_indices[:-1], step_indices[1:]):
            timestep_action = timestep_steps[current_step].expand(batch_size).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            flow_pred = self.model._predict_action_flow_with_video_state(
                noisy_action=current_latents,
                timestep_action=timestep_action,
                video_state=video_state,
            )
            pred_action = self.flow_pred_to_action(
                noisy_action_latents=current_latents,
                flow_pred=flow_pred,
                timestep_action=timestep_action,
            )
            if next_step == step_indices[-1]:
                current_latents = pred_action
            else:
                sigma_current = sigma_steps[current_step]
                sigma_next = sigma_steps[next_step]
                delta = sigma_next - sigma_current
                current_latents = self.infer_action_scheduler.step(
                    flow_pred, delta, current_latents
                )

        return {
            "action": current_latents[0].detach().cpu().float(),
            "final_latents": current_latents.detach().clone(),
            "step_indices": step_indices,
        }
