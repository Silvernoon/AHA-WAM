from __future__ import annotations

from typing import Optional

import torch

from .base_student import BaseStudent
from .base_teacher import BaseTeacher
from .fastwam_ode import FastWAMODE


class AHAWAMODE(FastWAMODE):
    """ODE distillation container for AHAWAM prior-only mode."""

    STUDENT_CAPTURE_TIMESTEP_MODES = ("per_chunk", "shared")

    def __init__(
        self,
        teacher: BaseTeacher,
        student: BaseStudent,
        teacher_task_name: str,
        student_task_name: str,
        teacher_num_inference_steps: int = 16,
        teacher_capture_step_indices: list[int] | tuple[int, ...] = (
            0,
            1,
            2,
            4,
            8,
            12,
            16,
        ),
        teacher_sigma_shift: Optional[float] = None,
        student_capture_timestep_mode: str = "per_chunk",
        initialize_student_from_teacher: bool = True,
    ) -> None:
        original_initialize_from_teacher = student.initialize_from_teacher
        if not initialize_student_from_teacher:
            student.initialize_from_teacher = lambda teacher: None
        try:
            super().__init__(
                teacher=teacher,
                student=student,
                teacher_task_name=teacher_task_name,
                student_task_name=student_task_name,
                teacher_num_inference_steps=teacher_num_inference_steps,
                teacher_capture_step_indices=teacher_capture_step_indices,
                teacher_sigma_shift=teacher_sigma_shift,
            )
        finally:
            student.initialize_from_teacher = original_initialize_from_teacher
        mode = str(student_capture_timestep_mode).strip().lower()
        if mode not in self.STUDENT_CAPTURE_TIMESTEP_MODES:
            raise ValueError(
                f"`student_capture_timestep_mode` must be one of "
                f"{self.STUDENT_CAPTURE_TIMESTEP_MODES}, got {student_capture_timestep_mode}"
            )
        self.student_capture_timestep_mode = mode

    def get_additional_trainable_modules(self):
        getter = getattr(self.student, "get_additional_trainable_modules", None)
        if not callable(getter):
            return {}
        modules = getter()
        return {} if modules is None else modules

    def _sample_chunk_anchor_indices(
        self,
        *,
        batch_size: int,
        num_chunks: int,
        num_anchor_candidates: int,
    ) -> torch.Tensor:
        if self.student_capture_timestep_mode == "shared":
            anchor_index = torch.randint(
                low=0,
                high=num_anchor_candidates,
                size=(batch_size, 1),
                device=self.device,
            )
            return anchor_index.expand(batch_size, num_chunks)
        return torch.randint(
            low=0,
            high=num_anchor_candidates,
            size=(batch_size, num_chunks),
            device=self.device,
        )

    def _prepare_offset_action_targets(
        self,
        sample,
        *,
        action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        model = self.student.model
        has_offset = getattr(model, "_has_action_offset", None)
        if not callable(has_offset) or not has_offset(sample):
            return action, action_is_pad, None

        batch_size = int(action.shape[0])
        offsets = model._normalize_action_offsets(sample, batch_size=batch_size)
        action_horizon = int(getattr(model, "action_horizon", int(action.shape[1])))
        action = model._slice_offset_sequence(
            action.to(device=offsets.device),
            offsets=offsets,
            action_horizon=action_horizon,
        )
        if action_is_pad is not None:
            action_is_pad = model._slice_offset_sequence(
                action_is_pad.to(device=offsets.device),
                offsets=offsets,
                action_horizon=action_horizon,
            )
        return action, action_is_pad, offsets

    def training_loss(self, sample, tiled: bool = False):
        if "video" not in sample or "action" not in sample:
            raise ValueError("`sample['video']` and `sample['action']` are required.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}"
            )
        action_is_pad = sample.get("action_is_pad", None)
        action, action_is_pad, offsets = self._prepare_offset_action_targets(
            sample,
            action=action,
            action_is_pad=action_is_pad,
        )
        batch_size = int(action.shape[0])
        action_horizon = int(action.shape[1])
        chunk_size = int(self.student.model.action_chunk_size)
        if action_horizon % chunk_size != 0:
            raise ValueError(
                "`sample['action']` horizon must be divisible by `action_chunk_size`, "
                f"got action_horizon={action_horizon}, action_chunk_size={chunk_size}."
            )
        num_chunks = action_horizon // chunk_size

        with torch.no_grad():
            teacher_rollout = self.teacher.rollout_action_latent_states(
                sample=sample,
                num_inference_steps=self.teacher_num_inference_steps,
                sigma_shift=self.teacher_sigma_shift,
                capture_step_indices=self.teacher_capture_step_indices,
                tiled=tiled,
            )

        video_state = teacher_rollout["video_state"]
        anchor_candidates = tuple(
            int(step) for step in teacher_rollout["capture_step_indices"][:-1]
        )
        if not anchor_candidates:
            raise ValueError(
                "Teacher capture steps must contain at least one non-final anchor."
            )

        anchor_index = self._sample_chunk_anchor_indices(
            batch_size=batch_size,
            num_chunks=num_chunks,
            num_anchor_candidates=len(anchor_candidates),
        )
        anchor_step_tensor = torch.tensor(
            anchor_candidates, device=self.device, dtype=torch.long
        )[anchor_index]

        captured_state_stack = torch.stack(
            [
                teacher_rollout["captured_states"][step]
                for step in teacher_rollout["capture_step_indices"][:-1]
            ],
            dim=1,
        ).to(device=self.device, dtype=self.torch_dtype)

        action_dim = int(captured_state_stack.shape[-1])
        captured_state_stack = captured_state_stack.view(
            batch_size, len(anchor_candidates), num_chunks, chunk_size, action_dim
        )
        noisy_chunk_stack = torch.gather(
            captured_state_stack.permute(0, 2, 1, 3, 4),
            dim=2,
            index=anchor_index.view(batch_size, num_chunks, 1, 1, 1).expand(
                -1, -1, 1, chunk_size, action_dim
            ),
        ).squeeze(2)
        noisy_action_latents = noisy_chunk_stack.reshape(batch_size, action_horizon, -1)

        timesteps = teacher_rollout["timesteps"].to(
            device=self.device, dtype=self.torch_dtype
        )
        timestep_action = timesteps[anchor_step_tensor].to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        target_action = teacher_rollout["final_latents"].to(
            device=self.device, dtype=self.torch_dtype
        )

        student_pred = self.student.predict_action_chunk(
            noisy_action_latents=noisy_action_latents,
            timestep_action=timestep_action,
            conditioning=video_state,
        )

        action_is_pad = (
            None
            if action_is_pad is None
            else action_is_pad.to(device=self.device, dtype=torch.bool)
        )
        action_loss_per_sample = self._masked_action_mse(
            pred_action=student_pred["pred_action"],
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        loss = action_loss_per_sample.mean()
        loss_dict = {
            "loss_ode": float(loss.detach().item()),
        }
        if offsets is not None:
            loss_dict["mean_offset"] = float(offsets.float().mean().item())
        return loss, loss_dict

    @torch.no_grad()
    def evaluate_action_val_losses(
        self,
        sample,
        *,
        denoise_steps: tuple[int, ...] = (1, 2, 4),
        tiled: bool = False,
    ) -> dict[str, float]:
        if "video" not in sample or "action" not in sample:
            return {}

        action = sample["action"]
        if action.ndim != 3:
            return {}

        action_is_pad = sample.get("action_is_pad", None)
        gt_action, action_is_pad, _ = self._prepare_offset_action_targets(
            sample,
            action=action,
            action_is_pad=action_is_pad,
        )
        teacher_rollout = self.teacher.rollout_action_latent_states(
            sample=sample,
            num_inference_steps=self.teacher_num_inference_steps,
            sigma_shift=self.teacher_sigma_shift,
            capture_step_indices=self.teacher_capture_step_indices,
            tiled=tiled,
        )
        target_action = teacher_rollout["final_latents"].to(
            device=self.device, dtype=self.torch_dtype
        )
        gt_action = gt_action.to(device=self.device, dtype=self.torch_dtype)
        action_is_pad = (
            None
            if action_is_pad is None
            else action_is_pad.to(device=self.device, dtype=torch.bool)
        )

        metrics: dict[str, float] = {}
        teacher_loss = self._masked_action_mse(
            pred_action=target_action,
            target_action=gt_action,
            action_is_pad=action_is_pad,
        )
        metrics["val_loss_teacher_rollout_vs_gt"] = float(teacher_loss.mean().item())

        student_infer_action = getattr(self.student, "infer_action")
        for step_count in denoise_steps:
            pred = student_infer_action(  # pyright: ignore[reportCallIssue]
                prompt=None,
                input_image=torch.empty(0),
                action_horizon=0,
                num_inference_steps=int(step_count),
                start_latents=teacher_rollout["initial_latents"],
                sample=sample,
                tiled=tiled,
            )
            loss_per_sample = self._masked_action_mse(
                pred_action=pred["final_latents"].to(
                    device=self.device, dtype=self.torch_dtype
                ),
                target_action=target_action,
                action_is_pad=action_is_pad,
            )
            metrics[f"val_loss_action_{int(step_count)}step"] = float(
                loss_per_sample.mean().item()
            )
        return metrics
