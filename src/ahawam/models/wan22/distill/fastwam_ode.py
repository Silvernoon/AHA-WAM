from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ahawam.utils.logging_config import get_logger

from .base_student import BaseStudent
from .base_teacher import BaseTeacher

logger = get_logger(__name__)


class FastWAMODE(nn.Module):
    """Minimal ODE distillation container that combines a frozen teacher and a trainable student."""

    def __init__(
        self,
        teacher: BaseTeacher,
        student: BaseStudent,
        teacher_task_name: str,
        student_task_name: str,
        teacher_num_inference_steps: int = 16,
        teacher_capture_step_indices: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8, 12, 16),
        teacher_sigma_shift: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.teacher = teacher.eval()
        self.teacher.requires_grad_(False)
        self.student = student
        self.student.initialize_from_teacher(self.teacher)

        self.teacher_task_name = str(teacher_task_name)
        self.student_task_name = str(student_task_name)
        self.teacher_num_inference_steps = int(teacher_num_inference_steps)
        self.teacher_capture_step_indices = tuple(sorted({int(x) for x in teacher_capture_step_indices}))
        self.teacher_sigma_shift = None if teacher_sigma_shift is None else float(teacher_sigma_shift)

        self.dit = self.student.dit
        self.device = self.student.device
        self.torch_dtype = self.student.torch_dtype
        self.video_expert = self.student.video_expert
        self.action_expert = self.student.action_expert
        self.mot = self.student.mot
        self.vae = self.student.vae
        self.text_encoder = self.student.text_encoder
        self.tokenizer = self.student.tokenizer
        self.proprio_encoder = getattr(self.student, "proprio_encoder", None)
        self.proprio_dim = getattr(self.student, "proprio_dim", None)

    @staticmethod
    def _masked_action_mse(
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    @torch.no_grad()
    def rollout_teacher_action_latent_states(
        self,
        *,
        input_image: torch.Tensor,
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        capture_step_indices: Optional[list[int] | tuple[int, ...]] = None,
    ) -> dict[str, Any]:
        use_capture_steps = self.teacher_capture_step_indices if capture_step_indices is None else capture_step_indices
        return self.teacher.rollout_action_latent_states(
            input_image=input_image,
            action_horizon=action_horizon,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=self.teacher_num_inference_steps,
            sigma_shift=self.teacher_sigma_shift,
            capture_step_indices=use_capture_steps,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def training_loss(self, sample, tiled: bool = False):
        if "video" not in sample or "action" not in sample:
            raise ValueError("`sample['video']` and `sample['action']` are required for FastWAMODE training.")

        video = sample["video"]
        action = sample["action"]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be [B,3,T,H,W], got {tuple(video.shape)}")
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be [B,T,D], got {tuple(action.shape)}")

        batch_size = int(video.shape[0])
        action_horizon = int(action.shape[1])
        action_is_pad = sample.get("action_is_pad", None)
        input_image = video[:, :, 0].to(device=self.device, dtype=self.torch_dtype)
        prompt = sample.get("prompt", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        proprio = sample.get("proprio", None)
        if context is None and isinstance(prompt, tuple):
            prompt = list(prompt)
        if proprio is not None:
            if proprio.ndim == 3:
                proprio = proprio[:, 0]
            elif proprio.ndim != 2:
                raise ValueError(f"`sample['proprio']` must be 2D or 3D, got {tuple(proprio.shape)}")

        # === 1. get teacher rollout ===
        with torch.no_grad():
            teacher_rollout = self.teacher.rollout_action_latent_states(
                input_image=input_image,
                action_horizon=action_horizon,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                num_inference_steps=self.teacher_num_inference_steps,
                sigma_shift=self.teacher_sigma_shift,
                capture_step_indices=self.teacher_capture_step_indices,
                tiled=tiled,
            )

            
        # === 2. prepare student input ===
        conditioning = self.student.prepare_action_conditioning(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            start_latents=teacher_rollout["initial_latents"],
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )

        # === 3. random anchor ===
        anchor_candidates = tuple(int(x) for x in teacher_rollout["capture_step_indices"][:-1])
        if not anchor_candidates:
            raise ValueError("Teacher capture steps must contain at least one non-final anchor state.")
        # choose a anchor for every batch
        anchor_index = torch.randint(
            low=0,
            high=len(anchor_candidates),
            size=(batch_size,),
            device=self.device,
        )
        
        anchor_step_tensor = torch.tensor(anchor_candidates, device=self.device, dtype=torch.long)[anchor_index]
        # shape (B,)

        captured_state_steps = teacher_rollout["capture_step_indices"]
        # content: (0,1,2,4,8,12,16)
        
        captured_state_stack = torch.stack(
            [teacher_rollout["captured_states"][step] for step in captured_state_steps[:-1]],
            dim=1,
        ).to(device=self.device, dtype=self.torch_dtype)
        # shape: (B, anchors, action_horizon, action_dim)
        
        noisy_action_latents = captured_state_stack[
            torch.arange(batch_size, device=self.device),
            anchor_index,
        ]
        # shape: (B, action_horizon, action_dim)

        
        # === 4. student pred action ===
        timesteps = teacher_rollout["timesteps"].to(device=self.device, dtype=self.torch_dtype)
        timestep_action = timesteps[anchor_step_tensor].to(device=self.device, dtype=self.torch_dtype)
        target_action = teacher_rollout["final_latents"].to(device=self.device, dtype=self.torch_dtype)

        student_pred = self.student.predict_action_chunk(
            noisy_action_latents=noisy_action_latents,
            timestep_action=timestep_action,
            conditioning=conditioning,
        )
        action_is_pad = None if action_is_pad is None else action_is_pad.to(device=self.device, dtype=torch.bool)
        action_loss_per_sample = self._masked_action_mse(
            pred_action=student_pred["pred_action"],
            target_action=target_action,
            action_is_pad=action_is_pad,
        )
        loss = action_loss_per_sample.mean()
        loss_dict = {
            "loss_ode": float(loss.detach().item()),
        }
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

        video = sample["video"]
        action = sample["action"]
        if video.ndim != 5 or action.ndim != 3:
            return {}

        action_horizon = int(action.shape[1])
        input_image = video[:, :, 0].to(device=self.device, dtype=self.torch_dtype)
        prompt = sample.get("prompt", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        proprio = sample.get("proprio", None)
        action_is_pad = sample.get("action_is_pad", None)
        if context is None and isinstance(prompt, tuple):
            prompt = list(prompt)
        if proprio is not None:
            if proprio.ndim == 3:
                proprio = proprio[:, 0]
            elif proprio.ndim != 2:
                raise ValueError(f"`sample['proprio']` must be 2D or 3D, got {tuple(proprio.shape)}")


        # === 1. teacher rollout ===
        teacher_rollout = self.teacher.rollout_action_latent_states(
            input_image=input_image,
            action_horizon=action_horizon,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=self.teacher_num_inference_steps,
            sigma_shift=self.teacher_sigma_shift,
            capture_step_indices=self.teacher_capture_step_indices,
            tiled=tiled,
        )
        target_action = teacher_rollout["final_latents"].to(device=self.device, dtype=self.torch_dtype)
        gt_action = action.to(device=self.device, dtype=self.torch_dtype)
        action_is_pad = None if action_is_pad is None else action_is_pad.to(device=self.device, dtype=torch.bool)

        # === 2. teacher rollout vs gt ===
        metrics: dict[str, float] = {}
        teacher_loss_per_sample = self._masked_action_mse(
            pred_action=target_action,
            target_action=gt_action,
            action_is_pad=action_is_pad,
        )
        metrics["val_loss_teacher_rollout_vs_gt"] = float(teacher_loss_per_sample.mean().item())

        # === 3. student vs teacher rollout ===
        for step_count in denoise_steps:
            pred = self.student.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                num_inference_steps=int(step_count),
                start_latents=teacher_rollout["initial_latents"],
                tiled=tiled,
            )
            loss_per_sample = self._masked_action_mse(
                pred_action=pred["final_latents"].to(device=self.device, dtype=self.torch_dtype),
                target_action=target_action,
                action_is_pad=action_is_pad,
            )
            metrics[f"val_loss_action_{int(step_count)}step"] = float(loss_per_sample.mean().item())
        return metrics

    @torch.no_grad()
    def evaluate_validation(
        self,
        sample,
        *,
        eval_num_inference_steps: int,
        eval_dir: Optional[str] = None,
        global_step: int = 0,
        process_index: int = 0,
    ) -> dict[str, float | str]:
        del eval_num_inference_steps, eval_dir, global_step, process_index
        val_loss, val_loss_dict = self.training_loss(sample)
        result: dict[str, float | str] = {
            "val_loss": float(val_loss.float().item()),
        }
        result.update(self.evaluate_action_val_losses(sample))
        for key, value in val_loss_dict.items():
            if str(key).startswith("val_"):
                result[str(key)] = float(value)
        return result

    @torch.no_grad()
    def predict_student_action_chunk(
        self,
        *,
        noisy_action_latents: torch.Tensor,
        timestep_action: torch.Tensor,
        conditioning: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        return self.student.predict_action_chunk(
            noisy_action_latents=noisy_action_latents,
            timestep_action=timestep_action,
            conditioning=conditioning,
        )

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        return self.student.infer(*args, **kwargs)

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        return self.student.infer_action(*args, **kwargs)

    @torch.no_grad()
    def infer_joint(self, *args, **kwargs):
        return self.student.infer_joint(*args, **kwargs)

    @torch.no_grad()
    def encode_prompt(self, *args, **kwargs):
        return self.student.encode_prompt(*args, **kwargs)

    @torch.no_grad()
    def _encode_video_latents(self, *args, **kwargs):
        return self.student._encode_video_latents(*args, **kwargs)

    @torch.no_grad()
    def _decode_latents(self, *args, **kwargs):
        return self.student._decode_latents(*args, **kwargs)

    def save_checkpoint(self, path, optimizer=None, step=None):
        self.student.save_checkpoint(path, optimizer=optimizer, step=step)
        payload = torch.load(path, map_location="cpu")
        payload["ode_metadata"] = {
            "teacher_task_name": self.teacher_task_name,
            "student_task_name": self.student_task_name,
            "teacher_num_inference_steps": self.teacher_num_inference_steps,
            "teacher_capture_step_indices": list(self.teacher_capture_step_indices),
            "teacher_sigma_shift": self.teacher_sigma_shift,
        }
        torch.save(payload, path)
        logger.info("Saved ODE student checkpoint with teacher metadata at %s", path)
        return payload

    def load_checkpoint(self, path, optimizer=None):
        return self.student.load_checkpoint(path, optimizer=optimizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.student.train(mode)
        return self

    def eval(self):
        super().eval()
        self.teacher.eval()
        self.student.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
