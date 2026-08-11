from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DA3VisualBackbone(nn.Module):
    """Frozen Depth Anything 3 backbone exposed as dense multi-view tokens."""

    def __init__(
        self,
        *,
        model_id: str = "depth-anything/DA3-BASE",
        revision: str | None = None,
        input_size: tuple[int, int] | list[int] = (238, 322),
        use_camera_encoder: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        frozen: bool = True,
    ) -> None:
        super().__init__()
        try:
            from depth_anything_3.cfg import create_object
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "DA3 visual conditioning requires `depth-anything-3` and safetensors."
            ) from exc

        checkpoint_dir = Path(model_id)
        if checkpoint_dir.is_dir():
            config_path = checkpoint_dir / "config.json"
            weights_path = checkpoint_dir / "model.safetensors"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(model_id, "config.json", revision=revision)
            )
            weights_path = Path(
                hf_hub_download(model_id, "model.safetensors", revision=revision)
            )
        checkpoint_config = json.loads(config_path.read_text())
        model_config = checkpoint_config["config"]
        checkpoint_state = load_file(str(weights_path), device="cpu")

        self.backbone = create_object(model_config["net"])
        backbone_prefix = "model.backbone."
        backbone_state = {
            key.removeprefix(backbone_prefix): value
            for key, value in checkpoint_state.items()
            if key.startswith(backbone_prefix)
        }
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.camera_encoder = None
        if use_camera_encoder:
            self.camera_encoder = create_object(model_config["cam_enc"])
            camera_prefix = "model.cam_enc."
            camera_state = {
                key.removeprefix(camera_prefix): value
                for key, value in checkpoint_state.items()
                if key.startswith(camera_prefix)
            }
            self.camera_encoder.load_state_dict(camera_state, strict=True)
        del checkpoint_state
        self.model_id = str(model_id)
        self.revision = revision
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.use_camera_encoder = bool(use_camera_encoder)
        self.ref_view_strategy = str(ref_view_strategy)
        self.frozen = bool(frozen)
        patch_size = getattr(self.backbone.pretrained, "patch_size", 14)
        self.patch_size = int(
            patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
        )

        if self.input_size[0] % self.patch_size or self.input_size[1] % self.patch_size:
            raise ValueError(
                f"DA3 input_size must be divisible by patch_size={self.patch_size}, "
                f"got {self.input_size}."
            )
        if self.frozen:
            self.requires_grad_(False)
            super().train(False)

    def train(self, mode: bool = True):
        if self.frozen:
            return super().train(False)
        return super().train(mode)

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        return (images - mean) / std

    def _prepare_images(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                "DA3 images must be [B,V,3,H,W], "
                f"got shape {tuple(images.shape)}."
            )
        images = images.float()
        if float(images.detach().amin().item()) < -0.05:
            images = (images + 1.0) * 0.5
        images = images.clamp(0.0, 1.0)
        batch_size, num_views, _, old_h, old_w = images.shape
        new_h, new_w = self.input_size
        if (old_h, old_w) != (new_h, new_w):
            images = F.interpolate(
                images.reshape(batch_size * num_views, 3, old_h, old_w),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(batch_size, num_views, 3, new_h, new_w)
            if intrinsics is not None:
                intrinsics = intrinsics.clone()
                intrinsics[..., 0, :] *= float(new_w) / float(old_w)
                intrinsics[..., 1, :] *= float(new_h) / float(old_h)
        return self._normalize(images), intrinsics

    def forward(
        self,
        images: torch.Tensor,
        *,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        images, intrinsics = self._prepare_images(images, intrinsics)
        backbone_dtype = next(self.backbone.parameters()).dtype
        images = images.to(dtype=backbone_dtype)
        if self.use_camera_encoder and (extrinsics is None or intrinsics is None):
            raise ValueError(
                "DA3 camera conditioning requires both extrinsics and intrinsics."
            )
        if extrinsics is not None:
            extrinsics = extrinsics.to(device=images.device, dtype=torch.float32)
        if intrinsics is not None:
            intrinsics = intrinsics.to(device=images.device, dtype=torch.float32)

        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            cam_token = None
            if self.camera_encoder is not None:
                camera_dtype = next(self.camera_encoder.parameters()).dtype
                cam_token = self.camera_encoder(
                    extrinsics.to(dtype=camera_dtype),
                    intrinsics.to(dtype=camera_dtype),
                    images.shape[-2:],
                )
            features, _ = self.backbone(
                images,
                cam_token=cam_token,
                export_feat_layers=[],
                ref_view_strategy=self.ref_view_strategy,
            )
            patch_tokens, camera_tokens = features[-1]

        grid_h = images.shape[-2] // self.patch_size
        grid_w = images.shape[-1] // self.patch_size
        if patch_tokens.shape[2] != grid_h * grid_w:
            raise ValueError(
                "DA3 patch-token/grid mismatch: "
                f"tokens={patch_tokens.shape[2]} grid={grid_h}x{grid_w}."
            )
        return {
            "patch_tokens": patch_tokens,
            "camera_tokens": camera_tokens,
            "grid_size": torch.tensor(
                [grid_h, grid_w], device=patch_tokens.device, dtype=torch.long
            ),
            "images": images,
        }
