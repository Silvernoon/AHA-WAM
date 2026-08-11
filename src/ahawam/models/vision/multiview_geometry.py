from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResamplerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, queries: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = queries
        normalized_context = self.context_norm(context)
        queries, _ = self.cross_attn(
            self.query_norm(queries), normalized_context, normalized_context, need_weights=False
        )
        queries = residual + queries
        return queries + self.ffn(self.ffn_norm(queries))


class SharedGeometricTokenResampler(nn.Module):
    """Compress dense DA3 view tokens into a fixed shared token budget."""

    def __init__(self, *, input_dim: int, output_dim: int = 4096, hidden_dim: int = 1024,
                 num_queries: int = 32, num_heads: int = 16, num_layers: int = 2,
                 ffn_dim: int = 4096, num_views: int = 3) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.input_dim, self.output_dim = int(input_dim), int(output_dim)
        self.hidden_dim, self.num_queries, self.num_views = int(hidden_dim), int(num_queries), int(num_views)
        self.input_proj = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.view_embedding = nn.Parameter(torch.randn(num_views, hidden_dim) / math.sqrt(hidden_dim))
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim) / math.sqrt(hidden_dim))
        self.blocks = nn.ModuleList([_ResamplerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.ndim != 4:
            raise ValueError(f"patch_tokens must be [B,V,P,C], got {tuple(patch_tokens.shape)}.")
        batch_size, num_views, tokens_per_view, input_dim = patch_tokens.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected DA3 token dim {self.input_dim}, got {input_dim}.")
        if num_views > self.num_views:
            raise ValueError(f"Configured for at most {self.num_views} views, got {num_views}.")
        context = self.input_proj(patch_tokens)
        context = context + self.view_embedding[:num_views].view(1, num_views, 1, self.hidden_dim)
        context = context.reshape(batch_size, num_views * tokens_per_view, self.hidden_dim)
        queries = self.queries.expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, context)
        tokens = torch.tanh(self.output_gate) * self.output_proj(self.output_norm(queries))
        mask = torch.ones((batch_size, self.num_queries), dtype=torch.bool, device=tokens.device)
        return tokens, mask


def _rotation_matrix_to_euler_xyz(rotation: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
    singular = sy < 1e-6
    x = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    y = torch.atan2(-rotation[..., 2, 0], sy)
    z = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    xs = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    return torch.stack([torch.where(singular, xs, x), y, torch.where(singular, torch.zeros_like(z), z)], dim=-1)


def _continuous_rope(x: torch.Tensor, coordinates: torch.Tensor, *, theta: float) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("Geo-RoPE subspace dimension must be even.")
    pair_count = x.shape[-1] // 2
    indices = torch.arange(pair_count, device=x.device) % coordinates.shape[-1]
    phase = coordinates.index_select(-1, indices).float()
    phase = phase * theta ** (-torch.arange(pair_count, device=x.device, dtype=torch.float32) / max(pair_count, 1))
    while phase.ndim < x.ndim - 1:
        phase = phase.unsqueeze(-2)
    cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    pairs = x.reshape(*x.shape[:-1], pair_count, 2)
    first, second = pairs.unbind(-1)
    return torch.stack([first * cos - second * sin, first * sin + second * cos], dim=-1).flatten(-2)


class GeometryAwareCrossViewAttention(nn.Module):
    """Per-frame cross-view attention with optional ray/pose Geo-RoPE."""

    def __init__(self, *, hidden_dim: int, num_heads: int, head_dim: int,
                 use_geo_rope: bool = True, rope_theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for split Geo-RoPE.")
        self.hidden_dim, self.num_heads, self.head_dim = int(hidden_dim), int(num_heads), int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.use_geo_rope, self.rope_theta = bool(use_geo_rope), float(rope_theta)
        self.norm = nn.LayerNorm(hidden_dim)
        self.q, self.k, self.v = nn.Linear(hidden_dim, self.inner_dim), nn.Linear(hidden_dim, self.inner_dim), nn.Linear(hidden_dim, self.inner_dim)
        self.out = nn.Linear(self.inner_dim, hidden_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _geometry_coordinates(*, intrinsics: torch.Tensor, extrinsics: torch.Tensor,
                              grid_size: tuple[int, int], image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if intrinsics.ndim != 4 or intrinsics.shape[-2:] != (3, 3):
            raise ValueError("intrinsics must be [B,V,3,3].")
        if extrinsics.ndim != 4 or extrinsics.shape[-2:] not in ((4, 4), (3, 4)):
            raise ValueError("extrinsics must be [B,V,4,4] or [B,V,3,4].")
        batch_size, num_views = intrinsics.shape[:2]
        grid_h, grid_w = grid_size
        image_h, image_w = image_size
        ys = (torch.arange(grid_h, device=intrinsics.device, dtype=torch.float32) + 0.5) * (image_h / grid_h)
        xs = (torch.arange(grid_w, device=intrinsics.device, dtype=torch.float32) + 0.5) * (image_w / grid_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(1, 1, -1, 3).expand(batch_size, num_views, -1, -1)
        rays = torch.einsum("bvij,bvpj->bvpi", torch.linalg.inv(intrinsics.float()), pixels)
        rotation, translation = extrinsics[..., :3, :3].float(), extrinsics[..., :3, 3].float()
        rays = F.normalize(torch.einsum("bvji,bvpj->bvpi", rotation, rays), dim=-1)
        euler = _rotation_matrix_to_euler_xyz(rotation)
        position = -torch.einsum("bvji,bvj->bvi", rotation, translation)
        optical = torch.einsum("bvji,j->bvi", rotation, rotation.new_tensor([0.0, 0.0, 1.0]))
        return rays, torch.cat([euler, translation, position, optical], dim=-1)

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int], image_size: tuple[int, int],
                intrinsics: torch.Tensor | None = None, extrinsics: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError(f"Cross-view tokens must be [B,V,F,P,D], got {tuple(tokens.shape)}.")
        batch_size, num_views, num_frames, token_count, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim or token_count != grid_size[0] * grid_size[1]:
            raise ValueError("Cross-view token shape/config mismatch.")
        normalized = self.norm(tokens)
        shape = (batch_size, num_views, num_frames, token_count, self.num_heads, self.head_dim)
        q, k, v = self.q(normalized).reshape(shape), self.k(normalized).reshape(shape), self.v(normalized).reshape(shape)
        if self.use_geo_rope:
            if intrinsics is None or extrinsics is None:
                raise ValueError("Geo-RoPE requires intrinsics and extrinsics.")
            rays, pose = self._geometry_coordinates(intrinsics=intrinsics.to(tokens.device), extrinsics=extrinsics.to(tokens.device), grid_size=grid_size, image_size=image_size)
            rays, pose = rays.unsqueeze(2).unsqueeze(-2), pose.unsqueeze(2).unsqueeze(3).unsqueeze(-2)
            split = self.head_dim // 2
            q = torch.cat([_continuous_rope(q[..., :split], rays, theta=self.rope_theta), _continuous_rope(q[..., split:], pose, theta=self.rope_theta)], dim=-1)
            k = torch.cat([_continuous_rope(k[..., :split], rays, theta=self.rope_theta), _continuous_rope(k[..., split:], pose, theta=self.rope_theta)], dim=-1)
        def flatten(x: torch.Tensor) -> torch.Tensor:
            return x.permute(0, 2, 4, 1, 3, 5).reshape(batch_size * num_frames, self.num_heads, num_views * token_count, self.head_dim)
        attended = F.scaled_dot_product_attention(flatten(q), flatten(k), flatten(v))
        attended = attended.reshape(batch_size, num_frames, self.num_heads, num_views, token_count, self.head_dim).permute(0, 3, 1, 4, 2, 5).reshape(batch_size, num_views, num_frames, token_count, self.inner_dim)
        return tokens + self.gate.to(tokens.dtype) * self.out(attended)


class Latent3DRelationAlignment(nn.Module):
    """Anchor-sampled spatial and temporal relation distillation from DA3."""

    def __init__(self, *, video_dim: int, teacher_dim: int, relation_dim: int = 512,
                 spatial_anchors: int = 64, temporal_anchors: int = 128) -> None:
        super().__init__()
        self.teacher_dim, self.spatial_anchors, self.temporal_anchors = int(teacher_dim), int(spatial_anchors), int(temporal_anchors)
        self.video_projector = nn.Sequential(nn.LayerNorm(video_dim), nn.Linear(video_dim, relation_dim))
        self.teacher_projector = nn.Sequential(nn.LayerNorm(teacher_dim), nn.Linear(teacher_dim, relation_dim, bias=False))
        self.teacher_projector.requires_grad_(False)

    @staticmethod
    def _relation_loss(student: torch.Tensor, teacher: torch.Tensor, anchor_count: int) -> torch.Tensor:
        if student.shape != teacher.shape:
            raise ValueError(f"REPA shape mismatch: {student.shape} vs {teacher.shape}.")
        anchors = torch.randperm(student.shape[-2], device=student.device)[:min(anchor_count, student.shape[-2])]
        student, teacher = F.normalize(student.float(), dim=-1), F.normalize(teacher.detach().float(), dim=-1)
        return F.smooth_l1_loss(torch.matmul(student, student.index_select(-2, anchors).transpose(-1, -2)), torch.matmul(teacher, teacher.index_select(-2, anchors).transpose(-1, -2)))

    def forward(self, *, video_tokens: torch.Tensor, teacher_tokens: torch.Tensor,
                teacher_grid_size: tuple[int, int], video_grid_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if video_tokens.ndim != 6 or teacher_tokens.ndim != 6:
            raise ValueError("REPA tokens must be [B,F,V,H,W,D].")
        batch_size, num_frames, num_views, video_h, video_w, _ = video_tokens.shape
        teacher_h, teacher_w = teacher_grid_size
        if video_tokens.shape[3:5] != video_grid_size or teacher_tokens.shape[:3] != (batch_size, num_frames, num_views) or teacher_tokens.shape[3:5] != teacher_grid_size:
            raise ValueError("REPA token layout mismatch.")
        teacher = teacher_tokens.permute(0, 1, 2, 5, 3, 4).reshape(batch_size * num_frames * num_views, self.teacher_dim, teacher_h, teacher_w)
        teacher = F.adaptive_avg_pool2d(teacher, video_grid_size).reshape(batch_size, num_frames, num_views, self.teacher_dim, video_h, video_w).permute(0, 1, 2, 4, 5, 3)
        student = self.video_projector(video_tokens)
        with torch.no_grad():
            teacher = self.teacher_projector(teacher)
        spatial_student, spatial_teacher = student.reshape(batch_size * num_frames, num_views * video_h * video_w, -1), teacher.reshape(batch_size * num_frames, num_views * video_h * video_w, -1)
        temporal_student, temporal_teacher = student.reshape(batch_size, num_frames * num_views * video_h * video_w, -1), teacher.reshape(batch_size, num_frames * num_views * video_h * video_w, -1)
        return self._relation_loss(spatial_student, spatial_teacher, self.spatial_anchors), self._relation_loss(temporal_student, temporal_teacher, self.temporal_anchors)
