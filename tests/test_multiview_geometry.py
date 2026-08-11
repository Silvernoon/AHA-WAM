import unittest

import torch
from ahawam.models.wan22.ahawam_chunk_base import AHAWAMChunkBase

from ahawam.models.vision.multiview_geometry import (
    GeometryAwareCrossViewAttention,
    Latent3DRelationAlignment,
    SharedGeometricTokenResampler,
)


class MultiviewGeometryTest(unittest.TestCase):
    def test_resampler_returns_fixed_shared_budget(self):
        module = SharedGeometricTokenResampler(
            input_dim=16,
            output_dim=24,
            hidden_dim=32,
            num_queries=5,
            num_heads=4,
            num_layers=2,
            ffn_dim=64,
            num_views=3,
        )
        tokens, mask = module(torch.randn(2, 3, 35, 16))
        self.assertEqual(tokens.shape, (2, 5, 24))
        self.assertEqual(mask.shape, (2, 5))
        self.assertTrue(mask.all())

    def test_geo_rope_cross_view_attention_preserves_shape(self):
        module = GeometryAwareCrossViewAttention(
            hidden_dim=16,
            num_heads=2,
            head_dim=8,
            use_geo_rope=True,
        )
        tokens = torch.randn(1, 3, 2, 6, 16)
        intrinsics = torch.tensor(
            [[4.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
        ).reshape(1, 1, 3, 3).expand(1, 3, 3, 3).clone()
        extrinsics = torch.eye(4).reshape(1, 1, 4, 4).expand(1, 3, 4, 4).clone()
        output = module(
            tokens,
            grid_size=(2, 3),
            image_size=(4, 6),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        self.assertEqual(output.shape, tokens.shape)
        self.assertTrue(torch.equal(output, tokens))
        module.gate.data.fill_(1.0)
        output = module(
            tokens,
            grid_size=(2, 3),
            image_size=(4, 6),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_relation_alignment_returns_spatial_and_temporal_losses(self):
        module = Latent3DRelationAlignment(
            video_dim=12,
            teacher_dim=16,
            relation_dim=8,
            spatial_anchors=3,
            temporal_anchors=4,
        )
        spatial, temporal = module(
            video_tokens=torch.randn(2, 3, 2, 2, 3, 12),
            teacher_tokens=torch.randn(2, 3, 2, 4, 5, 16),
            teacher_grid_size=(4, 5),
            video_grid_size=(2, 3),
        )
        self.assertEqual(spatial.ndim, 0)
        self.assertEqual(temporal.ndim, 0)
        self.assertTrue(torch.isfinite(spatial + temporal))

    def test_two_view_video_composition_preserves_world_resolution(self):
        video = torch.randn(1, 2, 3, 4, 48, 64)
        composed = AHAWAMChunkBase._compose_robotwin_multiview_video(video)
        self.assertEqual(composed.shape, (1, 3, 4, 384, 320))


if __name__ == "__main__":
    unittest.main()
