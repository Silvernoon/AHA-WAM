import unittest

import torch

from ahawam.models.wan22.wan_video_dit import WanVideoDiT


class MultiviewVideoDiTTest(unittest.TestCase):
    def test_pre_and_post_dit_preserve_explicit_view_axis(self):
        model = WanVideoDiT(
            hidden_dim=16,
            in_dim=4,
            ffn_dim=32,
            out_dim=4,
            text_dim=12,
            freq_dim=8,
            eps=1e-6,
            patch_size=(1, 2, 2),
            num_heads=2,
            attn_head_dim=8,
            num_layers=1,
            has_image_input=False,
            seperated_timestep=True,
            require_vae_embedding=False,
            require_clip_embedding=False,
            fuse_vae_embedding_in_latents=True,
        )
        latents = torch.randn(2, 3, 4, 3, 4, 6)
        state = model.pre_dit(
            x=latents,
            timestep=torch.rand(2),
            context=torch.randn(2, 5, 12),
            context_mask=torch.ones(2, 5, dtype=torch.bool),
            fuse_vae_embedding_in_latents=True,
        )
        self.assertEqual(state["tokens"].shape, (2, 3 * 3 * 2 * 3, 16))
        self.assertEqual(state["meta"]["num_views"], 3)
        self.assertEqual(state["meta"]["tokens_per_frame"], 3 * 2 * 3)
        output = model.post_dit(state["tokens"], state)
        self.assertEqual(output.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
