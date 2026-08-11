import unittest

import torch

from ahawam.models.wan22.action_dit import ActionDiT
from ahawam.models.wan22.mot import MoT
from ahawam.models.wan22.wan_video_dit import WanVideoDiT


class MultiviewMoTTest(unittest.TestCase):
    def test_joint_world_action_forward_with_cross_view_tokens(self):
        video = WanVideoDiT(
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
        action = ActionDiT(
            hidden_dim=16,
            action_dim=2,
            ffn_dim=32,
            text_dim=12,
            freq_dim=8,
            eps=1e-6,
            num_heads=2,
            attn_head_dim=8,
            num_layers=1,
            action_chunk_size=2,
        )
        mot = MoT(
            {"video": video, "action": action},
            mot_checkpoint_mixed_attn=False,
        )
        mot.configure_chunk_kv_cache_editor(query_dim=12)
        mot.configure_video_cross_view_attention(
            layer_indices=[0], hidden_dim=16, use_geo_rope=False
        )
        context = torch.randn(1, 5, 12)
        context_mask = torch.ones(1, 5, dtype=torch.bool)
        video_pre = video.pre_dit(
            x=torch.randn(1, 3, 4, 3, 4, 6),
            timestep=torch.rand(1),
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=True,
        )
        action_pre = action.pre_dit(
            action_tokens=torch.randn(1, 4, 2),
            timestep=torch.rand(1),
            context=context,
            context_mask=context_mask,
            chunk_size=2,
            single_branch_chunk_causal=True,
        )
        output = mot.forward_prior_action_with_chunk_updated_kv(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=torch.ones(
                video_pre["tokens"].shape[1],
                video_pre["tokens"].shape[1],
                dtype=torch.bool,
            ),
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            chunk_queries=torch.randn(1, 2, 3, 12),
            video_tokens_per_frame=18,
            action_chunk_size=2,
            video_view_shape=(3, 3, 6),
        )
        world = video.post_dit(output["video"], video_pre)
        actions = action.post_dit(output["action_prior"], action_pre)
        self.assertEqual(world.shape, (1, 3, 4, 3, 4, 6))
        self.assertEqual(actions.shape, (1, 4, 2))
        self.assertTrue(torch.isfinite(world).all())
        self.assertTrue(torch.isfinite(actions).all())


if __name__ == "__main__":
    unittest.main()
