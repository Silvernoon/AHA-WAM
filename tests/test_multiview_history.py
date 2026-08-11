import unittest

import torch

from ahawam.models.wan22.ahawam import AHAWAM
from ahawam.trainer import Wan22Trainer


class _HistoryEncoder:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def _encode_input_image_latents_tensor(self, input_image, tiled=False):
        del tiled
        return input_image[:, :1].unsqueeze(2)

class _EvalHistoryNormalizer:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    @staticmethod
    def _configured_num_history_frames():
        return 6

    @staticmethod
    def _has_shared_visual_stem():
        return True


class MultiviewHistoryTest(unittest.TestCase):
    def test_history_encoder_preserves_batch_view_and_time_axes(self):
        batch_size, num_history_frames, num_views = 2, 3, 2
        history = torch.zeros(batch_size, num_history_frames, num_views, 3, 2, 2)
        for batch_idx in range(batch_size):
            for frame_idx in range(num_history_frames):
                for view_idx in range(num_views):
                    history[batch_idx, frame_idx, view_idx].fill_(
                        100 * batch_idx + 10 * frame_idx + view_idx
                    )

        latents = AHAWAM._encode_training_history_latents(
            _HistoryEncoder(),
            {"video_history": history},
            batch_size=batch_size,
            num_history_frames=num_history_frames,
            tiled=False,
        )

        self.assertEqual(latents.shape, (2, 2, 1, 3, 2, 2))
        for batch_idx in range(batch_size):
            for frame_idx in range(num_history_frames):
                for view_idx in range(num_views):
                    expected = 100 * batch_idx + 10 * frame_idx + view_idx
                    self.assertTrue(
                        torch.equal(
                            latents[batch_idx, view_idx, 0, frame_idx],
                            torch.full((2, 2), expected, dtype=torch.float32),
                        )
                    )

    def test_eval_batching_accepts_multiview_chunk_and_history_images(self):
        sample = {
            "video": torch.zeros(9, 3, 3, 8, 8),
            "prompt": "test",
            "action": torch.zeros(80, 14),
            "proprio": torch.zeros(80, 14),
            "context": torch.zeros(8, 12),
            "context_mask": torch.ones(8, dtype=torch.bool),
            "action_offset": torch.tensor(5),
            "chunk_obs_images": torch.zeros(5, 3, 3, 8, 8),
            "chunk_obs_images_no_offset": torch.zeros(5, 3, 3, 8, 8),
            "video_history": torch.zeros(6, 3, 3, 8, 8),
            "video_history_valid_len": torch.tensor(6),
            "video_history_frame_indices": torch.arange(6),
        }

        batched = Wan22Trainer._to_batched_eval_sample(sample)

        self.assertEqual(batched["video"].shape, (1, 9, 3, 3, 8, 8))
        self.assertEqual(
            batched["chunk_obs_images"].shape, (1, 5, 3, 3, 8, 8)
        )
        self.assertEqual(
            batched["chunk_obs_images_no_offset"].shape, (1, 5, 3, 3, 8, 8)
        )
        self.assertEqual(
            batched["video_history"].shape, (1, 6, 3, 3, 8, 8)
        )
        self.assertEqual(batched["video_history_frame_indices"].shape, (1, 6))


    def test_eval_history_normalization_preserves_multiview_layout(self):
        history = torch.arange(1 * 6 * 3 * 3 * 2 * 2, dtype=torch.float32).reshape(
            1, 6, 3, 3, 2, 2
        )

        normalized = AHAWAM._normalize_eval_video_history(
            _EvalHistoryNormalizer(),
            video_history=history,
            video_history_valid_len=torch.tensor([4]),
            batch_size=1,
        )

        self.assertEqual(normalized.shape, (1, 3, 3, 4, 2, 2))
        expected = history[:, -4:].permute(0, 2, 3, 1, 4, 5)
        self.assertTrue(torch.equal(normalized, expected))

if __name__ == "__main__":
    unittest.main()
