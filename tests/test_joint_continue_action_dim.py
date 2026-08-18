from __future__ import annotations

import unittest

import torch

from experiments.tworoom.joint_continue_tworoom import action_encoder_input_dim


class ActionEncoderInputDimensionTest(unittest.TestCase):
    def test_public_input_dim_wins_over_internal_projection_width(self):
        class EmbedderLike(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = 25
                self.patch_embed = torch.nn.Conv1d(25, 10, kernel_size=1)
                self.embed = torch.nn.Sequential(
                    torch.nn.Linear(10, 32),
                    torch.nn.SiLU(),
                    torch.nn.Linear(32, 192),
                )

        self.assertEqual(action_encoder_input_dim(EmbedderLike()), 25)

    def test_conv_input_channels_are_used_without_public_metadata(self):
        encoder = torch.nn.Sequential(
            torch.nn.Conv1d(25, 10, kernel_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(10, 192),
        )
        self.assertEqual(action_encoder_input_dim(encoder), 25)

    def test_linear_only_encoder_remains_supported(self):
        encoder = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 192),
        )
        self.assertEqual(action_encoder_input_dim(encoder), 10)

    def test_unknown_encoder_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "Cannot infer action input"):
            action_encoder_input_dim(torch.nn.Identity())


if __name__ == "__main__":
    unittest.main()
