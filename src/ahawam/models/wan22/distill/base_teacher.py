from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseTeacher(nn.Module, ABC):
    """Role wrapper for ODE teachers."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattr__("model")
            return getattr(model, name)

    @abstractmethod
    @torch.no_grad()
    def rollout_action_latent_states(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def export_model_state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: dict[str, torch.Tensor], *, strict: bool = False):
        return self.model.load_state_dict(state_dict, strict=strict)

    def save_checkpoint(self, path, optimizer=None, step=None):
        return self.model.save_checkpoint(path, optimizer=optimizer, step=step)

    def load_checkpoint(self, path, optimizer=None):
        return self.model.load_checkpoint(path, optimizer=optimizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        self.model.eval()
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
