#!/usr/bin/env python3
"""AHAWAM policy wrapper for real-robot deployment."""

from __future__ import annotations

import copy
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


class AHAWAMPolicy:
    """AHAWAM policy wrapper used by ``deploy/server/wam_policy_server.py``.

    ``policy_path`` points to a deployment YAML file. The public keys are:
    ``checkpoint_path``, ``dataset_stats_path``, ``project_root``,
    ``hydra_config_name``, ``task``, device/precision fields, and inference
    sampling fields. The wrapper consumes a front RGB image plus a 14-D robot
    state and returns a denormalized action chunk.
    """

    def __init__(self, policy_path: str, task_name: Optional[str] = None):
        init_t0 = time.perf_counter()
        self.task_name = task_name
        self.current_instruction = ""
        self._observation: Optional[dict[str, Any]] = None

        cfg = self._load_deploy_config(policy_path)
        project_root = cfg.get("project_root", Path(__file__).resolve().parents[2])
        self.project_root = Path(project_root).expanduser().resolve()
        src_root = self.project_root / "src"
        for path in (self.project_root, src_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from ahawam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self._DEFAULT_PROMPT = DEFAULT_PROMPT

        configs_root = (self.project_root / "configs").resolve()
        hydra_config_name = str(cfg.get("hydra_config_name", "deploy"))
        task = cfg.get("task")
        overrides = []
        if not _is_none_like(task):
            overrides.append(f"task={str(task)}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            hydra_cfg = compose(config_name=hydra_config_name, overrides=overrides)

        logger.info("Hydra config composed: config=%s task=%s", hydra_config_name, task)

        device = str(os.environ.get("AHAWAM_DEPLOY_DEVICE_OVERRIDE") or cfg.get("device", "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to cpu.")
            device = "cpu"
        self.device = device

        mixed_precision = str(cfg.get("mixed_precision", "bf16"))
        model_dtype = _mixed_precision_to_dtype(mixed_precision)

        checkpoint_path = cfg.get("checkpoint_path")
        if _is_none_like(checkpoint_path):
            raise ValueError("`checkpoint_path` is required in the deploy config YAML.")
        checkpoint_path = Path(str(checkpoint_path)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        dataset_stats_path = cfg.get("dataset_stats_path")
        if _is_none_like(dataset_stats_path):
            raise ValueError("`dataset_stats_path` is required in the deploy config YAML.")
        dataset_stats_path = Path(str(dataset_stats_path)).expanduser()
        if not dataset_stats_path.is_absolute():
            dataset_stats_path = self.project_root / dataset_stats_path
        if not dataset_stats_path.exists():
            raise FileNotFoundError(f"Dataset stats not found: {dataset_stats_path}")

        model_cfg = OmegaConf.create(OmegaConf.to_container(hydra_cfg.model, resolve=True))
        model_cfg.load_text_encoder = True

        logger.info("Instantiating AHAWAM model...")
        model_t0 = time.perf_counter()
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(checkpoint_path))
        self.model = self.model.to(device).eval()
        logger.info("Model loaded in %.1f ms", (time.perf_counter() - model_t0) * 1000.0)

        self.processor = instantiate(hydra_cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        logger.info("Processor and normalizer initialized from %s", dataset_stats_path)

        eval_cfg = hydra_cfg.get("EVALUATION", {})
        action_horizon = _parse_optional_int(cfg.get("action_horizon"))
        if action_horizon is None:
            action_horizon = _parse_optional_int(eval_cfg.get("action_horizon"))
        if action_horizon is None:
            action_horizon = int(hydra_cfg.data.train.num_frames) - 1
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        self.action_horizon = action_horizon

        self.num_inference_steps = _parse_optional_int(
            cfg.get("num_inference_steps", eval_cfg.get("num_inference_steps"))
        )
        if self.num_inference_steps is None:
            self.num_inference_steps = 10
        self.sigma_shift = _parse_optional_float(cfg.get("sigma_shift", eval_cfg.get("sigma_shift")))
        self.seed = _parse_optional_int(cfg.get("seed"))
        self.text_cfg_scale = float(cfg.get("text_cfg_scale", eval_cfg.get("text_cfg_scale", 1.0)))
        self.negative_prompt = str(cfg.get("negative_prompt", eval_cfg.get("negative_prompt", "")))
        self.rand_device = str(cfg.get("rand_device", eval_cfg.get("rand_device", "cpu")))
        self.tiled = bool(cfg.get("tiled", eval_cfg.get("tiled", False)))
        self.warmup_rounds = max(0, int(cfg.get("warmup_rounds", 1)))

        self.action_chunk_size = int(self.model.action_chunk_size)
        self.num_chunks = int(self.action_horizon // self.action_chunk_size)
        if self.num_chunks <= 0:
            raise ValueError("action_horizon must be at least one action chunk.")

        self._episode_prefilled = False
        self.video_height = int(cfg.get("video_height", 384))
        self.video_width = int(cfg.get("video_width", 320))

        logger.info(
            "AHAWAM policy ready | device=%s dtype=%s | horizon=%d chunk_size=%d num_chunks=%d | image=%dx%d",
            device,
            model_dtype,
            self.action_horizon,
            self.action_chunk_size,
            self.num_chunks,
            self.video_width,
            self.video_height,
        )
        if self.warmup_rounds > 0:
            self._warmup()
        logger.info("AHAWAM initialization complete in %.1f ms", (time.perf_counter() - init_t0) * 1000.0)

    def set_language_instruction(self, instruction: str):
        self.current_instruction = instruction
        logger.info("Instruction set: %r", instruction)

    def update_observation_window(self, img_arr, state):
        """Buffer one synchronized camera set and robot state vector."""
        has_shared_stem = getattr(self.model, "_has_shared_visual_stem", None)
        requires_multiview = bool(
            callable(has_shared_stem) and has_shared_stem()
        )
        expected_views = 3 if requires_multiview else 1
        if len(img_arr) != expected_views:
            raise ValueError(
                f"Expected {expected_views} synchronized camera views, got {len(img_arr)}."
            )

        image_tensors = []
        for image in img_arr:
            rgb = np.asarray(image)
            if rgb.shape[:2] != (self.video_height, self.video_width):
                pil_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
                pil_img = pil_img.resize(
                    (self.video_width, self.video_height),
                    resample=Image.BILINEAR,
                )
                rgb = np.asarray(pil_img, dtype=np.uint8)
            image_tensors.append(
                torch.from_numpy(rgb.copy()).permute(2, 0, 1)
            )
        image_tensor = torch.stack(image_tensors, dim=0).to(
            device=self.device, dtype=self.model.torch_dtype
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        if requires_multiview:
            image_tensor = image_tensor.unsqueeze(0)
        state_vector = np.asarray(state, dtype=np.float32).reshape(-1)
        self._observation = {
            "image_tensor": image_tensor,
            "state_vector": state_vector,
        }

    def get_action(self) -> np.ndarray:
        """Run one action chunk inference."""
        if self._observation is None:
            raise RuntimeError("Call update_observation_window before get_action.")

        image_tensor = self._observation["image_tensor"]
        state_vector = self._observation["state_vector"]

        if not hasattr(self.model, "_inference_state") or self.model._inference_state is None:
            next_chunk_index = 0
        else:
            next_chunk_index = int(self.model._inference_state.get("next_chunk_index", 0))
        if next_chunk_index >= self.num_chunks:
            self._soft_reset_for_new_observation()

        proprio = self._normalize_state(state_vector)
        prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)

        with torch.no_grad():
            if not self._episode_prefilled:
                self._prefill_episode(prompt=prompt, image_tensor=image_tensor)
            pred = self._infer_action_chunk(image_tensor=image_tensor, proprio=proprio)

        action_tensor = pred["action_chunk"].unsqueeze(0)
        return self._denormalize_action(action_tensor)[0]

    def reset_obsrvationwindows(self):
        self._observation = None
        self._hard_reset()
        logger.info("AHAWAM episode reset.")

    def eval(self):
        self.model.eval()

    def _prefill_episode(self, *, prompt: str, image_tensor: torch.Tensor) -> None:
        video_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            "phase": "video",
        }
        self.model.infer_action(**video_kwargs)
        self._episode_prefilled = True

    def _infer_action_chunk(self, *, image_tensor: torch.Tensor, proprio: torch.Tensor) -> dict[str, Any]:
        if not self._episode_prefilled:
            raise RuntimeError("Episode video state not initialized before action inference.")
        action_kwargs = {
            "chunk_obs_image": image_tensor,
            "chunk_proprio": proprio,
            "sigma_shift": self.sigma_shift,
            "tiled": self.tiled,
            "phase": "action",
        }
        if self.num_inference_steps is not None:
            action_kwargs["num_inference_steps"] = self.num_inference_steps
        return self.model.infer_action(**action_kwargs)

    def prefill_video_only(self, image_tensor: torch.Tensor, prompt: Optional[str] = None, deep_copy: bool = True) -> dict:
        """Run only the video phase and return a state snapshot for async serving."""
        if prompt is None:
            prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)
        with torch.no_grad():
            self._prefill_episode(prompt=prompt, image_tensor=image_tensor)
        state = self.model._inference_state
        if state is None:
            raise RuntimeError("Video prefill did not produce inference state.")
        if deep_copy:
            return copy.deepcopy(state)
        return state

    def infer_action_chunk_only(self, inference_state: dict, image_tensor: torch.Tensor, proprio: torch.Tensor) -> np.ndarray:
        """Run only the action phase from a provided async state snapshot."""
        prev_state = self.model._inference_state
        self.model._inference_state = inference_state
        self._episode_prefilled = True
        try:
            with torch.no_grad():
                pred = self._infer_action_chunk(image_tensor=image_tensor, proprio=proprio)
        finally:
            self.model._inference_state = prev_state
        action_tensor = pred["action_chunk"].unsqueeze(0)
        return self._denormalize_action(action_tensor)[0]

    def _hard_reset(self) -> None:
        if hasattr(self.model, "reset_history"):
            self.model.reset_history()
        if hasattr(self.model, "_inference_state"):
            self.model._inference_state = None
        self._episode_prefilled = False

    def _soft_reset_for_new_observation(self) -> None:
        self._episode_prefilled = False

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _warmup(self) -> None:
        """Run optional warmup inference passes during initialization."""
        logger.info("Running warmup inference (%d rounds)...", self.warmup_rounds)
        dummy_image = torch.zeros(
            (1, 3, self.video_height, self.video_width),
            device=self.device,
            dtype=self.model.torch_dtype,
        )
        dummy_proprio = None
        if self.model.proprio_dim is not None:
            dummy_proprio = torch.zeros((1, self.model.proprio_dim), device=self.device, dtype=torch.float32)

        warmup_prompt = self._DEFAULT_PROMPT.format(task="warmup")
        infer_params = inspect.signature(self.model.infer_action).parameters
        optional_kwargs = {}
        if self.num_inference_steps is not None:
            optional_kwargs["num_inference_steps"] = self.num_inference_steps
        if self.sigma_shift is not None:
            optional_kwargs["sigma_shift"] = self.sigma_shift

        with torch.no_grad():
            for round_idx in range(self.warmup_rounds):
                if "phase" in infer_params:
                    self.model.infer_action(
                        prompt=warmup_prompt,
                        input_image=dummy_image,
                        action_horizon=self.action_horizon,
                        negative_prompt=self.negative_prompt,
                        text_cfg_scale=self.text_cfg_scale,
                        seed=0,
                        rand_device=self.rand_device,
                        tiled=self.tiled,
                        phase="video",
                        **optional_kwargs,
                    )
                    self.model.infer_action(
                        prompt=None,
                        input_image=dummy_image,
                        action_horizon=self.action_horizon,
                        chunk_obs_image=dummy_image,
                        chunk_proprio=dummy_proprio,
                        negative_prompt=self.negative_prompt,
                        text_cfg_scale=self.text_cfg_scale,
                        seed=0,
                        rand_device=self.rand_device,
                        tiled=self.tiled,
                        phase="action",
                        **optional_kwargs,
                    )
                    if hasattr(self.model, "_inference_state"):
                        self.model._inference_state = None
                else:
                    self.model.infer_action(
                        prompt=warmup_prompt,
                        input_image=dummy_image,
                        action_horizon=self.action_horizon,
                        proprio=dummy_proprio,
                        negative_prompt=self.negative_prompt,
                        text_cfg_scale=self.text_cfg_scale,
                        seed=0,
                        rand_device=self.rand_device,
                        tiled=self.tiled,
                        **optional_kwargs,
                    )
                logger.info("Warmup round %d/%d complete.", round_idx + 1, self.warmup_rounds)
        self._hard_reset()

    def _load_deploy_config(self, policy_path: str) -> dict[str, Any]:
        path = Path(policy_path)
        if path.suffix in (".yml", ".yaml") and path.is_file():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if path.is_dir():
            for name in ("deploy.yml", "deploy.yaml", "config.yaml"):
                cfg_file = path / name
                if cfg_file.exists():
                    with open(cfg_file, encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
        raise FileNotFoundError(
            f"No AHAWAM deploy config found at '{policy_path}'. "
            "Pass a YAML config file or place 'deploy.yml' in the directory."
        )

