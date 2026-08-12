import logging
import os
import inspect
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ahawam.datasets.lerobot.processors.ahawam_processor import AHAWAMProcessor
from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from ahawam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


def _sync_cuda(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


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


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _preprocess_camera_rgb(
    image: np.ndarray,
    *,
    processor_size_hw: tuple[int, int],
    video_size_hw: tuple[int, int],
) -> np.ndarray:
    """Mirror RobotVideoDataset's per-camera resize and center crop."""
    processor_height, processor_width = processor_size_hw
    video_height, video_width = video_size_hw
    resized = Image.fromarray(image.astype(np.uint8), mode="RGB").resize(
        (processor_width, processor_height),
        resample=Image.BILINEAR,
    )
    scale = max(
        video_width / processor_width,
        video_height / processor_height,
    )
    scaled_width = int(scale * processor_width + 0.5)
    scaled_height = int(scale * processor_height + 0.5)
    resized = resized.resize(
        (scaled_width, scaled_height),
        resample=Image.BICUBIC,
    )
    left = (scaled_width - video_width) // 2
    top = (scaled_height - video_height) // 2
    cropped = resized.crop(
        (left, top, left + video_width, top + video_height)
    )
    return np.asarray(cropped, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        chunks_per_video_prefill: Optional[int],
        num_inference_steps: Optional[int],
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        video_size_hw: tuple[int, int],
    ) -> None:
        self.num_views = int(
            OmegaConf.select(model_cfg, "shared_visual_config.num_views", default=1)
        )
        if self.num_views not in (1, 2, 3):
            raise ValueError(f"Unsupported deployment view count: {self.num_views}")
        video_height, video_width = (int(value) for value in video_size_hw)
        if video_height <= 0 or video_width <= 0:
            raise ValueError(f"Invalid deployment video size: {video_size_hw}")
        self.video_height = video_height
        self.video_width = video_width
        camera_shapes = [
            tuple(int(value) for value in meta["shape"][-2:])
            for meta in processor_cfg.shape_meta.images
        ]
        if not camera_shapes or any(shape != camera_shapes[0] for shape in camera_shapes):
            raise ValueError(
                "Deployment requires identical processor camera sizes, "
                f"got {camera_shapes}."
            )
        self.processor_camera_size_hw = camera_shapes[0]
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: AHAWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.num_inference_steps = 10 if num_inference_steps is None else int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self.num_chunks = int(self.action_horizon // self.model.action_chunk_size)
        if self.action_horizon % int(self.model.action_chunk_size) != 0:
            raise ValueError(
                f"`action_horizon` ({self.action_horizon}) must be divisible by "
                f"action_chunk_size ({int(self.model.action_chunk_size)})."
            )
        self.chunks_per_video_prefill = (
            2
            if chunks_per_video_prefill is None
            else int(chunks_per_video_prefill)
        )
        if self.chunks_per_video_prefill <= 0:
            raise ValueError(
                "`chunks_per_video_prefill` must be positive, "
                f"got {self.chunks_per_video_prefill}"
            )
        if self.chunks_per_video_prefill > self.num_chunks:
            raise ValueError(
                "`chunks_per_video_prefill` cannot exceed chunks in `action_horizon`: "
                f"{self.chunks_per_video_prefill} > {self.num_chunks}"
            )
        self.video_prefill_action_horizon = (
            self.chunks_per_video_prefill * int(self.model.action_chunk_size)
        )
        self._chunks_since_video_prefill = 0

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.step_count = 0
        self._episode_prefilled = False
        self._timing_rollout = {
            "infer_s": 0.0,
            "sim_s": 0.0,
            "infer_calls": 0.0,
            "prefill_s": 0.0,
            "prefill_calls": 0.0,
            "action_chunk_s": 0.0,
            "action_chunk_calls": 0.0,
        }
        for chunk_idx in range(self.num_chunks):
            self._timing_rollout[f"chunk_{chunk_idx + 1}_s"] = 0.0
            self._timing_rollout[f"chunk_{chunk_idx + 1}_calls"] = 0.0

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | "
            "horizon=%d | chunk=%d | num_chunks=%d | chunks_per_video_prefill=%d | "
            "video_prefill_horizon=%d | camera_resize=%s | video_crop=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            int(self.model.action_chunk_size),
            self.num_chunks,
            self.chunks_per_video_prefill,
            self.video_prefill_action_horizon,
            self.processor_camera_size_hw,
            (self.video_height, self.video_width),
        )
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy inference pass so compile overhead is paid during init."""
        if self.num_views > 1:
            dummy_image = torch.zeros(
                (self.num_views, 3, self.video_height, self.video_width),
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
        else:
            dummy_image = torch.zeros(
                (1, 3, self.video_height, self.video_width),
                device=self.model.device,
                dtype=self.model.torch_dtype,
            )
        dummy_proprio = None
        if self.model.proprio_dim is not None:
            dummy_proprio = torch.zeros(
                (1, self.model.proprio_dim),
                device=self.model.device,
                dtype=torch.float32,
            )

        infer_action_params = inspect.signature(self.model.infer_action).parameters
        warmup_prompt = DEFAULT_PROMPT.format(task="warmup")
        with torch.no_grad():
            if "phase" in infer_action_params:
                self.model.infer_action(
                    prompt=warmup_prompt,
                    input_image=dummy_image,
                    action_horizon=self.video_prefill_action_horizon,
                    negative_prompt=self.negative_prompt,
                    text_cfg_scale=self.text_cfg_scale,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=0,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                    phase="video",
                )
                self.model.infer_action(
                    prompt=None,
                    input_image=dummy_image,
                    action_horizon=self.video_prefill_action_horizon,
                    chunk_obs_image=dummy_image,
                    chunk_proprio=dummy_proprio,
                    negative_prompt=self.negative_prompt,
                    text_cfg_scale=self.text_cfg_scale,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=0,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                    phase="action",
                )
                self._reset_chunk_rollout_state()
            else:
                self.model.infer_action(
                    prompt=warmup_prompt,
                    input_image=dummy_image,
                    action_horizon=self.video_prefill_action_horizon,
                    proprio=dummy_proprio,
                    negative_prompt=self.negative_prompt,
                    text_cfg_scale=self.text_cfg_scale,
                    num_inference_steps=self.num_inference_steps,
                    sigma_shift=self.sigma_shift,
                    seed=0,
                    rand_device=self.rand_device,
                    tiled=self.tiled,
                )
        logger.info("Warmup inference finished.")

    def _prefill_episode(
        self,
        *,
        prompt: str,
        image_tensor: torch.Tensor,
    ) -> None:
        video_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.video_prefill_action_horizon,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            "phase": "video",
        }
        prefill_t0 = time.perf_counter() if self.timing_enabled else 0.0
        self.model.infer_action(**video_kwargs)
        if self.timing_enabled:
            self._timing_rollout["prefill_s"] += time.perf_counter() - prefill_t0
            self._timing_rollout["prefill_calls"] += 1.0
        self._episode_prefilled = True
        self._chunks_since_video_prefill = 0

    def _reset_chunk_rollout_state(self) -> None:
        """Hard reset: clears all inference state including history. Used on episode boundary."""
        if hasattr(self.model, "reset_history"):
            self.model.reset_history()
        if hasattr(self.model, "_inference_state"):
            self.model._inference_state = None
        self._episode_prefilled = False
        self._chunks_since_video_prefill = 0

    def _model_num_history_frames(self) -> int:
        getter = getattr(self.model, "_configured_num_history_frames", None)
        if callable(getter):
            return int(getter())
        return int(getattr(self.model, "num_history_frames", 0))

    def _soft_reset_for_new_observation(self) -> None:
        """Reset before a new video prefill while preserving history when configured."""
        if self._model_num_history_frames() <= 0:
            self._reset_chunk_rollout_state()
            return
        self._episode_prefilled = False
        self._chunks_since_video_prefill = 0

    def _infer_action_chunk(
        self,
        *,
        image_tensor: torch.Tensor,
        proprio: torch.Tensor,
    ) -> Dict[str, Any]:
        if not self._episode_prefilled:
            raise RuntimeError("Episode video state is not initialized before phase='action'.")

        action_kwargs = {
            "chunk_obs_image": image_tensor,
            "chunk_proprio": proprio,
            "sigma_shift": self.sigma_shift,
            "tiled": self.tiled,
            "phase": "action",
        }
        if self.num_inference_steps is not None:
            action_kwargs["num_inference_steps"] = self.num_inference_steps
        chunk_t0 = time.perf_counter() if self.timing_enabled else 0.0
        pred = self.model.infer_action(**action_kwargs)
        if self.timing_enabled:
            chunk_dt = time.perf_counter() - chunk_t0
            self._timing_rollout["action_chunk_s"] += chunk_dt
            self._timing_rollout["action_chunk_calls"] += 1.0
            chunk_index = int(pred["chunk_index"])
            if 0 <= chunk_index < self.num_chunks:
                self._timing_rollout[f"chunk_{chunk_index + 1}_s"] += chunk_dt
                self._timing_rollout[f"chunk_{chunk_index + 1}_calls"] += 1.0
        return pred

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

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        started = time.perf_counter()
        obs_data = observation["observation"]
        if self.num_views == 1:
            head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
            left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
            right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
            bottom = np.concatenate([left, right], axis=1)
            image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            head = _preprocess_camera_rgb(
                obs_data["head_camera"]["rgb"],
                processor_size_hw=self.processor_camera_size_hw,
                video_size_hw=(self.video_height, self.video_width),
            )
            left = _preprocess_camera_rgb(
                obs_data["left_camera"]["rgb"],
                processor_size_hw=self.processor_camera_size_hw,
                video_size_hw=(self.video_height, self.video_width),
            )
            right = _preprocess_camera_rgb(
                obs_data["right_camera"]["rgb"],
                processor_size_hw=self.processor_camera_size_hw,
                video_size_hw=(self.video_height, self.video_width),
            )
            views = np.stack(
                [left, right] if self.num_views == 2 else [head, left, right],
                axis=0,
            )
            image_tensor = torch.from_numpy(views).permute(0, 3, 1, 2)
        logger.info(
            "predict stage=image_cpu shape=%s elapsed_s=%.3f",
            tuple(image_tensor.shape),
            time.perf_counter() - started,
        )
        image_tensor = image_tensor.to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _predict_next_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        request_started = time.perf_counter()
        image_tensor = self._build_robotwin_image_tensor(observation)
        _sync_cuda(self.model.device)
        logger.info(
            "predict stage=image_device elapsed_s=%.3f",
            time.perf_counter() - request_started,
        )
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        normalize_started = time.perf_counter()
        proprio = self._normalize_state(state_vector)
        logger.info(
            "predict stage=normalize_state elapsed_s=%.3f",
            time.perf_counter() - normalize_started,
        )

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            if not self._episode_prefilled:
                prefill_started = time.perf_counter()
                logger.info("predict stage=video_prefill start")
                self._prefill_episode(prompt=prompt, image_tensor=image_tensor)
                _sync_cuda(self.model.device)
                logger.info(
                    "predict stage=video_prefill complete elapsed_s=%.3f",
                    time.perf_counter() - prefill_started,
                )
            action_started = time.perf_counter()
            logger.info("predict stage=action_chunk start")
            pred = self._infer_action_chunk(
                image_tensor=image_tensor,
                proprio=proprio,
            )
            _sync_cuda(self.model.device)
            logger.info(
                "predict stage=action_chunk complete elapsed_s=%.3f",
                time.perf_counter() - action_started,
            )
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
            self._timing_rollout["infer_calls"] += 1.0

        denormalize_started = time.perf_counter()
        action_tensor = pred["action_chunk"].unsqueeze(0)
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        logger.info(
            "predict stage=denormalize complete elapsed_s=%.3f total_s=%.3f "
            "finite=%s range=[%.4f,%.4f]",
            time.perf_counter() - denormalize_started,
            time.perf_counter() - request_started,
            bool(np.isfinite(action_chunk).all()),
            float(np.nanmin(action_chunk)),
            float(np.nanmax(action_chunk)),
        )
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        if not hasattr(self.model, "_inference_state") or self.model._inference_state is None:
            next_chunk_index = 0
        else:
            next_chunk_index = int(self.model._inference_state.get("next_chunk_index", 0))
        if (
            next_chunk_index >= self.chunks_per_video_prefill
            or self._chunks_since_video_prefill >= self.chunks_per_video_prefill
        ):
            self._soft_reset_for_new_observation()

        action_chunk = self._predict_next_chunk(observation=observation, instruction=instruction)
        expected_chunk_size = int(self.model.action_chunk_size)
        assert action_chunk.ndim == 2, (
            f"Expected action chunk to be rank-2 [T, D], got shape {tuple(action_chunk.shape)}"
        )
        assert action_chunk.shape[0] == expected_chunk_size, (
            f"Deque fill expects exactly {expected_chunk_size} step actions per chunk, "
            f"got {action_chunk.shape[0]} with shape {tuple(action_chunk.shape)}"
        )
        for action in action_chunk:
            self.pending_actions.append(np.asarray(action, dtype=np.float32))
        self._chunks_since_video_prefill += 1

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def predict_action(self, observation: Dict[str, Any]) -> np.ndarray:
        instruction = observation.get("_instruction")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("Remote action prediction requires `_instruction` in observation.")
        if self.should_request_observation():
            self._fill_action_queue(observation=observation, instruction=instruction)
        if self.should_request_observation():
            raise RuntimeError("No action generated for the current observation.")
        self.step_count += 1
        return self.pending_actions.popleft()

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if observation is None and self.should_request_observation():
            raise ValueError(
                "Observation is required when action queue is empty "
                "(chunk step for ahawam deploy)."
            )
        payload = {} if observation is None else dict(observation)
        payload["_instruction"] = task_env.get_instruction()
        action = self.predict_action(payload)
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0
        self._timing_rollout["infer_calls"] = 0.0
        self._timing_rollout["prefill_s"] = 0.0
        self._timing_rollout["prefill_calls"] = 0.0
        self._timing_rollout["action_chunk_s"] = 0.0
        self._timing_rollout["action_chunk_calls"] = 0.0
        for chunk_idx in range(self.num_chunks):
            self._timing_rollout[f"chunk_{chunk_idx + 1}_s"] = 0.0
            self._timing_rollout[f"chunk_{chunk_idx + 1}_calls"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        timing = {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
            "infer_calls": float(self._timing_rollout["infer_calls"]),
            "prefill_s": float(self._timing_rollout["prefill_s"]),
            "prefill_calls": float(self._timing_rollout["prefill_calls"]),
            "action_chunk_s": float(self._timing_rollout["action_chunk_s"]),
            "action_chunk_calls": float(self._timing_rollout["action_chunk_calls"]),
        }
        for chunk_idx in range(self.num_chunks):
            timing[f"chunk_{chunk_idx + 1}_s"] = float(
                self._timing_rollout[f"chunk_{chunk_idx + 1}_s"]
            )
            timing[f"chunk_{chunk_idx + 1}_calls"] = float(
                self._timing_rollout[f"chunk_{chunk_idx + 1}_calls"]
            )
        return timing

    def reset(self) -> None:
        self.pending_actions.clear()
        self._reset_chunk_rollout_state()
        self.episode_count += 1
        self.step_count = 0
        self.reset_timing_rollout()

    def reset_model(self) -> None:
        self.reset()



def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    chunks_per_video_prefill = _parse_optional_int(usr_args.get("chunks_per_video_prefill"))
    if chunks_per_video_prefill is None:
        chunks_per_video_prefill = _parse_optional_int(
            cfg.EVALUATION.get("chunks_per_video_prefill")
        )
    if chunks_per_video_prefill is None:
        chunks_per_video_prefill = 2

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = _parse_optional_int(cfg.EVALUATION.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = 10

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    video_size = tuple(int(value) for value in cfg.data.train.video_size)
    if len(video_size) != 2:
        raise ValueError(f"`data.train.video_size` must be [H,W], got {video_size}.")

    policy = WorldActionRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        chunks_per_video_prefill=chunks_per_video_prefill,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        video_size_hw=video_size,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    if hasattr(model, "call"):
        if obs is None:
            raise ValueError("Remote policy evaluation requires an observation.")
        payload = dict(obs)
        payload["_instruction"] = TASK_ENV.get_instruction()
        action = model.call(func_name="predict_action", obs=payload)
        TASK_ENV.take_action(np.asarray(action, dtype=np.float32), action_type="qpos")
        return
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
