#!/usr/bin/env python3
"""TCP policy server for AHAWAM real-robot deployment.

Run this on the GPU workstation. It intentionally has no ROS dependency: the
robot-side process sends NumPy/Python observations and this server returns a
14-D dual-arm action chunk.
"""

import argparse
import copy
import importlib
import logging
import multiprocessing as mp
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from deploy.common.tcp_protocol import recv_message, send_message


logger = logging.getLogger("wam_policy_server")

CUDA_FATAL_ERROR_PATTERNS = (
    "cuda error",
    "illegal instruction",
    "device-side assert",
    "unspecified launch failure",
    "an illegal memory access",
    "misaligned address",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for _path in (SRC_ROOT, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))



def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def is_cuda_fatal_error(exc_or_text):
    text = str(exc_or_text).lower()
    return any(pattern in text for pattern in CUDA_FATAL_ERROR_PATTERNS)


def _put_latest(mp_queue, item):
    """Put item into a small multiprocessing queue, dropping older items."""
    while True:
        try:
            mp_queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                mp_queue.get_nowait()
            except queue.Empty:
                return False


def _drain_queue(mp_queue):
    drained = 0
    while True:
        try:
            mp_queue.get_nowait()
            drained += 1
        except queue.Empty:
            return drained


def _tensor_tree_to_cpu(obj):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().to(device="cpu")
    if isinstance(obj, dict):
        return {k: _tensor_tree_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_tree_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tensor_tree_to_cpu(v) for v in obj)
    return obj


def _tensor_tree_to_device(obj, device):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _tensor_tree_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_tensor_tree_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_tensor_tree_to_device(v, device) for v in obj)
    return obj


class WAMPolicyAdapter:
    """Small compatibility layer around the policy implementation.

    Supported policy styles:
      1. control_your_robot style:
         update_observation_window(img_arr, state); get_action()
      2. chunk style:
         predict_action_chunk(observation_dict)
      3. single-action style:
         get_action(observation_dict)
    """

    def __init__(self, args):
        t0 = time.perf_counter()
        logger.info(
            "Loading policy class %s.%s",
            args.policy_module,
            args.policy_class,
        )
        policy_cls = import_class(args.policy_module, args.policy_class)
        self.args = args
        self.instruction = args.instruction

        logger.info("Constructing policy from --policy-path=%s", args.policy_path)
        construct_t0 = time.perf_counter()
        try:
            self.policy = policy_cls(args.policy_path, args.task_name)
        except TypeError:
            self.policy = policy_cls(args.policy_path)
        logger.info(
            "Policy object constructed in %.1f ms",
            (time.perf_counter() - construct_t0) * 1000.0,
        )

        if self._apply_instruction(args.instruction):
            logger.info("Language instruction set on policy object.")
        elif hasattr(self.policy, "random_set_language") and args.random_instruction:
            self.policy.random_set_language()
            logger.info("Language instruction randomized via random_set_language().")
        else:
            logger.info("Policy does not expose a language setter; using request payload only.")

        if hasattr(self.policy, "eval"):
            self.policy.eval()
            logger.info("Policy switched to eval mode.")

        logger.info("Policy loading finished in %.1f ms", (time.perf_counter() - t0) * 1000.0)

    def _apply_instruction(self, instruction):
        if instruction is None:
            return False
        self.instruction = instruction
        if hasattr(self.policy, "set_language_instruction"):
            self.policy.set_language_instruction(instruction)
            return True
        if hasattr(self.policy, "set_language"):
            self.policy.set_language(instruction)
            return True
        return False

    def reset(self, instruction=None):
        t0 = time.perf_counter()
        reset_name = None
        for name in ("reset_obsrvationwindows", "reset_observation_windows", "reset", "reset_obs"):
            fn = getattr(self.policy, name, None)
            if fn is not None:
                fn()
                reset_name = name
                break

        self._apply_instruction(instruction or self.instruction)

        logger.info(
            "Policy reset complete using %s in %.1f ms",
            reset_name or "no reset method",
            (time.perf_counter() - t0) * 1000.0,
        )

    def infer(self, request):
        t0 = time.perf_counter()
        images = request.get("images", {})
        front = images.get("front")
        if front is None:
            raise ValueError("request must contain images['front']")

        state = np.asarray(request["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.args.action_dim:
            raise ValueError(f"state dim must be {self.args.action_dim}, got {state.shape[0]}")

        request_instruction = request.get("instruction", self.instruction)
        self._apply_instruction(request_instruction)

        logger.info(
            "Preparing inference input: front_shape=%s front_dtype=%s state_shape=%s instruction=%r",
            getattr(front, "shape", None),
            getattr(front, "dtype", None),
            state.shape,
            request_instruction,
        )

        # Only img_arr[0] (front/head camera) is used by the model.
        # Agilex deployment uses single head camera (num_output_cameras=1).
        img_arr = [front]

        infer_t0 = time.perf_counter()
        if hasattr(self.policy, "update_observation_window"):
            update_t0 = time.perf_counter()
            self.policy.update_observation_window(img_arr, state)
            logger.info(
                "Observation window updated in %.1f ms",
                (time.perf_counter() - update_t0) * 1000.0,
            )
            action_chunk = self.policy.get_action()
        elif hasattr(self.policy, "predict_action_chunk"):
            observation = {
                "observation.images.front": front,
                "observation.state": state,
                "task": request_instruction,
                "prompt": request_instruction,
            }
            action_chunk = self.policy.predict_action_chunk(observation)
        else:
            observation = {
                "image": front,
                "state": state,
                "instruction": request_instruction,
            }
            action_chunk = self.policy.get_action(observation)
        logger.info(
            "Raw policy inference returned in %.1f ms",
            (time.perf_counter() - infer_t0) * 1000.0,
        )

        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        if action_chunk.shape[1] != self.args.action_dim:
            raise ValueError(
                f"action dim must be {self.args.action_dim}, got shape {action_chunk.shape}"
            )
        logger.info(
            "Inference output ready: action_chunk_shape=%s total_adapter_time=%.1f ms",
            action_chunk.shape,
            (time.perf_counter() - t0) * 1000.0,
        )
        return action_chunk


class PolicyRuntime:
    """Owns the policy object and handles requests in its current process."""

    def __init__(self, args):
        self.args = args
        logger.info("Starting WAM policy runtime with args: %s", vars(args))
        self.adapter = WAMPolicyAdapter(args)
        self.inference_step_count = 0

    def reset(self, instruction=None):
        t0 = time.perf_counter()
        self.adapter.reset(instruction=instruction)
        logger.info("Runtime reset handled in %.1f ms", (time.perf_counter() - t0) * 1000.0)
        return {"ok": True, "type": "reset_ack"}

    def handle_request(self, request):
        request_type = request.get("type")
        logger.info("Handling request type=%r", request_type)
        if request.get("type") == "reset":
            return self.reset(instruction=request.get("instruction"))

        if request.get("type") != "infer":
            logger.warning("Unknown request type received: %r", request.get("type"))
            return {"ok": False, "error": f"unknown request type: {request.get('type')}"}

        self.inference_step_count += 1
        inference_step = self.inference_step_count
        logger.info("Starting policy inference step #%d", inference_step)
        t0 = time.perf_counter()
        try:
            action_chunk = self.adapter.infer(request)
        except Exception as exc:
            logger.exception("Policy inference step #%d failed.", inference_step)
            reply = {"ok": False, "error": repr(exc), "server_inference_step": inference_step}
            if self.args.exit_on_cuda_fatal and is_cuda_fatal_error(exc):
                reply["fatal"] = True
                reply["fatal_reason"] = "cuda_fatal_error"
            return reply

        if self.args.max_return_actions > 0:
            logger.info(
                "Truncating action chunk from %d to max_return_actions=%d",
                action_chunk.shape[0],
                self.args.max_return_actions,
            )
            action_chunk = action_chunk[: self.args.max_return_actions]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Inference step #%d handled successfully in %.1f ms; returning action_chunk_shape=%s",
            inference_step,
            latency_ms,
            action_chunk.shape,
        )
        return {
            "ok": True,
            "type": "action_chunk",
            "action_chunk": action_chunk,
            "model_latency_ms": latency_ms,
            "server_inference_step": inference_step,
        }


class InProcessPolicyBackend:
    def __init__(self, args):
        self.args = args
        self.runtime = PolicyRuntime(args)

    def handle_request(self, request):
        return self.runtime.handle_request(request)

    def reset(self):
        return self.runtime.reset()

    def close(self):
        pass


def policy_worker_main(args, conn):
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Policy worker process started with pid=%s", os.getpid())
    try:
        runtime = PolicyRuntime(args)
        conn.send({"ok": True, "type": "worker_ready", "pid": os.getpid()})
        while True:
            request = conn.recv()
            if request.get("type") == "__shutdown__":
                logger.info("Policy worker received shutdown request.")
                return
            reply = runtime.handle_request(request)
            conn.send(reply)
            if isinstance(reply, dict) and reply.get("fatal") and args.exit_on_cuda_fatal:
                logger.critical("Fatal CUDA error reported inside worker; exiting worker immediately.")
                os._exit(2)
    except EOFError:
        logger.info("Policy worker parent pipe closed; exiting.")
    except Exception as exc:
        logger.exception("Policy worker failed: %r", exc)
        try:
            conn.send({"ok": False, "fatal": True, "fatal_reason": "worker_exception", "error": repr(exc)})
        except Exception:
            pass
        if is_cuda_fatal_error(exc) and args.exit_on_cuda_fatal:
            os._exit(2)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


class WorkerPolicyBackend:
    """Runs policy inference in a child process to isolate CUDA fatal failures."""

    def __init__(self, args):
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.process = None
        self.conn = None
        self.start_worker()

    def start_worker(self):
        self.close_worker(force=True)
        parent_conn, child_conn = self.ctx.Pipe()
        self.process = self.ctx.Process(
            target=policy_worker_main,
            args=(self.args, child_conn),
            daemon=True,
        )
        self.process.start()
        child_conn.close()
        self.conn = parent_conn
        logger.info("Started policy worker pid=%s", self.process.pid)

        if not self.conn.poll(self.args.worker_startup_timeout):
            exitcode = self.process.exitcode
            self.close_worker(force=True)
            raise TimeoutError(
                f"policy worker did not become ready within {self.args.worker_startup_timeout}s "
                f"(exitcode={exitcode})"
            )
        ready = self.conn.recv()
        if not ready.get("ok"):
            self.close_worker(force=True)
            raise RuntimeError(f"policy worker failed to initialize: {ready}")
        logger.info("Policy worker ready: %s", ready)

    def close_worker(self, force=False):
        if self.conn is not None:
            if not force:
                try:
                    self.conn.send({"type": "__shutdown__"})
                except Exception:
                    pass
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

        if self.process is not None:
            if self.process.is_alive():
                if not force:
                    self.process.join(timeout=2.0)
                if self.process.is_alive():
                    logger.warning("Terminating policy worker pid=%s", self.process.pid)
                    self.process.terminate()
                    self.process.join(timeout=5.0)
                if self.process.is_alive():
                    logger.warning("Killing policy worker pid=%s", self.process.pid)
                    self.process.kill()
                    self.process.join(timeout=5.0)
            self.process = None

    def _ensure_worker(self):
        if self.process is None or not self.process.is_alive() or self.conn is None:
            logger.warning("Policy worker is not alive; starting a new worker.")
            self.start_worker()

    def handle_request(self, request):
        self._ensure_worker()
        try:
            self.conn.send(request)
            if not self.conn.poll(self.args.worker_request_timeout):
                exitcode = self.process.exitcode if self.process is not None else None
                logger.error(
                    "Policy worker timed out after %.1f s while handling request type=%r",
                    self.args.worker_request_timeout,
                    request.get("type"),
                )
                self.close_worker(force=True)
                return {
                    "ok": False,
                    "fatal": True,
                    "fatal_reason": "policy_worker_timeout",
                    "error": f"policy worker timed out after {self.args.worker_request_timeout}s",
                    "worker_exitcode": exitcode,
                }
            reply = self.conn.recv()
        except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
            exitcode = self.process.exitcode if self.process is not None else None
            logger.exception("Policy worker died while handling request: exitcode=%s", exitcode)
            self.close_worker(force=True)
            return {
                "ok": False,
                "fatal": True,
                "fatal_reason": "policy_worker_died",
                "error": repr(exc),
                "worker_exitcode": exitcode,
            }

        if isinstance(reply, dict) and reply.get("fatal"):
            logger.critical("Policy worker reported fatal error: %s", reply)
            self.close_worker(force=True)
        return reply

    def reset(self):
        return self.handle_request({"type": "reset"})

    def close(self):
        self.close_worker(force=False)


class AsyncPolicyBackend:
    """Backend for async decoupled video/action inference.

    Uses AsyncAHAWAMRuntime with a single GPU thread.
    Supports new message types: "image", "action_request", "config".
    Falls back to synchronous "infer" for backward compatibility.
    """

    def __init__(self, args):
        self.args = args
        # Load policy in-process (async needs direct model access)
        self.adapter = WAMPolicyAdapter(args)
        self.policy = self.adapter.policy

        # Load async timing config if available
        import yaml as _yaml
        async_config = {}
        timing_config_path = getattr(args, "async_timing_config", None)
        if timing_config_path:
            with open(timing_config_path) as f:
                timing = _yaml.safe_load(f)
            async_config.update(timing)
            logger.info("Loaded async timing config from %s: %s", timing_config_path, timing)

        async_config.setdefault("action_priority", True)
        async_config.setdefault("max_video_kv_staleness_ms", 500.0)

        from deploy.server.async_runtime import AsyncAHAWAMRuntime
        self.runtime = AsyncAHAWAMRuntime(self.policy, async_config)
        self._async_started = False
        logger.info("AsyncPolicyBackend initialized.")

    def handle_request(self, request):
        """Dispatch based on message type."""
        msg_type = request.get("type")

        if msg_type == "reset":
            return self._handle_reset(request)
        elif msg_type == "image":
            return self._handle_image(request)
        elif msg_type == "action_request":
            return self._handle_action_request(request)
        elif msg_type == "config":
            return self._handle_config(request)
        elif msg_type == "infer":
            # Sync fallback: use the standard adapter path
            return self._handle_sync_infer(request)
        else:
            return {"ok": False, "error": f"unknown request type: {msg_type}"}

    def _handle_reset(self, request):
        """Reset for new episode."""
        instruction = request.get("instruction")
        if instruction:
            self.adapter._apply_instruction(instruction)

        # Reset runtime (clears state, history, queues)
        prompt = instruction or self.adapter.instruction
        self.runtime.reset(prompt=prompt)

        # Also reset the adapter state
        self.adapter.reset(instruction=instruction)

        if self._async_started:
            # Restart GPU loop with new prompt
            self.runtime.stop()
            self.runtime.start(prompt=prompt)

        logger.info("AsyncPolicyBackend reset (instruction=%r)", instruction)
        return {"ok": True, "type": "reset_ack"}

    def _handle_image(self, request):
        """Push image for async video prefill."""
        images = request.get("images", {})
        front = images.get("front")
        if front is None:
            return {"ok": False, "error": "image request must contain images['front']"}

        # Start runtime on first image if not started
        if not self._async_started:
            prompt = self.adapter.instruction or "default task"
            self.runtime.start(prompt=prompt)
            self._async_started = True

        # Preprocess image same way as adapter
        front = np.asarray(front, dtype=np.uint8)
        img_arr = [front]
        self.policy.update_observation_window(img_arr, np.zeros(self.args.action_dim, dtype=np.float32))
        image_tensor = self.policy._observation["image_tensor"]

        queued = self.runtime.push_image(image_tensor)
        return {"ok": True, "type": "image_ack", "queued": queued}

    def _handle_action_request(self, request):
        """Request an action chunk using latest video inference state."""
        images = request.get("images", {})
        front = images.get("front")
        state = request.get("state")

        if front is None or state is None:
            return {"ok": False, "error": "action_request must contain images['front'] and state"}

        if not self._async_started:
            return {"ok": False, "error": "async runtime not started; send an image first"}

        # Wait for first prefill
        if not self.runtime.first_prefill_done.wait(timeout=5.0):
            return {"ok": False, "error": "timeout waiting for first video prefill"}

        # Preprocess
        front = np.asarray(front, dtype=np.uint8)
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        img_arr = [front]
        self.policy.update_observation_window(img_arr, state)
        image_tensor = self.policy._observation["image_tensor"]
        proprio = self.policy._normalize_state(state)

        # Submit action request and wait for result
        self.runtime.request_action(image_tensor, proprio)
        result = self.runtime.get_action_result(timeout=5.0)

        if result is None:
            return {"ok": False, "error": "timeout waiting for action chunk"}

        action_chunk = result.action_chunk
        if self.args.max_return_actions > 0:
            action_chunk = action_chunk[:self.args.max_return_actions]

        return {
            "ok": True,
            "type": "action_chunk",
            "action_chunk": action_chunk,
            "model_latency_ms": result.latency_ms,
            "kv_version": result.kv_version,
        }

    def _handle_config(self, request):
        """Handle runtime config updates."""
        logger.info("Config update: %s", request)
        return {"ok": True, "type": "config_ack"}

    def _handle_sync_infer(self, request):
        """Fallback: synchronous inference via standard adapter."""
        t0 = time.perf_counter()
        try:
            action_chunk = self.adapter.infer(request)
        except Exception as exc:
            return {"ok": False, "error": repr(exc)}

        if self.args.max_return_actions > 0:
            action_chunk = action_chunk[:self.args.max_return_actions]

        return {
            "ok": True,
            "type": "action_chunk",
            "action_chunk": action_chunk,
            "model_latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def reset(self):
        return self._handle_reset({"type": "reset"})

    def close(self):
        self.runtime.stop()


def _dual_gpu_worker_args(args, gpu_id):
    worker_args = argparse.Namespace(**vars(args))
    worker_args.policy_worker_process = False
    worker_args.async_mode = False
    worker_args.dual_gpu_async = False
    worker_args.device_gpu = gpu_id
    return worker_args


def _bind_worker_to_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["AHAWAM_DEPLOY_DEVICE_OVERRIDE"] = "cuda"


def dual_video_worker_main(args, request_queue, kv_queue, ready_queue):
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _bind_worker_to_gpu(args.video_gpu)
    logger.info("Dual-GPU video worker pid=%s bound to CUDA_VISIBLE_DEVICES=%s", os.getpid(), args.video_gpu)
    try:
        adapter = WAMPolicyAdapter(_dual_gpu_worker_args(args, args.video_gpu))
        policy = adapter.policy
        version = 0
        generation = 0

        # Warmup: run N prefill rounds to trigger model warmup / model warmup paths compilation.
        # Split into two phases to cover both execution paths:
        #   Phase 1 (cold-start): _hard_reset() before each prefill to cover the
        #       "fresh episode" path (model._inference_state == None).
        #   Phase 2 (streaming): sequential prefills WITHOUT reset to cover the
        #       "incremental history state update" path used during real operation.
        # This avoids paying one-time setup costs during the first live request.
        if args.worker_warmup_rounds > 0:
            logger.info("Video worker starting warmup (%d rounds)...", args.worker_warmup_rounds)
            dummy_img = np.zeros((policy.video_height, policy.video_width, 3), dtype=np.uint8)
            dummy_state = np.zeros(args.action_dim, dtype=np.float32)
            policy.set_language_instruction(args.instruction or "warmup")

            # Phase 1: Cold-start warmup (first frame after reset)
            cold_rounds = min(10, args.worker_warmup_rounds // 3)
            for wi in range(cold_rounds):
                policy._hard_reset()
                policy.update_observation_window(
                    [dummy_img], dummy_state,
                )
                image_tensor = policy._observation["image_tensor"]
                policy.prefill_video_only(image_tensor, prompt=None, deep_copy=False)
                if wi == 0 or (wi + 1) % 10 == 0:
                    logger.info("Video worker warmup (cold-start) %d/%d", wi + 1, cold_rounds)
            logger.info("Video worker cold-start warmup done (%d rounds).", cold_rounds)

            # Phase 2: Streaming warmup (sequential prefills in mini-episodes)
            # Each mini-episode: reset once, then stream N frames without reset.
            # This exercises the streaming path while avoiding temporal RoPE overflow.
            streaming_rounds = args.worker_warmup_rounds - cold_rounds
            frames_per_episode = 10  # reset every N frames to stay within RoPE limits
            done = 0
            while done < streaming_rounds:
                policy._hard_reset()
                episode_len = min(frames_per_episode, streaming_rounds - done)
                for _ in range(episode_len):
                    policy.update_observation_window(
                        [dummy_img], dummy_state,
                    )
                    image_tensor = policy._observation["image_tensor"]
                    policy.prefill_video_only(image_tensor, prompt=None, deep_copy=False)
                    done += 1
                if done <= frames_per_episode or done % 25 == 0 or done >= streaming_rounds:
                    logger.info("Video worker warmup (streaming) %d/%d", done, streaming_rounds)
            logger.info("Video worker streaming warmup done (%d rounds, %d frames/episode).",
                        streaming_rounds, frames_per_episode)

            policy._hard_reset()
            logger.info("Video worker warmup complete.")

        ready_queue.put({"ok": True, "type": "video_worker_ready", "pid": os.getpid(), "gpu": args.video_gpu})
        while True:
            request = request_queue.get()
            msg_type = request.get("type")
            if msg_type == "__shutdown__":
                return
            if msg_type == "reset":
                adapter.reset(instruction=request.get("instruction"))
                generation = int(request.get("generation", generation + 1))
                version = 0
                continue
            if msg_type != "image":
                logger.warning("Video worker ignored message type=%r", msg_type)
                continue
            if int(request.get("generation", generation)) != generation:
                logger.info(
                    "Video worker dropped stale image generation=%s current=%s",
                    request.get("generation"),
                    generation,
                )
                continue

            images = request.get("images", {})
            front = images.get("front")
            if front is None:
                logger.warning("Video worker image request missing images['front']")
                continue

            t0 = time.perf_counter()
            try:
                front = np.asarray(front, dtype=np.uint8)
                policy.update_observation_window(
                    [front],
                    np.zeros(args.action_dim, dtype=np.float32),
                )
                image_tensor = policy._observation["image_tensor"]
                state = policy.prefill_video_only(image_tensor, prompt=None, deep_copy=False)
                cpu_state = _tensor_tree_to_cpu(state)
                t_cpu = time.perf_counter()
                version += 1
                total_ms = (t_cpu - t0) * 1000.0
                payload = {
                    "type": "kv_update",
                    "ok": True,
                    "generation": generation,
                    "version": version,
                    "timestamp": time.perf_counter(),
                    "source_timestamp": request.get("timestamp"),
                    "state": cpu_state,
                    "video_prefill_ms": total_ms,
                }
                _put_latest(kv_queue, payload)
                logger.info(
                    "Video state published version=%d total=%.1fms",
                    version, total_ms,
                )
            except Exception as exc:
                logger.exception("Video worker failed during prefill")
                if args.exit_on_cuda_fatal and is_cuda_fatal_error(exc):
                    ready_queue.put({
                        "ok": False,
                        "fatal": True,
                        "fatal_reason": "video_worker_cuda_fatal",
                        "error": repr(exc),
                    })
                    os._exit(2)
    except Exception as exc:
        logger.exception("Dual-GPU video worker failed")
        ready_queue.put({"ok": False, "fatal": True, "fatal_reason": "video_worker_exception", "error": repr(exc)})
        raise


def dual_action_worker_main(args, request_queue, response_queue, kv_queue, ready_queue):
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _bind_worker_to_gpu(args.action_gpu)
    logger.info("Dual-GPU action worker pid=%s bound to CUDA_VISIBLE_DEVICES=%s", os.getpid(), args.action_gpu)
    try:
        adapter = WAMPolicyAdapter(_dual_gpu_worker_args(args, args.action_gpu))
        policy = adapter.policy
        latest_state = None
        latest_version = 0
        latest_timestamp = 0.0
        latest_video_ms = -1.0
        generation = 0

        # Warmup: run N action chunk rounds to trigger model warmup / model warmup paths compilation
        if args.worker_warmup_rounds > 0:
            logger.info("Action worker starting warmup (%d rounds)...", args.worker_warmup_rounds)
            dummy_img = np.zeros((policy.video_height, policy.video_width, 3), dtype=np.uint8)
            dummy_state = np.zeros(args.action_dim, dtype=np.float32)
            policy.set_language_instruction(args.instruction or "warmup")
            # First do a prefill to get a valid state state for action warmup
            policy._hard_reset()
            policy.update_observation_window(
                [dummy_img], dummy_state,
            )
            image_tensor = policy._observation["image_tensor"]
            kv_state = policy.prefill_video_only(image_tensor, prompt=None, deep_copy=True)
            proprio = policy._normalize_state(dummy_state)
            for wi in range(args.worker_warmup_rounds):
                policy.infer_action_chunk_only(
                    copy.deepcopy(kv_state), image_tensor, proprio,
                )
                if wi == 0 or (wi + 1) % 25 == 0:
                    logger.info("Action worker warmup %d/%d", wi + 1, args.worker_warmup_rounds)
            policy._hard_reset()
            logger.info("Action worker warmup complete.")

        ready_queue.put({"ok": True, "type": "action_worker_ready", "pid": os.getpid(), "gpu": args.action_gpu})

        def drain_kv_updates():
            nonlocal latest_state, latest_version, latest_timestamp, latest_video_ms
            drained = 0
            while True:
                try:
                    update = kv_queue.get_nowait()
                except queue.Empty:
                    break
                if not update.get("ok"):
                    continue
                if int(update.get("generation", generation)) != generation:
                    logger.info(
                        "Action worker dropped stale state generation=%s current=%s",
                        update.get("generation"),
                        generation,
                    )
                    continue
                latest_state = _tensor_tree_to_device(update["state"], policy.device)
                latest_version = int(update["version"])
                latest_timestamp = float(update["timestamp"])
                latest_video_ms = float(update.get("video_prefill_ms", -1.0))
                drained += 1
            if drained:
                logger.info("Action worker loaded latest state version=%d", latest_version)

        while True:
            drain_kv_updates()
            try:
                request = request_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            msg_type = request.get("type")
            if msg_type == "__shutdown__":
                return
            if msg_type == "reset":
                adapter.reset(instruction=request.get("instruction"))
                generation = int(request.get("generation", generation + 1))
                latest_state = None
                latest_version = 0
                latest_timestamp = 0.0
                latest_video_ms = -1.0
                continue
            if msg_type != "action_request":
                logger.warning("Action worker ignored message type=%r", msg_type)
                continue
            if int(request.get("generation", generation)) != generation:
                response_queue.put({
                    "ok": False,
                    "type": "action_error",
                    "request_id": request.get("request_id"),
                    "error": "stale action request generation",
                })
                continue

            request_id = request.get("request_id")
            t0 = time.perf_counter()
            drain_kv_updates()
            if latest_state is None:
                deadline = time.perf_counter() + max(0.0, args.first_kv_timeout_s)
                while latest_state is None and time.perf_counter() < deadline:
                    try:
                        update = kv_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    if update.get("ok") and int(update.get("generation", generation)) == generation:
                        latest_state = _tensor_tree_to_device(update["state"], policy.device)
                        latest_version = int(update["version"])
                        latest_timestamp = float(update["timestamp"])
                        latest_video_ms = float(update.get("video_prefill_ms", -1.0))
                if latest_state is None:
                    response_queue.put({
                        "ok": False,
                        "type": "action_error",
                        "request_id": request_id,
                        "error": "timeout waiting for first video state",
                    })
                    continue

            # Reject action if inference state is too stale
            if args.kv_max_age_ms > 0 and latest_timestamp > 0.0:
                kv_age_ms = (time.perf_counter() - latest_timestamp) * 1000.0
                if kv_age_ms > args.kv_max_age_ms:
                    logger.warning(
                        "inference state too stale (%.0f ms > %.0f ms). Rejecting action request_id=%s.",
                        kv_age_ms, args.kv_max_age_ms, request_id,
                    )
                    response_queue.put({
                        "ok": False,
                        "type": "action_error",
                        "request_id": request_id,
                        "error": f"inference state too stale ({kv_age_ms:.0f} ms > {args.kv_max_age_ms:.0f} ms)",
                        "kv_age_ms": kv_age_ms,
                    })
                    continue

            images = request.get("images", {})
            front = images.get("front")
            state = request.get("state")
            if front is None or state is None:
                response_queue.put({
                    "ok": False,
                    "type": "action_error",
                    "request_id": request_id,
                    "error": "action_request must contain images['front'] and state",
                })
                continue

            try:
                front = np.asarray(front, dtype=np.uint8)
                state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
                if state_vec.shape[0] != args.action_dim:
                    raise ValueError(f"state dim must be {args.action_dim}, got {state_vec.shape[0]}")
                policy.update_observation_window([front], state_vec)
                image_tensor = policy._observation["image_tensor"]
                proprio = policy._normalize_state(state_vec)
                # Deep copy state state to prevent in-place tensor mutation
                # from corrupting latest_state for subsequent action requests
                action_state = copy.deepcopy(latest_state)
                action_state.pop("_chunk_video_kv_cache", None)
                action_chunk = policy.infer_action_chunk_only(
                    inference_state=action_state,
                    image_tensor=image_tensor,
                    proprio=proprio,
                )
                action_chunk = np.asarray(action_chunk, dtype=np.float32)
                if args.max_return_actions > 0:
                    action_chunk = action_chunk[: args.max_return_actions]
                latency_ms = (time.perf_counter() - t0) * 1000.0
                kv_age_ms = (
                    (time.perf_counter() - latest_timestamp) * 1000.0
                    if latest_timestamp > 0.0 else float("inf")
                )
                response_queue.put({
                    "ok": True,
                    "type": "action_chunk",
                    "request_id": request_id,
                    "action_chunk": action_chunk,
                    "model_latency_ms": latency_ms,
                    "kv_version": latest_version,
                    "kv_age_ms": kv_age_ms,
                    "video_prefill_ms": latest_video_ms,
                })
                logger.info(
                    "Action chunk request_id=%s done latency=%.1fms kv_version=%d kv_age=%.1fms",
                    request_id,
                    latency_ms,
                    latest_version,
                    kv_age_ms,
                )
            except Exception as exc:
                logger.exception("Action worker failed during action inference")
                reply = {
                    "ok": False,
                    "type": "action_error",
                    "request_id": request_id,
                    "error": repr(exc),
                }
                if args.exit_on_cuda_fatal and is_cuda_fatal_error(exc):
                    reply["fatal"] = True
                    reply["fatal_reason"] = "action_worker_cuda_fatal"
                    response_queue.put(reply)
                    os._exit(2)
                response_queue.put(reply)
    except Exception as exc:
        logger.exception("Dual-GPU action worker failed")
        ready_queue.put({"ok": False, "fatal": True, "fatal_reason": "action_worker_exception", "error": repr(exc)})
        raise


class DualGPUAsyncPolicyBackend:
    """Two-process dual-GPU backend.

    The video process owns GPU0 and continuously publishes latest video state.
    The action process owns GPU1 and serves latency-sensitive action requests.
    """

    def __init__(self, args):
        self.args = args
        self.ctx = mp.get_context("spawn")
        self.video_queue = self.ctx.Queue(maxsize=max(1, args.image_queue_size))
        self.kv_queue = self.ctx.Queue(maxsize=1)
        self.action_queue = self.ctx.Queue(maxsize=1)
        self.response_queue = self.ctx.Queue(maxsize=4)
        self.ready_queue = self.ctx.Queue(maxsize=4)
        self.request_lock = threading.Lock()
        self.action_request_lock = threading.Lock()
        self.video_queue_lock = threading.Lock()  # guards video_queue put/drain atomicity
        self.request_id = 0
        self.generation = 0
        self.response_buffer = {}
        self.processes = []
        self._start_workers()

    def _start_workers(self):
        video = self.ctx.Process(
            target=dual_video_worker_main,
            args=(self.args, self.video_queue, self.kv_queue, self.ready_queue),
            daemon=True,
        )
        action = self.ctx.Process(
            target=dual_action_worker_main,
            args=(self.args, self.action_queue, self.response_queue, self.kv_queue, self.ready_queue),
            daemon=True,
        )
        video.start()
        action.start()
        self.processes = [video, action]
        logger.info("Started dual-GPU workers: video pid=%s action pid=%s", video.pid, action.pid)

        ready = []
        deadline = time.perf_counter() + self.args.worker_startup_timeout
        while len(ready) < 2 and time.perf_counter() < deadline:
            try:
                msg = self.ready_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if not msg.get("ok"):
                self.close()
                raise RuntimeError(f"dual-GPU worker failed to initialize: {msg}")
            ready.append(msg)
        if len(ready) < 2:
            self.close()
            raise TimeoutError(f"dual-GPU workers did not become ready: ready={ready}")
        logger.info("Dual-GPU workers ready: %s", ready)

    def reset(self):
        return self.handle_request({"type": "reset", "instruction": self.args.instruction})

    def _next_request_id(self):
        with self.request_lock:
            self.request_id += 1
            return self.request_id

    def _workers_alive(self):
        dead = [proc for proc in self.processes if not proc.is_alive()]
        if not dead:
            return True, None
        details = [
            {"pid": proc.pid, "exitcode": proc.exitcode}
            for proc in dead
        ]
        return False, details

    def _fatal_worker_reply(self, details):
        return {
            "ok": False,
            "fatal": True,
            "fatal_reason": "dual_gpu_worker_died",
            "error": f"dual-GPU worker died: {details}",
            "workers": details,
        }

    def _wait_action_response(self, request_id):
        if request_id in self.response_buffer:
            return self.response_buffer.pop(request_id)
        deadline = time.perf_counter() + self.args.action_timeout_s
        while time.perf_counter() < deadline:
            alive, details = self._workers_alive()
            if not alive:
                return self._fatal_worker_reply(details)
            timeout = max(0.0, min(0.1, deadline - time.perf_counter()))
            try:
                reply = self.response_queue.get(timeout=timeout)
            except queue.Empty:
                continue
            rid = reply.get("request_id")
            if rid == request_id:
                return reply
            self.response_buffer[rid] = reply
        return {
            "ok": False,
            "type": "action_error",
            "request_id": request_id,
            "error": f"timeout waiting for action worker after {self.args.action_timeout_s}s",
        }

    def handle_request(self, request):
        alive, details = self._workers_alive()
        if not alive:
            return self._fatal_worker_reply(details)

        msg_type = request.get("type")
        if msg_type == "reset":
            with self.action_request_lock:
                with self.request_lock:
                    self.generation += 1
                    generation = self.generation
                with self.video_queue_lock:
                    _drain_queue(self.video_queue)
                    _drain_queue(self.action_queue)
                    _drain_queue(self.kv_queue)
                    _drain_queue(self.response_queue)
                    self.response_buffer.clear()
                    reset_msg = {
                        "type": "reset",
                        "instruction": request.get("instruction"),
                        "generation": generation,
                    }
                    _put_latest(self.video_queue, reset_msg)
                    _put_latest(self.action_queue, reset_msg)
                return {"ok": True, "type": "reset_ack"}
        if msg_type == "image":
            request = dict(request)
            with self.request_lock:
                request["generation"] = self.generation
            with self.video_queue_lock:
                queued = _put_latest(self.video_queue, request)
            return {"ok": True, "type": "image_ack", "queued": queued}
        if msg_type == "action_request":
            if not self.action_request_lock.acquire(blocking=False):
                return {
                    "ok": False,
                    "type": "action_error",
                    "error": "another action request is already pending",
                }
            try:
                alive, details = self._workers_alive()
                if not alive:
                    return self._fatal_worker_reply(details)
                request = dict(request)
                request["request_id"] = self._next_request_id()
                with self.request_lock:
                    request["generation"] = self.generation
                queued = _put_latest(self.action_queue, request)
                if not queued:
                    return {"ok": False, "type": "action_error", "error": "failed to queue action request"}
                return self._wait_action_response(request["request_id"])
            finally:
                self.action_request_lock.release()
        if msg_type == "infer":
            # Compatibility path: feed the image to video worker, then request action.
            image_request = dict(request)
            image_request["type"] = "image"
            with self.video_queue_lock:
                _put_latest(self.video_queue, image_request)
            action_request = dict(request)
            action_request["type"] = "action_request"
            return self.handle_request(action_request)
        return {"ok": False, "error": f"unknown request type: {msg_type}"}

    def close(self):
        for q in (self.video_queue, self.action_queue):
            try:
                q.put_nowait({"type": "__shutdown__"})
            except Exception:
                pass
        for proc in self.processes:
            if proc.is_alive():
                proc.join(timeout=2.0)
            if proc.is_alive():
                logger.warning("Terminating dual-GPU worker pid=%s", proc.pid)
                proc.terminate()
                proc.join(timeout=5.0)
            if proc.is_alive():
                logger.warning("Killing dual-GPU worker pid=%s", proc.pid)
                proc.kill()
                proc.join(timeout=5.0)


class Server:
    def __init__(self, args):
        self.args = args
        logger.info("Starting WAM policy server with args: %s", vars(args))
        if getattr(args, "dual_gpu_async", False):
            logger.info(
                "Using DUAL-GPU async backend: video_gpu=%s action_gpu=%s",
                args.video_gpu,
                args.action_gpu,
            )
            self.backend = DualGPUAsyncPolicyBackend(args)
        elif getattr(args, "async_mode", False):
            logger.info("Using ASYNC decoupled video/action backend.")
            self.backend = AsyncPolicyBackend(args)
        elif args.policy_worker_process:
            logger.info("Using isolated policy worker process for CUDA inference.")
            self.backend = WorkerPolicyBackend(args)
        else:
            logger.warning("Policy worker process disabled; CUDA runs in the TCP server process.")
            self.backend = InProcessPolicyBackend(args)
        self.request_count = 0
        self.request_count_lock = threading.Lock()

    def handle_request(self, request):
        return self.backend.handle_request(request)

    def _handle_connection(self, conn, addr):
        logger.info("Client connected: %s", addr)
        if not getattr(self.args, "dual_gpu_async", False):
            reset_reply = self.backend.reset()
            if not reset_reply.get("ok", False):
                logger.warning("Policy reset on client connection failed: %s", reset_reply)

        # For dual-GPU async mode, use a reader thread so that image pushes
        # are never blocked by action_request processing.
        if getattr(self.args, "dual_gpu_async", False):
            self._handle_connection_dual_gpu(conn, addr)
            return

        with conn:
            while True:
                try:
                    logger.info("Waiting for client request from %s...", addr)
                    recv_t0 = time.perf_counter()
                    request, request_bytes = recv_message(conn)
                    recv_ms = (time.perf_counter() - recv_t0) * 1000.0
                    with self.request_count_lock:
                        self.request_count += 1
                        request_count = self.request_count
                    logger.info(
                        "Received request #%d from %s: type=%r payload=%d bytes recv+deserialize=%.1f ms",
                        request_count,
                        addr,
                        request.get("type") if isinstance(request, dict) else type(request).__name__,
                        request_bytes,
                        recv_ms,
                    )
                    handle_t0 = time.perf_counter()
                    reply = self.handle_request(request)
                    handle_ms = (time.perf_counter() - handle_t0) * 1000.0
                    send_t0 = time.perf_counter()
                    reply_bytes = send_message(conn, reply)
                    send_ms = (time.perf_counter() - send_t0) * 1000.0
                    logger.info(
                        "Sent reply for request #%d: ok=%s type=%r payload=%d bytes handle=%.1f ms send=%.1f ms",
                        request_count,
                        reply.get("ok") if isinstance(reply, dict) else None,
                        reply.get("type") if isinstance(reply, dict) else type(reply).__name__,
                        reply_bytes,
                        handle_ms,
                        send_ms,
                    )
                    if isinstance(reply, dict) and reply.get("fatal"):
                        logger.critical("Fatal policy error was reported for request #%d: %s", request_count, reply)
                        break
                except ConnectionError:
                    logger.info("Client disconnected: %s", addr)
                    break
                except Exception as exc:
                    logger.exception("Connection error from %s: %r", addr, exc)
                    break

    def _handle_connection_dual_gpu(self, conn, addr):
        """Handle connection for dual-GPU async mode.

        Uses a reader thread to continuously read messages from socket.
        Image messages are forwarded to the video worker immediately without
        blocking on action request processing. This ensures the video worker
        is never starved of frames during action inference.
        """
        import queue as thread_queue

        send_lock = threading.Lock()
        pending_queue = thread_queue.Queue(maxsize=64)
        stop_event = threading.Event()

        def _send_reply(reply):
            with send_lock:
                send_message(conn, reply)

        def _reader_thread():
            """Read messages from socket, dispatch images immediately."""
            try:
                while not stop_event.is_set():
                    try:
                        request, request_bytes = recv_message(conn)
                    except ConnectionError:
                        pending_queue.put(None)  # Signal disconnect
                        return
                    except Exception:
                        pending_queue.put(None)
                        return

                    msg_type = request.get("type") if isinstance(request, dict) else None

                    # Fast path: image messages are handled directly in reader thread
                    # to avoid being blocked by action_request processing.
                    if msg_type == "image":
                        reply = self.backend.handle_request(request)
                        _send_reply(reply)
                        continue

                    # All other messages go to pending_queue for main thread
                    pending_queue.put((request, request_bytes))
            except Exception:
                pending_queue.put(None)

        reader = threading.Thread(target=_reader_thread, daemon=True)
        reader.start()

        with conn:
            try:
                while True:
                    item = pending_queue.get()
                    if item is None:
                        # Client disconnected or reader error
                        logger.info("Client disconnected: %s", addr)
                        break

                    request, request_bytes = item
                    msg_type = request.get("type") if isinstance(request, dict) else None
                    with self.request_count_lock:
                        self.request_count += 1
                        request_count = self.request_count
                    logger.info(
                        "Processing request #%d from %s: type=%r payload=%d bytes",
                        request_count, addr, msg_type, request_bytes,
                    )

                    handle_t0 = time.perf_counter()
                    reply = self.backend.handle_request(request)
                    handle_ms = (time.perf_counter() - handle_t0) * 1000.0
                    _send_reply(reply)

                    logger.info(
                        "Sent reply for request #%d: ok=%s type=%r handle=%.1f ms",
                        request_count,
                        reply.get("ok") if isinstance(reply, dict) else None,
                        reply.get("type") if isinstance(reply, dict) else type(reply).__name__,
                        handle_ms,
                    )

                    if isinstance(reply, dict) and reply.get("fatal"):
                        logger.critical("Fatal error for request #%d: %s", request_count, reply)
                        break
            finally:
                stop_event.set()
                reader.join(timeout=2.0)

    def serve_forever(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.info("Binding TCP server to %s:%s", self.args.host, self.args.port)
        server_socket.bind((self.args.host, self.args.port))
        server_socket.listen(self.args.listen_backlog)
        logger.info("Listening on %s:%s", self.args.host, self.args.port)

        try:
            while True:
                logger.info("Waiting for robot client connection...")
                conn, addr = server_socket.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.args.threaded_connections:
                    thread = threading.Thread(
                        target=self._handle_connection,
                        args=(conn, addr),
                        daemon=True,
                    )
                    thread.start()
                else:
                    self._handle_connection(conn, addr)
        finally:
            logger.info("Closing server socket.")
            server_socket.close()
            self.backend.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--listen-backlog", type=int, default=8)
    parser.add_argument(
        "--threaded-connections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Handle multiple TCP client connections concurrently.",
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task-name", default="wam")
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--policy-module",
        default="deploy.server.ahawam_policy",
        help="Python module containing the WAM policy class.",
    )
    parser.add_argument("--policy-class", default="AHAWAMPolicy")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--max-return-actions",
        type=int,
        default=0,
        help="Optional cap on returned action chunk length; 0 means no cap.",
    )
    parser.add_argument(
        "--random-instruction",
        action="store_true",
        help="Use the policy's random_set_language() instead of the fixed instruction when available.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Server logging verbosity.",
    )
    parser.add_argument(
        "--exit-on-cuda-fatal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat CUDA illegal instruction/device assert/launch failures as fatal. "
            "In worker mode the worker exits; in in-process mode the server exits."
        ),
    )
    parser.add_argument(
        "--policy-worker-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the policy in an isolated child process. This keeps fatal CUDA/flash-attn "
            "failures out of the TCP server process and lets the server stop/restart the worker."
        ),
    )
    parser.add_argument(
        "--worker-startup-timeout",
        type=float,
        default=4800.0,
        help="Seconds to wait for the policy worker to load the model and report ready.",
    )
    parser.add_argument(
        "--worker-request-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for one worker request before killing the worker as hung.",
    )
    parser.add_argument(
        "--async-mode",
        action="store_true",
        default=False,
        help=(
            "Enable async decoupled video/action inference. "
            "Requires AHAWAM policy with action_offset support."
        ),
    )
    parser.add_argument(
        "--async-timing-config",
        default=None,
        help=(
            "Path to async_timing.yaml generated by calibrate_async_timing.py. "
            "Used to configure async intervals and staleness thresholds."
        ),
    )
    parser.add_argument(
        "--dual-gpu-async",
        action="store_true",
        default=False,
        help=(
            "Run dual-process async AHAWAM deployment: one full model on video GPU "
            "for video prefill and one full model on the action device."
        ),
    )
    parser.add_argument("--video-gpu", default="0", help="Physical GPU id for the video prefill worker.")
    parser.add_argument("--action-gpu", default="1", help="Physical GPU id for the action worker.")
    parser.add_argument(
        "--image-queue-size",
        type=int,
        default=1,
        help="Latest-wins queue size for video image requests.",
    )
    parser.add_argument(
        "--first-kv-timeout-s",
        type=float,
        default=10.0,
        help="Seconds action worker waits for the first video state before failing an action request.",
    )
    parser.add_argument(
        "--action-timeout-s",
        type=float,
        default=10.0,
        help="Seconds server waits for one dual-GPU action response.",
    )
    parser.add_argument(
        "--kv-max-age-ms",
        type=float,
        default=0.0,
        help="Max allowed inference state age in ms before rejecting action requests; 0 disables staleness rejection.",
    )
    parser.add_argument(
        "--worker-warmup-rounds",
        type=int,
        default=150,
        help="Number of warmup inference rounds for each GPU worker before accepting requests. "
             "Runs optional model warmup before accepting requests. Set to 0 to skip.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    Server(args).serve_forever()


if __name__ == "__main__":
    main()
