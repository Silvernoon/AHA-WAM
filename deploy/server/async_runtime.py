#!/usr/bin/env python3
"""Async runtime for decoupled video/action inference.

Architecture:
  - Single GPU thread owns the model and CUDA context
  - Priority event loop: action_request > video_prefill
  - StateBuffer provides thread-safe double buffering for video inference state
  - Network I/O is handled externally (by AsyncPolicyBackend in wam_policy_server)

Usage:
    from deploy.server.async_runtime import AsyncAHAWAMRuntime
    runtime = AsyncAHAWAMRuntime(policy, config)
    runtime.start(prompt="pick up the coin")
    ...
    runtime.stop()
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ActionRequest:
    """Request payload for an action chunk inference."""
    image: torch.Tensor
    proprio: torch.Tensor
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class ActionResult:
    """Result of an action chunk inference."""
    action_chunk: np.ndarray
    kv_version: int
    latency_ms: float
    timestamp: float = field(default_factory=time.perf_counter)


class StateBuffer:
    """Thread-safe double buffer for video inference state (inference_state).

    The GPU thread publishes new states after video prefill; the same GPU thread
    consumes the latest state when processing action requests. Thread safety is
    needed because the network thread checks availability.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._inference_state: Optional[dict] = None
        self._version: int = 0
        self._timestamp: float = 0.0

    @property
    def available(self) -> bool:
        with self._lock:
            return self._inference_state is not None

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def age_ms(self) -> float:
        """Age of current inference state in milliseconds."""
        with self._lock:
            if self._timestamp == 0.0:
                return float("inf")
            return (time.perf_counter() - self._timestamp) * 1000.0

    def publish(self, state: dict) -> int:
        """Publish a new inference state (called from GPU thread after video prefill)."""
        with self._lock:
            self._inference_state = state
            self._version += 1
            self._timestamp = time.perf_counter()
            v = self._version
        logger.debug("StateBuffer: published version %d", v)
        return v

    def consume(self) -> tuple:
        """Get the latest inference state and its version.

        Returns:
            (inference_state dict, version int) or (None, 0) if not available.
        """
        with self._lock:
            return self._inference_state, self._version

    def clear(self):
        """Clear the buffer (e.g., on episode reset)."""
        with self._lock:
            self._inference_state = None
            self._version = 0
            self._timestamp = 0.0


class AsyncAHAWAMRuntime:
    """GPU thread event loop for decoupled video/action inference.

    Single GPU thread processes events with priority:
      1. action_request (latency-sensitive)
      2. video prefill (can wait)
      3. idle (wait on condition)

    The network layer pushes images and action requests into queues;
    results are published to output queues.
    """

    def __init__(self, policy, config: dict):
        """
        Args:
            policy: AHAWAM policy instance (already loaded and in eval mode).
            config: Dict with async configuration:
                - action_priority: bool (default True)
                - max_video_kv_staleness_ms: float (default 500)
        """
        self._policy = policy
        self._config = config
        self._kv_buffer = StateBuffer()

        # Queues for inter-thread communication
        self._image_queue: queue.Queue = queue.Queue(maxsize=2)
        self._action_request_queue: queue.Queue = queue.Queue(maxsize=1)
        self._action_result_queue: queue.Queue = queue.Queue(maxsize=1)

        # Synchronization
        self._condition = threading.Condition()
        self._first_prefill_done = threading.Event()
        self._running = False
        self._gpu_thread: Optional[threading.Thread] = None

        # Prompt for video prefill
        self._prompt: Optional[str] = None

        # Config
        self._action_priority = config.get("action_priority", True)
        self._max_staleness_ms = config.get("max_video_kv_staleness_ms", 500.0)

        # Stats
        self._stats = {
            "video_prefills": 0,
            "action_chunks": 0,
            "action_stale_rejects": 0,
        }

    @property
    def kv_buffer(self) -> StateBuffer:
        return self._kv_buffer

    @property
    def first_prefill_done(self) -> threading.Event:
        return self._first_prefill_done

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def start(self, prompt: str):
        """Start the GPU event loop thread."""
        if self._running:
            logger.warning("AsyncAHAWAMRuntime already running.")
            return
        self._prompt = prompt
        self._running = True
        self._first_prefill_done.clear()
        self._kv_buffer.clear()
        self._gpu_thread = threading.Thread(
            target=self._gpu_loop, name="async-ahawam-worker", daemon=True
        )
        self._gpu_thread.start()
        logger.info("AsyncAHAWAMRuntime started (prompt=%r)", prompt[:50])

    def stop(self):
        """Stop the GPU event loop and wait for thread exit."""
        if not self._running:
            return
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=5.0)
            self._gpu_thread = None
        logger.info(
            "AsyncAHAWAMRuntime stopped. Stats: %s", self._stats
        )

    def reset(self, prompt: Optional[str] = None):
        """Reset for new episode: clear buffers, reset policy, optionally update prompt."""
        self._first_prefill_done.clear()
        self._kv_buffer.clear()

        # Drain queues
        for q in (self._image_queue, self._action_request_queue, self._action_result_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Reset policy state (history state etc.)
        self._policy._hard_reset()

        if prompt is not None:
            self._prompt = prompt

        self._stats = {
            "video_prefills": 0,
            "action_chunks": 0,
            "action_stale_rejects": 0,
        }
        logger.info("AsyncAHAWAMRuntime reset (prompt=%r)", (self._prompt or "")[:50])

    def push_image(self, image_tensor: torch.Tensor) -> bool:
        """Push a new image for video prefill (called from network thread).

        Returns True if queued, False if queue full (image dropped).
        """
        try:
            # Replace old image if queue is full (latest-wins policy)
            if self._image_queue.full():
                try:
                    self._image_queue.get_nowait()
                except queue.Empty:
                    pass
            self._image_queue.put_nowait(image_tensor)
            with self._condition:
                self._condition.notify()
            return True
        except queue.Full:
            return False

    def request_action(self, image_tensor: torch.Tensor, proprio: torch.Tensor) -> bool:
        """Submit an action request (called from network thread).

        Returns True if queued, False if a previous request is still pending.
        """
        req = ActionRequest(image=image_tensor, proprio=proprio)
        try:
            # Replace pending request (latest-wins)
            if self._action_request_queue.full():
                try:
                    self._action_request_queue.get_nowait()
                except queue.Empty:
                    pass
            self._action_request_queue.put_nowait(req)
            with self._condition:
                self._condition.notify()
            return True
        except queue.Full:
            return False

    def get_action_result(self, timeout: Optional[float] = None) -> Optional[ActionResult]:
        """Wait for and return the next action result.

        Args:
            timeout: Max seconds to wait (None = block forever).

        Returns:
            ActionResult or None if timeout.
        """
        try:
            return self._action_result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # GPU thread
    # ------------------------------------------------------------------

    def _gpu_loop(self):
        """Single GPU thread event loop with priority scheduling."""
        logger.info("GPU loop started.")
        try:
            while self._running:
                did_work = False

                # Priority 1: Action chunk (latency-sensitive)
                if self._action_priority:
                    did_work = self._try_process_action() or did_work

                # Priority 2: Video prefill
                if not did_work or not self._action_priority:
                    did_work = self._try_process_video() or did_work

                # If action wasn't priority, try it after video
                if not self._action_priority and not did_work:
                    did_work = self._try_process_action() or did_work

                # Idle: wait for new events
                if not did_work:
                    with self._condition:
                        self._condition.wait(timeout=0.001)
        except Exception:
            logger.exception("GPU loop crashed!")
            self._running = False

    def _try_process_action(self) -> bool:
        """Try to process one action request. Returns True if processed."""
        try:
            req = self._action_request_queue.get_nowait()
        except queue.Empty:
            return False

        # Must have inference state available
        if not self._kv_buffer.available:
            logger.warning("Action request received but no inference state available yet. Waiting...")
            # Put request back and wait for first prefill
            self._first_prefill_done.wait(timeout=2.0)
            if not self._kv_buffer.available:
                logger.error("Timeout waiting for first video prefill.")
                return True  # consumed the request (dropped)

        # Check staleness
        staleness_ms = self._kv_buffer.age_ms
        if staleness_ms > self._max_staleness_ms:
            self._stats["action_stale_rejects"] += 1
            logger.warning(
                "inference state too stale (%.0f ms > %.0f ms). Dropping action request.",
                staleness_ms, self._max_staleness_ms,
            )
            return True

        # Run action inference
        state, version = self._kv_buffer.consume()
        t0 = time.perf_counter()
        try:
            action_chunk = self._policy.infer_action_chunk_only(
                inference_state=state,
                image_tensor=req.image,
                proprio=req.proprio,
            )
        except Exception:
            logger.exception("Action chunk inference failed!")
            return True

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = ActionResult(
            action_chunk=action_chunk,
            kv_version=version,
            latency_ms=latency_ms,
        )

        # Publish result (replace if consumer is slow)
        if self._action_result_queue.full():
            try:
                self._action_result_queue.get_nowait()
            except queue.Empty:
                pass
        self._action_result_queue.put_nowait(result)

        self._stats["action_chunks"] += 1
        logger.debug(
            "Action chunk done: %.1f ms, kv_version=%d", latency_ms, version
        )
        return True

    def _try_process_video(self) -> bool:
        """Try to process one video prefill. Returns True if processed.

        Drains the image queue and only uses the latest image (skip stale frames).
        """
        image_tensor = None
        skipped = 0
        while True:
            try:
                img = self._image_queue.get_nowait()
                if image_tensor is not None:
                    skipped += 1
                image_tensor = img
            except queue.Empty:
                break

        if image_tensor is None:
            return False

        if skipped > 0:
            logger.debug("Video prefill: skipped %d stale frame(s), using latest.", skipped)

        # Run video prefill
        t0 = time.perf_counter()
        try:
            state = self._policy.prefill_video_only(image_tensor, self._prompt)
        except Exception:
            logger.exception("Video prefill failed!")
            return True

        version = self._kv_buffer.publish(state)
        self._first_prefill_done.set()
        self._stats["video_prefills"] += 1

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug("Video prefill done: %.1f ms → kv_version=%d", latency_ms, version)
        return True

    # ------------------------------------------------------------------
    # Calibration (synchronous, for timing measurement)
    # ------------------------------------------------------------------

    def timing_calibration(
        self,
        image_tensor: torch.Tensor,
        proprio: torch.Tensor,
        rounds: int = 5,
    ) -> dict:
        """Run synchronous timing calibration (call before start()).

        Returns dict with video_prefill_ms, action_chunk_ms (p50 values).
        """
        if self._running:
            raise RuntimeError("Cannot calibrate while runtime is running. Call stop() first.")

        prompt = self._prompt or "calibration"
        video_times = []
        action_times = []

        for _ in range(rounds):
            self._policy._hard_reset()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = self._policy.prefill_video_only(image_tensor, prompt)
            torch.cuda.synchronize()
            video_times.append((time.perf_counter() - t0) * 1000.0)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._policy.infer_action_chunk_only(state, image_tensor, proprio)
            torch.cuda.synchronize()
            action_times.append((time.perf_counter() - t0) * 1000.0)

        # Drop first round if enough data
        if rounds > 2:
            video_times = video_times[1:]
            action_times = action_times[1:]

        video_arr = np.array(video_times)
        action_arr = np.array(action_times)

        return {
            "video_prefill_ms": float(np.percentile(video_arr, 50)),
            "video_prefill_p95_ms": float(np.percentile(video_arr, 95)),
            "action_chunk_ms": float(np.percentile(action_arr, 50)),
            "action_chunk_p95_ms": float(np.percentile(action_arr, 95)),
        }
