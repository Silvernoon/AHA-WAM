#!/usr/bin/env python3
"""Dummy async client for testing the dual-device AHAWAM server path.

This exercises the decoupled protocol without ROS:
- one TCP connection continuously pushes ``type: image`` messages;
- a second TCP connection sends ``reset`` and ``action_request`` messages.
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time

import numpy as np

from deploy.common.tcp_protocol import recv_message, send_message

logger = logging.getLogger("wam_policy_async_dummy_client")


def make_dummy_image(height: int, width: int, mode: str = "random", seed: int | None = None) -> np.ndarray:
    """Generate a dummy RGB image."""
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return np.full((height, width, 3), 127, dtype=np.uint8)


def make_dummy_state(action_dim: int) -> np.ndarray:
    """Generate a dummy joint state."""
    return np.zeros(action_dim, dtype=np.float32)


def connect(host: str, port: int, timeout_s: float) -> socket.socket:
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(timeout_s)
    conn.connect((host, port))
    return conn


def image_push_loop(args: argparse.Namespace, stop_event: threading.Event, stats: dict[str, int]) -> None:
    """Push latest camera frames on a dedicated TCP connection."""
    interval = 1.0 / args.image_push_fps
    image = make_dummy_image(args.image_height, args.image_width, args.image_mode, args.seed)
    pushed = 0

    try:
        conn = connect(args.host, args.port, args.socket_timeout)
    except Exception as exc:
        logger.error("Image channel failed to connect: %s", exc)
        return

    logger.info("Image channel started: fps=%s", args.image_push_fps)
    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            message = {
                "type": "image",
                "images": {"front": image},
                "timestamp": time.time(),
            }
            try:
                send_message(conn, message)
                reply, _ = recv_message(conn)
            except Exception as exc:
                if not stop_event.is_set():
                    logger.error("Image channel error: %s", exc)
                break

            if reply.get("ok"):
                pushed += 1
            else:
                logger.warning("Image push rejected: %s", reply.get("error", reply))

            sleep_time = interval - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        conn.close()
        stats["images_pushed"] = pushed
        logger.info("Image channel stopped: pushed=%d", pushed)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    image = make_dummy_image(args.image_height, args.image_width, args.image_mode, args.seed)
    state = make_dummy_state(args.action_dim)

    logger.info("Connecting action channel to %s:%d", args.host, args.port)
    action_conn = connect(args.host, args.port, args.socket_timeout)

    try:
        logger.info("Sending reset instruction=%r", args.instruction)
        send_message(action_conn, {"type": "reset", "instruction": args.instruction})
        reply, _ = recv_message(action_conn)
        if not reply.get("ok"):
            logger.error("Reset failed: %s", reply)
            return 1

        stop_event = threading.Event()
        stats = {"images_pushed": 0}
        image_thread = threading.Thread(
            target=image_push_loop,
            args=(args, stop_event, stats),
            daemon=True,
        )
        image_thread.start()

        logger.info("Waiting %.1f s for initial video prefill", args.prefill_wait)
        time.sleep(args.prefill_wait)

        interval = 1.0 / args.action_request_rate
        latencies_ms: list[float] = []
        state_versions: list[int] = []
        state_ages_ms: list[float] = []
        errors = 0

        for idx in range(args.num_action_requests):
            t0 = time.perf_counter()
            message = {
                "type": "action_request",
                "images": {"front": image},
                "state": state,
                "timestamp": time.time(),
            }
            try:
                send_message(action_conn, message)
                reply, reply_bytes = recv_message(action_conn)
            except Exception as exc:
                logger.error("Action request #%d failed: %s", idx + 1, exc)
                errors += 1
                continue

            rtt_ms = (time.perf_counter() - t0) * 1000.0
            if not reply.get("ok"):
                logger.warning("Action request #%d rejected: %s", idx + 1, reply.get("error", reply))
                errors += 1
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
                continue

            action_chunk = np.asarray(reply["action_chunk"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != args.action_dim:
                logger.error("Bad action shape: %s", action_chunk.shape)
                errors += 1
            if not np.all(np.isfinite(action_chunk)):
                logger.error("Action chunk contains NaN or Inf")
                errors += 1

            version = int(reply.get("kv_version", reply.get("state_version", -1)))
            age_ms = float(reply.get("kv_age_ms", reply.get("state_age_ms", -1.0)))
            latencies_ms.append(rtt_ms)
            state_versions.append(version)
            if age_ms >= 0:
                state_ages_ms.append(age_ms)

            logger.info(
                "Action %d/%d: shape=%s model=%.1fms rtt=%.1fms state_version=%s state_age=%.1fms reply=%d bytes",
                idx + 1,
                args.num_action_requests,
                action_chunk.shape,
                float(reply.get("model_latency_ms", -1.0)),
                rtt_ms,
                version,
                age_ms,
                reply_bytes,
            )

            sleep_time = interval - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

        stop_event.set()
        image_thread.join(timeout=2.0)

        logger.info("=" * 60)
        logger.info("ASYNC DUMMY TEST SUMMARY")
        logger.info("Images pushed: %d", stats["images_pushed"])
        logger.info("Action requests: %d", args.num_action_requests)
        logger.info("Errors: %d", errors)
        if latencies_ms:
            arr = np.asarray(latencies_ms)
            logger.info(
                "RTT ms: p50=%.1f p95=%.1f mean=%.1f min=%.1f max=%.1f",
                np.percentile(arr, 50),
                np.percentile(arr, 95),
                np.mean(arr),
                np.min(arr),
                np.max(arr),
            )
        if state_ages_ms:
            arr = np.asarray(state_ages_ms)
            logger.info(
                "State age ms: p50=%.1f p95=%.1f mean=%.1f max=%.1f",
                np.percentile(arr, 50),
                np.percentile(arr, 95),
                np.mean(arr),
                np.max(arr),
            )
        if state_versions:
            logger.info("State versions seen: %d -> %d", min(state_versions), max(state_versions))
        logger.info("RESULT: %s", "PASS" if errors == 0 else f"FAIL ({errors} errors)")
        logger.info("=" * 60)
        return 0 if errors == 0 else 1
    finally:
        action_conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dummy async client for testing dual-device AHAWAM serving.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--socket-timeout", type=float, default=60.0)
    parser.add_argument("--instruction", default="perform the task")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-mode", choices=("random", "gray"), default="gray")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-push-fps", type=float, default=30.0, help="Image push rate in Hz.")
    parser.add_argument("--action-request-rate", type=float, default=5.0, help="Action request rate in Hz.")
    parser.add_argument("--num-action-requests", type=int, default=10)
    parser.add_argument("--prefill-wait", type=float, default=2.0)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
