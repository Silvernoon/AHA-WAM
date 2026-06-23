#!/usr/bin/env python3
"""Dummy TCP client for AHAWAM policy server testing without ROS."""

import argparse
import logging
import socket
import time
from pathlib import Path

import numpy as np
from PIL import Image

from deploy.common.tcp_protocol import recv_message, send_message


logger = logging.getLogger("wam_policy_dummy_client")



def make_image(args):
    if args.image:
        img = Image.open(args.image).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    if args.dummy_image == "random":
        rng = np.random.default_rng(args.seed)
        return rng.integers(
            0,
            256,
            size=(args.image_height, args.image_width, 3),
            dtype=np.uint8,
        )

    value = int(max(0, min(255, args.gray_value)))
    return np.full((args.image_height, args.image_width, 3), value, dtype=np.uint8)


def connect(args):
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.settimeout(args.socket_timeout)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.connect((args.host, args.port))
    return conn


def setup_logging(log_level):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)

    image = make_image(args)
    state = np.zeros((args.action_dim,), dtype=np.float32)
    velocity = np.zeros((args.action_dim,), dtype=np.float32)
    effort = np.zeros((args.action_dim,), dtype=np.float32)
    base_vel = np.zeros((2,), dtype=np.float32)

    logger.info(
        "Connecting to local policy server %s:%d image_shape=%s image_dtype=%s",
        args.host,
        args.port,
        image.shape,
        image.dtype,
    )

    with connect(args) as conn:
        if args.send_reset:
            reset_msg = {"type": "reset", "instruction": args.instruction}
            t0 = time.perf_counter()
            request_bytes = send_message(conn, reset_msg)
            reply, reply_bytes = recv_message(conn)
            logger.info(
                "Reset reply: ok=%s type=%r request=%d bytes reply=%d bytes rtt=%.1f ms",
                reply.get("ok"),
                reply.get("type"),
                request_bytes,
                reply_bytes,
                (time.perf_counter() - t0) * 1000.0,
            )
            if not reply.get("ok", False):
                raise RuntimeError(f"reset failed: {reply}")

        for step in range(args.num_requests):
            request = {
                "type": "infer",
                "episode_id": args.episode_id,
                "step": step,
                "timestamp": time.time(),
                "instruction": args.instruction,
                "state": state,
                "velocity": velocity,
                "effort": effort,
                "base_vel": base_vel,
                "images": {"front": image},
            }

            logger.info("Sending dummy infer request #%d", step + 1)
            t0 = time.perf_counter()
            request_bytes = send_message(conn, request)
            reply, reply_bytes = recv_message(conn)
            round_trip_ms = (time.perf_counter() - t0) * 1000.0

            if not reply.get("ok", False):
                logger.error("Server returned error: %s", reply)
                if reply.get("fatal"):
                    logger.error("Server marked this error as fatal; stop testing and restart server before retrying.")
                return 2

            action_chunk = np.asarray(reply["action_chunk"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != args.action_dim:
                raise ValueError(f"bad action chunk shape: {action_chunk.shape}")
            if not np.all(np.isfinite(action_chunk)):
                raise ValueError("action chunk contains NaN or Inf")

            logger.info(
                "Reply #%d: server_step=%s action_shape=%s model=%.1f ms rtt=%.1f ms request=%d bytes reply=%d bytes",
                step + 1,
                reply.get("server_inference_step"),
                action_chunk.shape,
                float(reply.get("model_latency_ms", -1.0)),
                round_trip_ms,
                request_bytes,
                reply_bytes,
            )

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    logger.info("Dummy TCP client completed successfully.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--socket-timeout", type=float, default=900.0)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--image", default=None, help="Optional RGB image path. If omitted, a dummy image is generated.")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--dummy-image", choices=("gray", "random"), default="gray")
    parser.add_argument("--gray-value", type=int, default=127)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--send-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
