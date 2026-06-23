# AHAWAM Real-Robot Deployment

This directory contains the public real-robot deployment surface for AHAWAM:

```text
deploy/
├── deploy_example.yml          # Copy to deploy/deploy.yml and fill local paths
├── common/tcp_protocol.py      # Length-prefixed TCP helpers
├── server/ahawam_policy.py     # AHAWAM policy wrapper
├── server/async_runtime.py     # Async video/action runtime
├── server/wam_policy_server.py # TCP policy server
└── client/
    ├── wam_policy_dummy_client.py         # Sync smoke test client
    ├── wam_policy_async_dummy_client.py   # Async/dual-device protocol smoke test
    └── wam_remote_client_node.py          # ROS robot-side client
```

This is a reference server-client framework for real-robot deployment. It keeps
the public protocol and runtime structure simple on purpose, and includes an
asynchronous dual-stream WAM example where the video branch and action branch can
run in separate workers/threads.

The TCP transport uses Python pickle to preserve NumPy arrays. Only use it on a
trusted robot network.

## 1. Prepare config

```bash
cp deploy/deploy_example.yml deploy/deploy.yml
```

Edit at least:

```yaml
checkpoint_path: ./checkpoints/your_checkpoint.pt
dataset_stats_path: ./checkpoints/your_dataset_stats.json
project_root: .
hydra_config_name: deploy
task: null
```

`hydra_config_name: deploy` composes [`configs/deploy.yml`](../configs/deploy.yml).
Set `task` only when you need to override the default task selected by that Hydra
config.

## 2. Start the server

Run on the GPU workstation:

```bash
python deploy/server/wam_policy_server.py \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14 \
  --host 0.0.0.0 \
  --port 10000
```

Async single-worker mode:

```bash
python deploy/server/wam_policy_server.py \
  --async-mode \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14
```

Async two-device mode. This demonstrates the dual-stream WAM runtime, with the
video branch and action branch assigned to separate workers/devices:

```bash
python deploy/server/wam_policy_server.py \
  --dual-gpu-async \
  --video-gpu 0 \
  --action-gpu 1 \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14
```

## 3. Smoke test without ROS

Use the sync dummy client for the regular `infer` request path:

```bash
python deploy/client/wam_policy_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "your task instruction" \
  --num-requests 1
```

Use the async dummy client to exercise the dual-device protocol path. It opens
separate image and action TCP channels, pushes `image` messages in the
background, and sends `action_request` messages on the foreground channel:

```bash
python deploy/client/wam_policy_async_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "your task instruction" \
  --num-action-requests 10
```

Use the server IP instead of `127.0.0.1` when testing from another machine.

## 4. Run the ROS client

Run on the robot computer after verifying camera/state topics and action limits:

```bash
python deploy/client/wam_remote_client_node.py \
  --server-ip <GPU_WORKSTATION_IP> \
  --server-port 10000 \
  --instruction "your task instruction" \
  --dry-run
```

Remove `--dry-run` only after validating the received action chunks, ROS topics,
joint limits, gripper limits, and emergency-stop procedure.

## Message protocol

All messages are dictionaries sent through `deploy/common/tcp_protocol.py`.

Sync request:

```python
{
    "type": "infer",
    "instruction": "your task instruction",
    "state": np.ndarray,          # shape [14]
    "images": {"front": image}, # uint8 HxWx3 RGB
}
```

Async mode additionally supports:

- `{"type": "image", "images": {"front": image}}`
- `{"type": "action_request", "state": state, "images": {"front": image}}`
- `{"type": "reset", "instruction": "..."}`

## Notes

- The public release keeps the sync and async video/action protocols.
- Private acceleration backends are intentionally omitted from this deployment surface.
- The ROS client is a template for real hardware; audit safety limits before moving a robot.
