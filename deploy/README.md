# AHA-WAM Real-Robot Deployment

This directory provides server and client utilities for running AHA-WAM on a
real robot.

```text
deploy/
├── deploy_example.yml                 # Example runtime config
├── common/tcp_protocol.py             # Length-prefixed TCP helpers
├── server/ahawam_policy.py            # AHA-WAM policy wrapper
├── server/async_runtime.py            # Async video/action runtime
├── server/wam_policy_server.py        # TCP policy server
└── client/
    ├── wam_policy_dummy_client.py         # Sync test client
    ├── wam_policy_async_dummy_client.py   # Async/dual-device protocol test client
    └── wam_remote_client_node.py          # ROS robot-side client
```

The server receives robot observations over TCP and returns action chunks. The
same server supports synchronous inference and asynchronous video/action serving.
Use the asynchronous mode when the video branch and action branch should run at
separate rates or on separate devices.

The TCP transport uses Python pickle to preserve NumPy arrays. Run it only on a
trusted robot network.

## 1. Prepare the config

Create a local runtime config from the example:

```bash
cp deploy/deploy_example.yml deploy/deploy.yml
```

Edit at least these fields:

```yaml
checkpoint_path: ./checkpoints/your_checkpoint.pt
dataset_stats_path: ./checkpoints/your_dataset_stats.json
project_root: .
hydra_config_name: deploy
task: null
```

`hydra_config_name: deploy` composes [`configs/deploy.yml`](../configs/deploy.yml).
If `task` is left `null`, the task default from `configs/deploy.yml` is used. Set
`task` to a task config name when you want to override that default.

## 2. Start the server

Run the policy server on the GPU workstation:

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

For single-process asynchronous serving:

```bash
python deploy/server/wam_policy_server.py \
  --async-mode \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14
```

For dual-device asynchronous serving:

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

## 3. Test without ROS

Use the sync test client for the regular `infer` request path:

```bash
python deploy/client/wam_policy_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "your task instruction" \
  --num-requests 1
```

Use the async test client for the dual-device protocol path. It opens separate
image and action TCP channels, pushes `image` messages in the background, and
sends `action_request` messages on the foreground channel:

```bash
python deploy/client/wam_policy_async_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "your task instruction" \
  --num-action-requests 10
```

Use the server IP instead of `127.0.0.1` when testing from another machine.

## 4. Run the ROS client

Run the ROS client on the robot computer after checking camera/state topics and
action limits.

For a synchronous server, the client sends one `infer` request whenever it needs
a new action chunk:

```bash
python deploy/client/wam_remote_client_node.py \
  --server-ip <GPU_WORKSTATION_IP> \
  --server-port 10000 \
  --instruction "your task instruction" \
  --dry-run
```

For an asynchronous server started with `--async-mode` or `--dual-gpu-async`, use
the decoupled image stream and action request protocol:

```bash
python deploy/client/wam_remote_client_node.py \
  --server-ip <GPU_WORKSTATION_IP> \
  --server-port 10000 \
  --instruction "your task instruction" \
  --async-policy-protocol \
  --image-stream-rate 30 \
  --dry-run
```

The async client switch is `--async-policy-protocol`. In this mode, the client
keeps sending the latest camera frame on one TCP channel and requests action
chunks on another channel. Tune `--image-stream-rate` for your camera, network,
and server throughput; in real deployments, try several values and monitor
end-to-end action latency and frame freshness. Start with `--dry-run` to inspect
the received action chunks. Remove it after confirming the ROS topics, joint
limits, gripper limits, and emergency-stop procedure for your robot.

## Message protocol

All messages are dictionaries sent through `deploy/common/tcp_protocol.py`.

Sync request:

```python
{
    "type": "infer",
    "instruction": "your task instruction",
    "state": np.ndarray,          # shape [14]
    "images": {
        "cam_high": head_image,
        "cam_left_wrist": left_wrist_image,
        "cam_right_wrist": right_wrist_image,
    },  # synchronized uint8 HxWx3 RGB
}
```

DA3 shared-stem checkpoints require all three synchronized views in the fixed
order shown above. Legacy single-view checkpoints continue to accept
`{"images": {"front": image}}`.

Async mode additionally supports:

- `{"type": "image", "images": synchronized_three_view_dict}`
- `{"type": "action_request", "state": state, "images": synchronized_three_view_dict}`
- `{"type": "reset", "instruction": "..."}`
