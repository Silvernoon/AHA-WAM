#!/usr/bin/env python3
# -- coding: UTF-8 --
"""ROS bridge/client for AHAWAM real-robot deployment.

This node runs on the robot computer. It subscribes to ROS observations,
sends plain Python/NumPy data to the policy server, then publishes the
returned 14-D action chunk to /master/joint_left and /master/joint_right.

With `roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true`, those
/master/joint_* topics are the command inputs consumed by the Piper ROS node to
control the physical puppet arms.
"""

import argparse
import select
import signal
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
import rospy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

from deploy.common.tcp_protocol import recv_message as _recv_message, send_message


PIPER_JOINT_LIMITS_RAD = np.array(
    [
        [-2.61799, 2.61799],  # joint0: -150 to 150 deg
        [0.0, 3.14159],       # joint1: 0 to 180 deg
        [-2.96706, 0.0],      # joint2: -170 to 0 deg
        [-1.74533, 1.74533],  # joint3: -100 to 100 deg
        [-1.22173, 1.22173],  # joint4: -70 to 70 deg
        [-2.09440, 2.09440],  # joint5: -120 to 120 deg
    ],
    dtype=np.float32,
)


class StopController:
    def __init__(self):
        self.stop_requested = False
        self.force_exit = False

    def install(self):
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        if self.stop_requested:
            self.force_exit = True
            print("\n[wam_remote_client] second Ctrl-C received, forcing shutdown")
            rospy.signal_shutdown("forced by second Ctrl-C")
            return
        self.stop_requested = True
        print("\n[wam_remote_client] Ctrl-C received, stopping inference and resetting...")


def recv_message(conn):
    message, _ = _recv_message(conn)
    return message



def make_joint_state(position):
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    msg.position = np.asarray(position, dtype=np.float32).reshape(7).tolist()
    return msg


def enter_pressed():
    return bool(select.select([sys.stdin], [], [], 0)[0])


class PolicyClient:
    def __init__(self, host, port, timeout_s):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.conn = None

    def connect(self):
        self.close()
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.settimeout(self.timeout_s)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.connect((self.host, self.port))
        self.conn = conn

    def request(self, message):
        if self.conn is None:
            self.connect()
        try:
            send_message(self.conn, message)
            return recv_message(self.conn)
        except Exception:
            self.close()
            raise

    def reset_episode(self, instruction):
        return self.request({"type": "reset", "instruction": instruction})

    def close(self):
        if self.conn is not None:
            try:
                self.conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.conn.close()
            self.conn = None


class LatestFrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def update(self, frame):
        with self._lock:
            self._frame = {
                "timestamp": frame["timestamp"],
                "front": frame["front"].copy(),
                "state": frame["state"].copy(),
                "velocity": frame["velocity"].copy(),
                "effort": frame["effort"].copy(),
                "base_vel": frame["base_vel"].copy(),
            }

    def get(self):
        with self._lock:
            if self._frame is None:
                return None
            return {
                "timestamp": self._frame["timestamp"],
                "front": self._frame["front"].copy(),
                "state": self._frame["state"].copy(),
                "velocity": self._frame["velocity"].copy(),
                "effort": self._frame["effort"].copy(),
                "base_vel": self._frame["base_vel"].copy(),
            }


def build_policy_request(msg_type, args, frame, step):
    return {
        "type": msg_type,
        "episode_id": args.episode_id,
        "step": step,
        "timestamp": frame["timestamp"],
        "instruction": args.instruction,
        "state": frame["state"],
        "velocity": frame["velocity"],
        "effort": frame["effort"],
        "base_vel": frame["base_vel"],
        "images": {"front": frame["front"]},
    }


class RosOperator:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.img_front_deque = deque()
        self.puppet_left_deque = deque()
        self.puppet_right_deque = deque()
        self.robot_base_deque = deque()
        self.left_cmd_pub = None
        self.right_cmd_pub = None
        self.init_ros()

    def init_ros(self):
        rospy.init_node("wam_remote_client_node", anonymous=True, disable_signals=True)
        rospy.Subscriber(
            self.args.img_front_topic,
            Image,
            self.img_front_callback,
            queue_size=10,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.puppet_left_topic,
            JointState,
            self.puppet_left_callback,
            queue_size=50,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.args.puppet_right_topic,
            JointState,
            self.puppet_right_callback,
            queue_size=50,
            tcp_nodelay=True,
        )
        if self.args.use_robot_base:
            rospy.Subscriber(
                self.args.robot_base_topic,
                Odometry,
                self.robot_base_callback,
                queue_size=20,
                tcp_nodelay=True,
            )

        # In Piper mode=1 these /master/joint_* topics are command topics for
        # the physical puppet arms.
        self.left_cmd_pub = rospy.Publisher(self.args.cmd_left_topic, JointState, queue_size=10)
        self.right_cmd_pub = rospy.Publisher(self.args.cmd_right_topic, JointState, queue_size=10)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 100:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def puppet_left_callback(self, msg):
        if len(self.puppet_left_deque) >= 200:
            self.puppet_left_deque.popleft()
        self.puppet_left_deque.append(msg)

    def puppet_right_callback(self, msg):
        if len(self.puppet_right_deque) >= 200:
            self.puppet_right_deque.popleft()
        self.puppet_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 100:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def get_frame(self):
        if not self.img_front_deque:
            return None, "no front image"
        if not self.puppet_left_deque:
            return None, "no left puppet state"
        if not self.puppet_right_deque:
            return None, "no right puppet state"
        if self.args.use_robot_base and not self.robot_base_deque:
            return None, "no robot base state"

        frame_time = self.img_front_deque[-1].header.stamp.to_sec()

        if self.puppet_left_deque[-1].header.stamp.to_sec() < frame_time:
            return None, "left puppet state older than image"
        if self.puppet_right_deque[-1].header.stamp.to_sec() < frame_time:
            return None, "right puppet state older than image"
        if self.args.use_robot_base and self.robot_base_deque[-1].header.stamp.to_sec() < frame_time:
            return None, "base state older than image"

        while self.img_front_deque and self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        while self.puppet_left_deque and self.puppet_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_left_deque.popleft()
        while self.puppet_right_deque and self.puppet_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_right_deque.popleft()

        if not self.img_front_deque or not self.puppet_left_deque or not self.puppet_right_deque:
            return None, "synchronized queues emptied"

        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), "passthrough")
        left = self.puppet_left_deque.popleft()
        right = self.puppet_right_deque.popleft()

        base_vel = np.zeros(2, dtype=np.float32)
        if self.args.use_robot_base:
            while self.robot_base_deque and self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            if not self.robot_base_deque:
                return None, "base queue emptied"
            base = self.robot_base_deque.popleft()
            base_vel = np.array([base.twist.twist.linear.x, base.twist.twist.angular.z], dtype=np.float32)

        state = np.concatenate(
            [
                np.asarray(left.position, dtype=np.float32),
                np.asarray(right.position, dtype=np.float32),
            ],
            axis=0,
        )
        velocity = np.concatenate(
            [
                np.asarray(left.velocity if left.velocity else np.zeros(7), dtype=np.float32),
                np.asarray(right.velocity if right.velocity else np.zeros(7), dtype=np.float32),
            ],
            axis=0,
        )
        effort = np.concatenate(
            [
                np.asarray(left.effort if left.effort else np.zeros(7), dtype=np.float32),
                np.asarray(right.effort if right.effort else np.zeros(7), dtype=np.float32),
            ],
            axis=0,
        )

        if state.shape[0] != 14:
            return None, f"bad state dim: {state.shape}"

        return {
            "timestamp": frame_time,
            "front": np.asarray(img_front),
            "state": state,
            "velocity": velocity,
            "effort": effort,
            "base_vel": base_vel,
        }, None

    def publish_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(14)
        left_msg = make_joint_state(action[:7])
        right_msg = make_joint_state(action[7:14])
        right_msg.header.stamp = left_msg.header.stamp
        self.left_cmd_pub.publish(left_msg)
        self.right_cmd_pub.publish(right_msg)

    def latest_state(self):
        if not self.puppet_left_deque or not self.puppet_right_deque:
            return None
        left = np.asarray(self.puppet_left_deque[-1].position, dtype=np.float32)
        right = np.asarray(self.puppet_right_deque[-1].position, dtype=np.float32)
        if left.shape[0] != 7 or right.shape[0] != 7:
            return None
        return np.concatenate([left, right], axis=0)


class SafetyFilter:
    def __init__(self, args):
        self.args = args
        self.last_action = None
        max_delta_single = np.array(
            [args.max_joint_delta] * 6 + [args.max_gripper_delta],
            dtype=np.float32,
        )
        self.max_delta = np.concatenate([max_delta_single, max_delta_single], axis=0)
        self.joint_min = np.concatenate(
            [PIPER_JOINT_LIMITS_RAD[:, 0], [args.gripper_min], PIPER_JOINT_LIMITS_RAD[:, 0], [args.gripper_min]],
            axis=0,
        )
        self.joint_max = np.concatenate(
            [PIPER_JOINT_LIMITS_RAD[:, 1], [args.gripper_max], PIPER_JOINT_LIMITS_RAD[:, 1], [args.gripper_max]],
            axis=0,
        )

    def reset(self, current_state):
        self.last_action = np.asarray(current_state, dtype=np.float32).reshape(14)

    def filter(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(14)
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains NaN or Inf")

        action = np.clip(action, self.joint_min, self.joint_max)
        if self.last_action is not None:
            delta = np.clip(action - self.last_action, -self.max_delta, self.max_delta)
            action = self.last_action + delta

        action = np.clip(action, self.joint_min, self.joint_max)
        self.last_action = action.copy()
        return action


def clip_action(action, args):
    action = np.asarray(action, dtype=np.float32).reshape(14)
    joint_min = np.concatenate(
        [PIPER_JOINT_LIMITS_RAD[:, 0], [args.gripper_min], PIPER_JOINT_LIMITS_RAD[:, 0], [args.gripper_min]],
        axis=0,
    )
    joint_max = np.concatenate(
        [PIPER_JOINT_LIMITS_RAD[:, 1], [args.gripper_max], PIPER_JOINT_LIMITS_RAD[:, 1], [args.gripper_max]],
        axis=0,
    )
    return np.clip(action, joint_min, joint_max)


def reset_target(args):
    return clip_action(np.concatenate([args.reset_left, args.reset_right], axis=0), args)


def reset_puppet_arms(ros, args, start_state=None):
    if args.no_reset_on_exit or args.dry_run:
        return

    target = reset_target(args)
    if start_state is None:
        start_state = ros.latest_state()
    if start_state is None:
        start_state = target.copy()
        rospy.logwarn("no latest puppet state available; publishing reset target directly")
    start_state = clip_action(start_state, args)

    steps = max(1, args.reset_steps)
    rate = rospy.Rate(args.reset_rate)
    rospy.logwarn("resetting puppet arms to configured reset pose...")
    for idx in range(1, steps + 1):
        if rospy.is_shutdown():
            break
        alpha = idx / float(steps)
        action = start_state + (target - start_state) * alpha
        ros.publish_action(clip_action(action, args))
        rate.sleep()

    for _ in range(max(0, args.reset_hold_steps)):
        if rospy.is_shutdown():
            break
        ros.publish_action(target)
        rate.sleep()
    rospy.logwarn("reset command sequence finished")


def initialize_puppet_arms(ros, args, init_start_state):
    if args.dry_run:
        return clip_action(init_start_state, args)

    target = clip_action(init_start_state, args)
    deadline = time.time() + max(0.0, args.init_state_timeout)
    start_state = ros.latest_state()
    while start_state is None and not rospy.is_shutdown() and time.time() < deadline:
        rospy.logwarn_throttle(1.0, "waiting for latest puppet state before initialization...")
        rospy.sleep(0.05)
        start_state = ros.latest_state()

    if start_state is None:
        start_state = target.copy()
        rospy.logwarn("no latest puppet state available; publishing init target directly")
    start_state = clip_action(start_state, args)

    steps = max(1, args.init_steps)
    rate = rospy.Rate(args.init_rate)
    rospy.logwarn("initializing puppet arms to init_start_state...")
    for idx in range(1, steps + 1):
        if rospy.is_shutdown():
            break
        alpha = idx / float(steps)
        action = start_state + (target - start_state) * alpha
        ros.publish_action(clip_action(action, args))
        rate.sleep()

    for _ in range(max(0, args.init_hold_steps)):
        if rospy.is_shutdown():
            break
        ros.publish_action(target)
        rate.sleep()
    rospy.logwarn("initialization command sequence finished")
    return target.copy()


def image_stream_loop(args, ros, frame_store, stop, stream_stop):
    client = PolicyClient(args.server_ip, args.server_port, args.socket_timeout)
    rate = rospy.Rate(args.image_stream_rate)
    try:
        client.connect()
        while (
            not rospy.is_shutdown()
            and not stop.stop_requested
            and not stream_stop.is_set()
        ):
            frame, err = ros.get_frame()
            if frame is None:
                rospy.logwarn_throttle(2.0, f"waiting for synchronized image stream frame: {err}")
                rate.sleep()
                continue
            frame_store.update(frame)
            try:
                reply = client.request(build_policy_request("image", args, frame, step=0))
                if not reply.get("ok", False):
                    rospy.logwarn_throttle(1.0, f"image stream rejected: {reply.get('error')}")
            except Exception as exc:
                rospy.logwarn(f"image stream connection failed, reconnecting: {exc}")
                client.close()
                rospy.sleep(0.5)
                try:
                    client.connect()
                except Exception as reconnect_exc:
                    rospy.logwarn_throttle(1.0, f"image stream reconnect failed: {reconnect_exc}")
            rate.sleep()
    finally:
        client.close()


def run_async(args):
    stop = StopController()
    stop.install()
    ros = RosOperator(args)
    action_client = PolicyClient(args.server_ip, args.server_port, args.socket_timeout)
    safety = SafetyFilter(args)
    frame_store = LatestFrameStore()
    stream_stop = threading.Event()

    init_start_state = [-0.446322184, 1.368098032, -0.784003136, -0.086557128, 0.995546524, 0.152652444, 0.0,
                   0.403968152, 1.4532770840000002, -0.762058584, 0.0, 0.9592106720000001, -0.037556932, 0]

    print(f"[wam_remote_client] connecting action channel to {args.server_ip}:{args.server_port}")
    action_client.connect()
    action_client.reset_episode(args.instruction)

    image_thread = threading.Thread(
        target=image_stream_loop,
        args=(args, ros, frame_store, stop, stream_stop),
        daemon=True,
    )
    image_thread.start()

    if not args.no_wait_enter:
        input(
            "Press Enter to start WAM async remote inference. "
            "Make sure piper is launched with mode:=1 auto_enable:=true..."
        )

    initialized_state = initialize_puppet_arms(ros, args, init_start_state)
    if initialized_state is not None:
        safety.reset(initialized_state)

    step = 0
    reset_start_state = initialized_state
    try:
        while not rospy.is_shutdown() and not stop.stop_requested and step < args.max_steps:
            if enter_pressed():
                _ = sys.stdin.read(1)
                print("[wam_remote_client] stop requested from keyboard")
                break

            frame = frame_store.get()
            if frame is None:
                rospy.logwarn_throttle(1.0, "waiting for first streamed frame before action request...")
                rospy.sleep(0.02)
                continue

            if safety.last_action is None:
                safety.reset(frame["state"])
            reset_start_state = frame["state"].copy()

            request = build_policy_request("action_request", args, frame, step)
            t0 = time.perf_counter()
            reply = action_client.request(request)
            round_trip_ms = (time.perf_counter() - t0) * 1000.0
            if not reply.get("ok", False):
                raise RuntimeError(f"policy server returned error: {reply.get('error')}")

            action_chunk = np.asarray(reply["action_chunk"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 14:
                raise ValueError(f"bad action chunk shape: {action_chunk.shape}")

            execute_n = min(args.execute_actions, action_chunk.shape[0])
            for i in range(execute_n):
                if stop.stop_requested:
                    break
                action = safety.filter(action_chunk[i])
                if args.dry_run:
                    rospy.loginfo_throttle(1.0, f"dry-run action left={action[:7]} right={action[7:14]}")
                else:
                    print(action)
                    ros.publish_action(action)
                    reset_start_state = action.copy()
                step += 1
                rospy.sleep(1.0 / args.control_rate)
                if rospy.is_shutdown() or stop.stop_requested or step >= args.max_steps:
                    break

            rospy.loginfo(
                "step=%d chunk=%d exec=%d model=%.1fms rtt=%.1fms kv=%s kv_age=%.1fms video=%.1fms",
                step,
                action_chunk.shape[0],
                execute_n,
                float(reply.get("model_latency_ms", -1.0)),
                round_trip_ms,
                str(reply.get("kv_version", "?")),
                float(reply.get("kv_age_ms", -1.0)),
                float(reply.get("video_prefill_ms", -1.0)),
            )
    finally:
        stream_stop.set()
        image_thread.join(timeout=2.0)
        if not stop.force_exit:
            reset_puppet_arms(ros, args, reset_start_state)
        action_client.close()
        rospy.signal_shutdown("wam async remote client finished")


def run(args):
    if args.async_policy_protocol:
        return run_async(args)

    stop = StopController()
    stop.install()
    ros = RosOperator(args)
    client = PolicyClient(args.server_ip, args.server_port, args.socket_timeout)
    safety = SafetyFilter(args)
    rate = rospy.Rate(args.request_rate)

    init_start_state = [-0.446322184, 1.368098032, -0.784003136, -0.086557128, 0.995546524, 0.152652444, 0.0, 
                   0.403968152, 1.4532770840000002, -0.762058584, 0.0, 0.9592106720000001, -0.037556932, 0]

    print(f"[wam_remote_client] connecting to {args.server_ip}:{args.server_port}")
    client.connect()
    client.reset_episode(args.instruction)

    if not args.no_wait_enter:
        input(
            "Press Enter to start WAM remote inference. "
            "Make sure piper is launched with mode:=1 auto_enable:=true..."
        )

    initialized_state = initialize_puppet_arms(ros, args, init_start_state)
    if initialized_state is not None:
        safety.reset(initialized_state)

    step = 0
    reset_start_state = initialized_state
    try:
        while not rospy.is_shutdown() and not stop.stop_requested and step < args.max_steps:
            if enter_pressed():
                _ = sys.stdin.read(1)
                print("[wam_remote_client] stop requested from keyboard")
                break

            frame, err = ros.get_frame()
            if frame is None:
                rospy.logwarn_throttle(2.0, f"waiting for synchronized frame: {err}")
                rate.sleep()
                continue

            if safety.last_action is None:
                safety.reset(frame["state"])

            reset_start_state = frame["state"].copy()
            request = {
                "type": "infer",
                "episode_id": args.episode_id,
                "step": step,
                "timestamp": frame["timestamp"],
                "instruction": args.instruction,
                "state": frame["state"],
                "velocity": frame["velocity"],
                "effort": frame["effort"],
                "base_vel": frame["base_vel"],
                "images": {"front": frame["front"]},
            } # 4060 向 4090发送的机械臂信息

            t0 = time.perf_counter()
            reply = client.request(request) # reply: 4090返回的动作信息
            round_trip_ms = (time.perf_counter() - t0) * 1000.0
            if not reply.get("ok", False):
                raise RuntimeError(f"policy server returned error: {reply.get('error')}")

            action_chunk = np.asarray(reply["action_chunk"], dtype=np.float32)
            if action_chunk.ndim != 2 or action_chunk.shape[1] != 14:
                raise ValueError(f"bad action chunk shape: {action_chunk.shape}")

            execute_n = min(args.execute_actions, action_chunk.shape[0])
            for i in range(execute_n):
                if stop.stop_requested:
                    break
                action = safety.filter(action_chunk[i])
                # action = action_chunk[i]
                if args.dry_run:
                    rospy.loginfo_throttle(1.0, f"dry-run action left={action[:7]} right={action[7:14]}")
                else:
                    print(action)
                    ros.publish_action(action) # 把action_chunk发布给master/joint_left和master/joint_right，控制机械臂执行动作
                    reset_start_state = action.copy()
                step += 1
                rospy.sleep(1.0 / args.control_rate)
                if rospy.is_shutdown() or stop.stop_requested or step >= args.max_steps:
                    break

            rospy.loginfo(
                "step=%d chunk=%d exec=%d model=%.1fms rtt=%.1fms",
                step,
                action_chunk.shape[0],
                execute_n,
                float(reply.get("model_latency_ms", -1.0)),
                round_trip_ms,
            )
            rate.sleep()
    finally:
        if not stop.force_exit:
            reset_puppet_arms(ros, args, reset_start_state)
        client.close()
        rospy.signal_shutdown("wam remote client finished")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=10000)
    parser.add_argument("--socket-timeout", type=float, default=10.0)
    parser.add_argument("--instruction", required=True)  
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10000)

    parser.add_argument("--img-front-topic", default="/camera_f/color/image_raw")
    parser.add_argument("--puppet-left-topic", default="/puppet/joint_left")
    parser.add_argument("--puppet-right-topic", default="/puppet/joint_right")
    parser.add_argument("--cmd-left-topic", default="/master/joint_left")
    parser.add_argument("--cmd-right-topic", default="/master/joint_right")
    parser.add_argument("--robot-base-topic", default="/odom_raw")
    parser.add_argument("--use-robot-base", action="store_true")

    parser.add_argument("--request-rate", type=float, default=30.0)
    parser.add_argument(
        "--async-policy-protocol",
        action="store_true",
        help="Use decoupled image stream + action_request protocol for dual-GPU async server.",
    )
    parser.add_argument(
        "--image-stream-rate",
        type=float,
        default=30.0,
        help="Rate for sending latest observations in async policy protocol.",
    )
    parser.add_argument("--control-rate", type=float, default=30.0)
    parser.add_argument(
        "--execute-actions",
        type=int,
        default=4,
        help="Number of actions to execute from each returned chunk before re-querying.",
    )
    parser.add_argument("--max-joint-delta", type=float, default=0.05)
    parser.add_argument("--max-gripper-delta", type=float, default=0.01)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=0.07)
    parser.add_argument(
        "--reset-left",
        type=float,
        nargs=7,
        default=[-0.0013, 0.0021, 0.0158, -0.0326, -0.0029, 0.0010, 0.0],
        help="Left puppet reset pose: 6 joints in rad plus gripper in meters.",
    )
    parser.add_argument(
        "--reset-right",
        type=float,
        nargs=7,
        default=[-0.0013, 0.0044, 0.0345, -0.0536, -0.0048, -0.0021, 0.0],
        help="Right puppet reset pose: 6 joints in rad plus gripper in meters.",
    )
    parser.add_argument("--reset-steps", type=int, default=120)
    parser.add_argument("--reset-rate", type=float, default=60.0)
    parser.add_argument("--reset-hold-steps", type=int, default=20)
    parser.add_argument("--init-steps", type=int, default=120)
    parser.add_argument("--init-rate", type=float, default=60.0)
    parser.add_argument("--init-hold-steps", type=int, default=20)
    parser.add_argument("--init-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-reset-on-exit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait-enter", action="store_true")
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
