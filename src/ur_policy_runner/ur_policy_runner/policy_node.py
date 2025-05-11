import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped

import torch
import numpy as np
import yaml
import gym
from rl_games.algos_torch.players import PpoPlayerContinuous
from ament_index_python.packages import get_package_share_directory
import os
import sys
import time


class PolicyNode(Node):
    def __init__(self, policy, obs_dim, device, action_scale, hole_pos):
        super().__init__('ur_policy_node')

        self.policy = policy
        self.obs_dim = obs_dim
        self.device = device
        self.action_scale = action_scale
        self.hole_pos = hole_pos

        self.ee_pose = None
        self.ee_twist = None
        self.prev_action = np.zeros(action_scale.shape[0], dtype=np.float32)

        self._prev_targ_pos = None
        self.use_leaky = True
        self.pos_err_thresh = np.array([0.005, 0.005, 0.005])

        self.create_subscription(TwistStamped, '/cartesian_controller/current_twist', self.twist_cb, 10)
        self.create_subscription(PoseStamped, '/tool0', self.pose_cb, 10)
        self.action_pub = self.create_publisher(PoseStamped, '/cartesian_controller/target_frame', 10)
        self.timer = self.create_timer(0.02, self.run_policy)

        self.start_time = time.time()
        self.timeout_logged = False

    def twist_cb(self, msg):
        v = msg.twist.linear
        w = msg.twist.angular
        self.ee_twist = np.array([v.x, v.y, v.z, w.x, w.y, w.z], dtype=np.float32)

    def pose_cb(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        self.ee_pose = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w])

    def run_policy(self):
        if self.ee_pose is None or self.ee_twist is None:
            now = time.time()
            if not self.timeout_logged and now - self.start_time > 5.0:
                self.get_logger().warn("Haven't received ee_pose or ee_twist for 5 seconds.")
                self.timeout_logged = True  # 避免重复输出
            return
    
        # === 构造观测向量 ===
        if self.ee_pose is None or self.ee_twist is None:
            self.get_logger().warn("ee_pose or ee_twist is not received yet. Waiting for sensor data...")
            obs_vec = np.zeros(self.obs_dim, dtype=np.float32)
            return
        else:
            fingertip_pos = self.ee_pose[:3]
            obs_dict = {
                "fingertip_pos": fingertip_pos,
                "fingertip_pos_rel_fixed": fingertip_pos - self.hole_pos,
                "fingertip_quat": self.ee_pose[3:7],
                "ee_linvel": self.ee_twist[:3],
                "ee_angvel": self.ee_twist[3:6],
                "prev_actions": self.prev_action,
            }
            obs_vec = np.concatenate([v for v in obs_dict.values()]).astype(np.float32)

            if obs_vec.shape[0] < self.obs_dim:
                obs_vec = np.pad(obs_vec, (0, self.obs_dim - obs_vec.shape[0]))
            elif obs_vec.shape[0] > self.obs_dim:
                obs_vec = obs_vec[:self.obs_dim]


        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_tensor = self.policy.get_action(obs_tensor)
        action = action_tensor.squeeze().cpu().numpy()
        self.prev_action = action.copy()

        # === 按元素缩放动作 ===
        if self.action_scale.shape[0] == action.shape[0]:
            scaled_action = action * self.action_scale
        else:
            self.get_logger().error("Action scale shape mismatch! Exiting node.")
            rclpy.shutdown()
            sys.exit(1)
        self.get_logger().info(f"[DEBUG] Action output: {np.round(scaled_action, 3)}")

        curr_pos = self.ee_pose[:3]
        if self._prev_targ_pos is None:
            self._prev_targ_pos = curr_pos.copy()

        targ_pos = self._prev_targ_pos + scaled_action[:3]
        if self.use_leaky:
            pos_err = targ_pos - curr_pos
            pos_err_clip = np.clip(pos_err, -self.pos_err_thresh, self.pos_err_thresh)
            targ_pos = curr_pos + pos_err_clip

        self._prev_targ_pos = targ_pos.copy()

        # === 发布 PoseStamped 目标 ===
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = targ_pos[0]
        msg.pose.position.y = targ_pos[1]
        msg.pose.position.z = targ_pos[2]

        msg.pose.orientation.x = self.ee_pose[3]
        msg.pose.orientation.y = self.ee_pose[4]
        msg.pose.orientation.z = self.ee_pose[5]
        msg.pose.orientation.w = self.ee_pose[6]

        self.action_pub.publish(msg)


def main():
    print("Initializing ROS 2...")
    rclpy.init()

    package_dir = get_package_share_directory('ur_policy_runner')
    config_path = os.path.join(package_dir, 'policy', 'config.yaml')
    model_path = os.path.join(package_dir, 'policy', 'nn', 'industreal_policy_insert_pegs.pth')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    obs_dim = config["task"]["env"]["numObservations"]
    act_dim = config["task"]["env"]["numActions"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config["train"]["params"]["config"]["env_info"] = {
        "observation_space": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        "action_space": gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
    }
    config["train"]["params"]["config"]["device_name"] = device.type

    scale_config_path = os.path.join(package_dir, 'policy', 'PegInsertion.yaml')

    with open(scale_config_path, 'r') as f:
        scale_config = yaml.safe_load(f)

    scale_list = scale_config.get("action_scale", [0.0006])
    if len(scale_list) == 1:
        scale_list = scale_list * act_dim
    action_scale = np.array(scale_list)

    # 读取hole位置，3维
    hole_pos_list = scale_config.get("hole_position")
    if len(hole_pos_list) != 3:
        raise ValueError("Expected 3D hole_position in PegInsertion.yaml")
    hole_pos = np.array(hole_pos_list, dtype=np.float32)



    policy = PpoPlayerContinuous(params=config["train"]["params"])
    policy.restore(model_path)
    policy.reset()

    node = PolicyNode(policy, obs_dim, device, action_scale, hole_pos)
    print("Starting ROS 2 spin loop...")
    rclpy.spin(node)
    print("Node shutdown.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
