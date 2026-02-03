"""Inference script for the trained QNetwork.

Loads the saved model weights from `camera/Qlerning/results/qnetwork_model.pth`
and runs a loop that:
  1. Spins ROS2 nodes to receive sensor + odometry data.
  2. Builds the state tensors expected by QNetwork.
  3. Performs a forward pass to obtain Q-values for velocity & steering.
  4. Selects the greedy (argmax) action indices.
  5. Optionally publishes the action to `/cmd_vel` using `SetAction`.

You can run without actuating (just print actions) by passing `--dry-run`.

Usage examples:
  python3 -m camera.Qlerning.inference --steps 200
  python3 -m camera.Qlerning.inference --steps 1000 --device cpu --dry-run

Exposed CLI flags:
  --steps        How many decision steps to run (default: 300)
  --model        Path to model weights (default: camera/Qlerning/results/qnetwork_model.pth)
  --device       Device override: cpu|cuda (auto-detect by default)
  --dry-run      Do not publish motions, only print chosen actions
  --sleep        Seconds to sleep between action decisions (default: 0.1)

Edge cases handled:
  - Missing model file: warns and uses randomly initialized network.
  - Missing sensor data: fills zero vectors.
  - CUDA unavailable but requested: falls back to CPU.

"""

import argparse
import os
import time
from typing import Tuple

import torch
import rclpy

from jetracer.nodes.qlearning.train_process.q_network import QNetwork
from jetracer.nodes.qlearning.state_loader import StateLoader
from jetracer.nodes.qlearning.set_action import SetAction
from jetracer.nodes.sensors.get_odometry import OdometrySubscriber

def _convert_state(raw_state: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Convert raw state dict into tensors expected by QNetwork.

	raw_state keys:
	  sensor: ndarray-like shape (3, N, ...) or None
	  position: (x, y)
	  yaw: float

	Returns: (vec1[100], vec2[100], vec3[100], pos[2], yaw[1]) float32 tensors.
	If sensor rows shorter than 100 elements they're padded with zeros.
	"""
	sensor = raw_state.get('sensor')
	if sensor is None:
		vec1 = torch.zeros(100, dtype=torch.float32)
		vec2 = torch.zeros(100, dtype=torch.float32)
		vec3 = torch.zeros(100, dtype=torch.float32)
	else:
		def to_vec(row):
			flat = torch.as_tensor(row, dtype=torch.float32).flatten()
			if flat.numel() >= 100:
				return flat[:100]
			return torch.cat([flat, torch.zeros(100 - flat.numel(), dtype=torch.float32)])
		vec1 = to_vec(sensor[0])
		vec2 = to_vec(sensor[1])
		vec3 = to_vec(sensor[2])
	x, y = raw_state.get('position', (0.0, 0.0))
	yaw = raw_state.get('yaw', 0.0)
	pos = torch.tensor([x, y], dtype=torch.float32)
	ywa = torch.tensor([yaw], dtype=torch.float32)
	return vec1, vec2, vec3, pos, ywa


def choose_action(q_network: QNetwork, state_tensors, device: torch.device):
	"""Run forward pass and select greedy action indices.
	Returns: (velocity_idx, steering_idx)
	"""
	vec1, vec2, vec3, pos, ywa = state_tensors
	with torch.no_grad():
		v_q, s_q = q_network(
			vec1.unsqueeze(0).to(device),
			vec2.unsqueeze(0).to(device),
			vec3.unsqueeze(0).to(device),
			pos.unsqueeze(0).to(device),
			ywa.unsqueeze(0).to(device),
		)
		vel_idx = int(torch.argmax(v_q, dim=-1).item())
		str_idx = int(torch.argmax(s_q, dim=-1).item())
	return vel_idx, str_idx


def run_inference(steps: int, model_path: str, device_str: str, dry_run: bool, sleep_sec: float):
	# Resolve device
	if device_str:
		if device_str == 'cuda' and not torch.cuda.is_available():
			print('[Inference] Requested CUDA but not available – using CPU.')
			device = torch.device('cpu')
		else:
			device = torch.device(device_str)
	else:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'[Inference] Using device: {device}')

	# ROS2 init
	rclpy.init()
	odom_sub = OdometrySubscriber()
	state_loader = StateLoader(odom_sub)
	action_commander = SetAction() if not dry_run else None

	# Model
	q_network = QNetwork().to(device)
	if os.path.exists(model_path):
		try:
			state = torch.load(model_path, map_location=device)
			q_network.load_state_dict(state)
			q_network.eval()
			print(f'[Inference] Loaded model weights from {model_path}')
		except Exception as e:
			print(f'[Inference] Failed to load model ({e}) – continuing with random weights.')
	else:
		print(f'[Inference] Model file not found: {model_path} – using randomly initialized weights.')

	# Main loop
	for step in range(steps):
		# Spin nodes minimally to update data
		rclpy.spin_once(odom_sub, timeout_sec=0.05)
		if action_commander is not None:
			rclpy.spin_once(action_commander.node, timeout_sec=0.0)

		raw_state = state_loader.get_state()
		state_tensors = _convert_state(raw_state)
		vel_idx, str_idx = choose_action(q_network, state_tensors, device)

		# Log action & basic state info
		x, y = raw_state.get('position', (0.0, 0.0))
		yaw = raw_state.get('yaw', 0.0)
		print(f"[Step {step}] pos=({x:.2f},{y:.2f}) yaw={yaw:.2f} -> action vel_idx={vel_idx} str_idx={str_idx}")

		if not dry_run and action_commander is not None:
			action_commander.go_vehicle(vel_idx, str_idx)

		time.sleep(sleep_sec)

	# Cleanup
	if action_commander is not None:
		action_commander.node.destroy_node()
	odom_sub.destroy_node()
	rclpy.shutdown()
	print('[Inference] Finished.')


def parse_args():
	parser = argparse.ArgumentParser(description='Run inference with trained QNetwork.')
	parser.add_argument('--steps', type=int, default=300, help='Number of inference steps to run.')
	parser.add_argument('--model', type=str, default='camera/Qlerning/results/qnetwork_model.pth', help='Path to model weights file.')
	parser.add_argument('--device', type=str, default='', help='Force device (cpu|cuda). If empty auto-detect.')
	parser.add_argument('--dry-run', action='store_true', help='Do not publish actions, only print them.')
	parser.add_argument('--sleep', type=float, default=0.1, help='Sleep seconds between decisions.')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	run_inference(
		steps=args.steps,
		model_path=args.model,
		device_str=args.device,
		dry_run=args.dry_run,
		sleep_sec=args.sleep,
	)

