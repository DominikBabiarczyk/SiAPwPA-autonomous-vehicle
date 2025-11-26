import random
from collections import deque
from typing import Optional, Tuple, List
# Opcjonalny import h5py do obsługi pliku replay_buffer.h5
try:
	import h5py  # type: ignore
	_HAS_H5PY = True
except ImportError:
	_HAS_H5PY = False

import torch
import torch.nn as nn
import torch.optim as optim
from .q_network import QNetwork


class ReplayBuffer:
	"""Prosty bufor doświadczeń do przechowywania i próbkowania przejść.

	Każde przejście: (state, action, reward, next_state, done)
	gdzie:
	  state: (vec1, vec2, vec3, position, yaw)
	  action: (velocity_idx, steering_idx)
	  reward: float
	  next_state: jak state
	  done: bool (koniec epizodu)
	"""

	def __init__(self, capacity: int = 50_000):
		self.capacity = capacity
		self.buffer = deque(maxlen=capacity)

	def __len__(self):
		return len(self.buffer)

	def push(
		self,
		state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
		action: Tuple[int, int],
		reward: float,
		next_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
		done: bool,
	):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size: int):
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = zip(*batch)

		# Rozpakowanie i złączenie
		vec1, vec2, vec3, pos, yaw = zip(*states)
		n_vec1, n_vec2, n_vec3, n_pos, n_yaw = zip(*next_states)

		return (
			torch.stack(vec1),
			torch.stack(vec2),
			torch.stack(vec3),
			torch.stack(pos),
			torch.stack(yaw),
			torch.tensor([a[0] for a in actions], dtype=torch.long),
			torch.tensor([a[1] for a in actions], dtype=torch.long),
			torch.tensor(rewards, dtype=torch.float32),
			torch.stack(n_vec1),
			torch.stack(n_vec2),
			torch.stack(n_vec3),
			torch.stack(n_pos),
			torch.stack(n_yaw),
			torch.tensor(dones, dtype=torch.float32),
		)

	def load_from_hdf5(self, h5_path: str, limit: Optional[int] = None) -> int:
		"""Wczytuje przejścia z pliku HDF5 i dodaje je do bufora.

		Parametry:
		- h5_path: ścieżka do pliku HDF5.
		- limit: maksymalna liczba przejść do wczytania (None = wszystkie).

		Zwraca: liczbę faktycznie dodanych przejść.

		Założenie: struktura datasetów zgodna z zapisem w Learning_management:
		  vec1, vec2, vec3, pos, yaw,
		  next_vec1, next_vec2, next_vec3, next_pos, next_yaw,
		  action_vel, action_str, reward, done
		"""
		if not _HAS_H5PY:
			print("[ReplayBuffer] h5py nie jest dostępne - pomijam wczytanie.")
			return 0
		if not h5py.is_hdf5(h5_path):
			print(f"[ReplayBuffer] Plik {h5_path} nie jest prawidłowym plikiem HDF5.")
			return 0
		added = 0
		with h5py.File(h5_path, 'r') as f:
			length = f['reward'].shape[0]
			if limit is not None:
				length = min(length, limit)
			for i in range(length):
				# Pobierz część stanu
				vec1 = torch.from_numpy(f['vec1'][i]).float()
				vec2 = torch.from_numpy(f['vec2'][i]).float()
				vec3 = torch.from_numpy(f['vec3'][i]).float()
				pos = torch.from_numpy(f['pos'][i]).float()
				yaw = torch.from_numpy(f['yaw'][i]).float()
				# Następny stan
				n_vec1 = torch.from_numpy(f['next_vec1'][i]).float()
				n_vec2 = torch.from_numpy(f['next_vec2'][i]).float()
				n_vec3 = torch.from_numpy(f['next_vec3'][i]).float()
				n_pos = torch.from_numpy(f['next_pos'][i]).float()
				n_yaw = torch.from_numpy(f['next_yaw'][i]).float()
				# Akcja / nagroda / done
				a_vel = int(f['action_vel'][i])
				a_str = int(f['action_str'][i])
				r = float(f['reward'][i])
				d = bool(f['done'][i])
				self.push((vec1, vec2, vec3, pos, yaw), (a_vel, a_str), r, (n_vec1, n_vec2, n_vec3, n_pos, n_yaw), d)
				added += 1
		print(f"[ReplayBuffer] Wczytano {added} przejść z {h5_path}")
		return added

	def sample_hdf5_batch(self, h5_path: str, batch_size: int):
		"""Losowo próbuje batch bez ładowania całego pliku do bufora.
		Zwraca tuplę identyczną jak sample(), ale dane pobrane bezpośrednio z pliku.
		"""
		if not _HAS_H5PY:
			raise RuntimeError("h5py nie jest dostępne.")
		with h5py.File(h5_path, 'r') as f:
			length = f['reward'].shape[0]
			if batch_size > length:
				raise ValueError(f"Żądany batch_size {batch_size} > liczba rekordów {length}")
			indices = random.sample(range(length), batch_size)
			vec1 = torch.stack([torch.from_numpy(f['vec1'][i]).float() for i in indices])
			vec2 = torch.stack([torch.from_numpy(f['vec2'][i]).float() for i in indices])
			vec3 = torch.stack([torch.from_numpy(f['vec3'][i]).float() for i in indices])
			pos = torch.stack([torch.from_numpy(f['pos'][i]).float() for i in indices])
			yaw = torch.stack([torch.from_numpy(f['yaw'][i]).float() for i in indices])
			n_vec1 = torch.stack([torch.from_numpy(f['next_vec1'][i]).float() for i in indices])
			n_vec2 = torch.stack([torch.from_numpy(f['next_vec2'][i]).float() for i in indices])
			n_vec3 = torch.stack([torch.from_numpy(f['next_vec3'][i]).float() for i in indices])
			n_pos = torch.stack([torch.from_numpy(f['next_pos'][i]).float() for i in indices])
			n_yaw = torch.stack([torch.from_numpy(f['next_yaw'][i]).float() for i in indices])
			a_vel = torch.tensor([int(f['action_vel'][i]) for i in indices], dtype=torch.long)
			a_str = torch.tensor([int(f['action_str'][i]) for i in indices], dtype=torch.long)
			rewards = torch.tensor([float(f['reward'][i]) for i in indices], dtype=torch.float32)
			dones = torch.tensor([float(f['done'][i]) for i in indices], dtype=torch.float32)
			return (
				vec1,
				vec2,
				vec3,
				pos,
				yaw,
				a_vel,
				a_str,
				rewards,
				n_vec1,
				n_vec2,
				n_vec3,
				n_pos,
				n_yaw,
				dones,
		)


class DQNTrainer:
	"""
		Klasa odpowiedzialna za trenowanie sieci QNetwork.
	"""

	def __init__(
		self,
		q_network: Optional[QNetwork],
		buffer: ReplayBuffer,
		lr: float = 1e-3,
		gamma: float = 0.99,
		batch_size: int = 32,
		device: Optional[str] = None,
	):
		self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.q_network = q_network or QNetwork()
		self.q_network.to(self.device)
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
		self.gamma = gamma
		self.batch_size = batch_size
		self.replay_buffer = buffer
		self.loss_fn = nn.MSELoss()

	def load_buffer_from_hdf5(self, h5_path: str, limit: Optional[int] = None) -> int:
		"""Ładuje przejścia z pliku HDF5 do wewnętrznego ReplayBuffer.
		Zwraca liczbę dodanych rekordów."""
		return self.replay_buffer.load_from_hdf5(h5_path, limit)

	def select_action(self, state, epsilon: float = 0.1) -> Tuple[int, int]:
		"""Wybór akcji metodą epsilon-greedy.
		state: (vec1, vec2, vec3, position, yaw) - każdy element tensor shape (dim,)
		Zwraca: (velocity_idx, steering_idx)
		"""
		if random.random() < epsilon:
			velocity_idx = random.randint(0, 10)
			steering_idx = random.randint(0, 8)
			return velocity_idx, steering_idx

		vec1, vec2, vec3, pos, yaw = [s.to(self.device).unsqueeze(0) for s in state]
		with torch.no_grad():
			velocity_logits, steering_logits = self.q_network(vec1, vec2, vec3, pos, yaw)
			# Argmax po wymiarze akcji
			velocity_idx = torch.argmax(velocity_logits, dim=-1).item()
			steering_idx = torch.argmax(steering_logits, dim=-1).item()
		return velocity_idx, steering_idx

	def add_transition(
		self,
		state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
		action: Tuple[int, int],
		reward: float,
		next_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
		done: bool,
	):
		# Przeniesienie wszystkiego na CPU do bufora (bufor przechowuje CPU tensory)
		cpu_state = tuple(s.detach().cpu() for s in state)
		cpu_next_state = tuple(s.detach().cpu() for s in next_state)
		self.replay_buffer.push(cpu_state, action, reward, cpu_next_state, done)

	def can_train(self) -> bool:
		return len(self.replay_buffer) >= self.batch_size

	def train_step(self) -> Optional[dict]:
		"""Trenuje sieć na jednym batchu z bufora.
		Zwraca słownik metryk lub None jeśli za mało danych."""
		if not self.can_train():
			return None

		(
			vec1,
			vec2,
			vec3,
			pos,
			yaw,
			act_vel,
			act_str,
			rewards,
			n_vec1,
			n_vec2,
			n_vec3,
			n_pos,
			n_yaw,
			dones,
		) = self.replay_buffer.sample(self.batch_size)

		vec1 = vec1.to(self.device)
		vec2 = vec2.to(self.device)
		vec3 = vec3.to(self.device)
		pos = pos.to(self.device)
		yaw = yaw.to(self.device)
		act_vel = act_vel.to(self.device)
		act_str = act_str.to(self.device)
		rewards = rewards.to(self.device)
		n_vec1 = n_vec1.to(self.device)
		n_vec2 = n_vec2.to(self.device)
		n_vec3 = n_vec3.to(self.device)
		n_pos = n_pos.to(self.device)
		n_yaw = n_yaw.to(self.device)
		dones = dones.to(self.device)

		current_vel_logits, current_str_logits = self.q_network(
			vec1, vec2, vec3, pos, yaw
		)

		batch_indices = torch.arange(self.batch_size, device=self.device)
		current_q_vel = current_vel_logits[batch_indices, act_vel]
		current_q_str = current_str_logits[batch_indices, act_str]

		with torch.no_grad():
			next_vel_logits, next_str_logits = self.q_network(
				n_vec1, n_vec2, n_vec3, n_pos, n_yaw
			)
			max_next_q_vel = torch.max(next_vel_logits, dim=-1).values
			max_next_q_str = torch.max(next_str_logits, dim=-1).values
			target_q_vel = rewards + self.gamma * max_next_q_vel * (1 - dones)
			target_q_str = rewards + self.gamma * max_next_q_str * (1 - dones)

		loss_vel = self.loss_fn(current_q_vel, target_q_vel)
		loss_str = self.loss_fn(current_q_str, target_q_str)
		loss = loss_vel + loss_str

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
		self.optimizer.step()

		return {
			"loss_total": loss.item(),
			"loss_velocity": loss_vel.item(),
			"loss_steering": loss_str.item(),
			"buffer_size": len(self.replay_buffer),
		}

	def save(self, path: str):
		torch.save(self.q_network.state_dict(), path)

	def load(self, path: str):
		state_dict = torch.load(path, map_location=self.device)
		self.q_network.load_state_dict(state_dict)


if __name__ == "__main__":
	MODEL_PATH = "camera/Qlerning/results/qnetwork_model.pth"
	trainer = DQNTrainer()
	trainer.load_buffer_from_hdf5('camera/Qlerning/results/replay_buffer.h5', limit=5000)

	# Wczytaj model jeśli istnieje
	import os
	if os.path.exists(MODEL_PATH):
		print(f"[DQNTrainer] Wczytuję model z {MODEL_PATH}")
		trainer.load(MODEL_PATH)
	else:
		print(f"[DQNTrainer] Brak pliku {MODEL_PATH}, trenuję od zera.")

	for step in range(1000):
		metrics = trainer.train_step()
		if metrics:
			print(step, metrics)

	# Zapisz model po treningu
	trainer.save(MODEL_PATH)
	print(f"[DQNTrainer] Model zapisany do {MODEL_PATH}")
