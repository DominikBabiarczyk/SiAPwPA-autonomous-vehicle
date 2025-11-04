from camera.Qlerning.awarding_prizes import AwardingPrizes
from camera.Qlerning.position_progressor import PositionProgressor
from camera.Qlerning.set_action import SetAction
from camera.Qlerning.state_loader import StateLoader
from camera.Qlerning.train_process.q_network import QNetwork
from camera.Qlerning.train_process.train_method import ReplayBuffer
from camera.get_odometry import OdometrySubscriber
import rclpy
import numpy as np
import torch
import subprocess
import os
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

class LearningManagement:
    def __init__(self):
        self.odometry_subscriber = OdometrySubscriber()
        self.position_progressor = PositionProgressor(self.odometry_subscriber)
        self.awarding_prizes = AwardingPrizes(self.odometry_subscriber)
        self.set_action = SetAction()
        self.state_loader = StateLoader(self.odometry_subscriber)
        self.q_network = QNetwork()
        self.m_memory = ReplayBuffer(10000)
        self.cmd = [
            "gz", "service", "-s", "/world/Trapezoid/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--req", 'name: "saye_1" position { x: 1.0 y: 0.4 z: 0.1 } orientation { w: 1.0 }',
            "--timeout", "3000"
        ]
        # Inicjalizacja pliku HDF5 do trwałego zapisu przejść
        self.h5_path = "replay_buffer.h5"
        if _HAS_H5PY:
            self._init_hdf5()
        else:
            print("[HDF5] h5py nie jest zainstalowane - pomijam zapis do pliku. Zainstaluj pakiet aby włączyć funkcję.")

    def _convert_state(self, raw_state):
        """Konwersja słownika stanu na krotkę tensorów oczekiwaną przez QNetwork.
        raw_state: {'sensor': ndarray|None, 'position': (x,y), 'yaw': yaw}
        Zwraca: (vec1[100], vec2[100], vec3[100], pos[2], yaw[1]) jako tensory float32.
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

    # ---------------- HDF5 zapisywanie -----------------
    def _init_hdf5(self):
        """Tworzy strukturę pliku HDF5 jeśli nie istnieje."""
        if os.path.exists(self.h5_path):
            return  # już istnieje
        with h5py.File(self.h5_path, 'w') as f:
            # Każde pole ma pierwszy wymiar rozszerzalny (liczba przejść)
            def dset(name, shape, dtype='float32'):
                f.create_dataset(name, shape=(0,) + shape, maxshape=(None,) + shape, dtype=dtype, chunks=True, compression='gzip', compression_opts=4)
            dset('vec1', (100,))
            dset('vec2', (100,))
            dset('vec3', (100,))
            dset('pos', (2,))
            dset('yaw', (1,))
            dset('next_vec1', (100,))
            dset('next_vec2', (100,))
            dset('next_vec3', (100,))
            dset('next_pos', (2,))
            dset('next_yaw', (1,))
            dset('action_vel', (), dtype='int64')
            dset('action_str', (), dtype='int64')
            dset('reward', (), dtype='float32')
            dset('done', (), dtype='uint8')
        print(f"[HDF5] Utworzono plik {self.h5_path}")

    def _save_transition_hdf5(self, state_tensors, action, reward, next_state_tensors, done):
        """Zapisuje pojedyncze przejście do pliku HDF5 (append)."""
        if not _HAS_H5PY:
            return
        vec1, vec2, vec3, pos, yaw = state_tensors
        n_vec1, n_vec2, n_vec3, n_pos, n_yaw = next_state_tensors
        with h5py.File(self.h5_path, 'a') as f:
            current = f['reward'].shape[0]
            new_size = current + 1
            # Resize wszystkich datasetów
            for name in f.keys():
                f[name].resize(new_size, axis=0)
            f['vec1'][current] = vec1.cpu().numpy()
            f['vec2'][current] = vec2.cpu().numpy()
            f['vec3'][current] = vec3.cpu().numpy()
            f['pos'][current] = pos.cpu().numpy()
            f['yaw'][current] = yaw.cpu().numpy()
            f['next_vec1'][current] = n_vec1.cpu().numpy()
            f['next_vec2'][current] = n_vec2.cpu().numpy()
            f['next_vec3'][current] = n_vec3.cpu().numpy()
            f['next_pos'][current] = n_pos.cpu().numpy()
            f['next_yaw'][current] = n_yaw.cpu().numpy()
            f['action_vel'][current] = int(action[0])
            f['action_str'][current] = int(action[1])
            f['reward'][current] = float(reward)
            f['done'][current] = 1 if done else 0
        # Można dodać prosty log co N zapisów
        if (new_size % 1000) == 0:
            print(f"[HDF5] Zapisano {new_size} przejść do {self.h5_path}")
    

    def launch_qlearning(self):
        commander = SetAction()
        for i in range(10000):
            rclpy.spin_once(commander.node, timeout_sec=0.05)
            rclpy.spin_once(self.odometry_subscriber, timeout_sec=0.0)

            raw_state = self.state_loader.get_state()
            state_tensors = self._convert_state(raw_state)
            # print(f"State step {i}: {raw_state}")

            with torch.no_grad():
                v_q, s_q = self.q_network(
                    state_tensors[0].unsqueeze(0),
                    state_tensors[1].unsqueeze(0),
                    state_tensors[2].unsqueeze(0),
                    state_tensors[3].unsqueeze(0),
                    state_tensors[4].unsqueeze(0),
                )
                best_action_velocity = int(torch.argmax(v_q, dim=-1).item())
                best_action_angular = int(torch.argmax(s_q, dim=-1).item())

            commander.go_vehicle(best_action_velocity, best_action_angular)

            reward, collision, target = self.awarding_prizes.check_and_award()

            # STAN PO AKCJI
            raw_next_state = self.state_loader.get_state()
            next_state_tensors = self._convert_state(raw_next_state)

            done = False

            # ZAPIS DO PAMIĘCI + HDF5
            # self.m_memory.push(state_tensors, (best_action_velocity, best_action_angular), reward, next_state_tensors, done)
            self._save_transition_hdf5(state_tensors, (best_action_velocity, best_action_angular), reward, next_state_tensors, done)

            if collision or target:
                move = subprocess.run(self.cmd, capture_output=True, text=True)

            print(f"Action step {i}: vel={best_action_velocity}, steer={best_action_angular}")
            print(f"Reward step {i}: {reward}")

            progress = self.position_progressor.get_position_progress()
            print(f"Progress step {i}: {progress}")
        commander.node.destroy_node()

    def check_progress(self):
        commander = SetAction()

        for i in range(10000):
            rclpy.spin_once(commander.node, timeout_sec=0.1)

            state = self.state_loader.get_state()
            print(f"State step {i}: {state}")
            v1 = torch.from_numpy(state["sensor"][0]).float()
            v2 = torch.from_numpy(state["sensor"][1]).float()
            v3 = torch.from_numpy(state["sensor"][2]).float()
            pos = torch.from_numpy(np.array(state["position"])).float()
            yaw = torch.tensor([state["yaw"]], dtype=torch.float32)

            action_velocity, action_angular = self.q_network.forward(
                v1.unsqueeze(0),
                v2.unsqueeze(0),
                v3.unsqueeze(0),
                pos.unsqueeze(0),
                yaw.unsqueeze(0)
            )
            best_action_velocity = np.argmax(action_velocity.detach().numpy())
            best_action_angular = np.argmax(action_angular.detach().numpy())

            best_action_velocity, best_action_angular = (7, 4)

            commander.go_vehicle(10, 4)

            reward, collision, target = self.awarding_prizes.check_and_award()

            next_state_raw = self.state_loader.get_state()

            print(f"State after action step {i}: {next_state_raw}")

            done = False
            # Konwersja stanu i następnego stanu do tensorów przed zapisem
            state_tensors = self._convert_state(state)
            next_state_tensors = self._convert_state(next_state_raw)
            # self.m_memory.push(state_tensors, (best_action_velocity, best_action_angular), reward, next_state_tensors, done)
            self._save_transition_hdf5(state_tensors, (best_action_velocity, best_action_angular), reward, next_state_tensors)

            if collision or target:
                move = subprocess.run(self.cmd, capture_output=True, text=True)
    
        commander.node.destroy_node()


        return self.m_memory

if __name__ == '__main__':
    rclpy.init()
    manager = LearningManagement()
    memory = manager.launch_qlearning()
    # print(memory)
    rclpy.shutdown()