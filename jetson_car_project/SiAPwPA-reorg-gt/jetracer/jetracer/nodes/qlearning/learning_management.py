from jetracer.nodes.qlearning.awarding_prizes import AwardingPrizes
from jetracer.nodes.qlearning.position_progressor import PositionProgressor
from jetracer.nodes.qlearning.set_action import SetAction
from jetracer.nodes.qlearning.state_loader import StateLoader
from jetracer.nodes.qlearning.train_process.q_network import QNetwork
from jetracer.nodes.qlearning.train_process.train_method import DQNTrainer, ReplayBuffer
from jetracer.nodes.sensors.get_odometry import OdometrySubscriber
import rclpy
import numpy as np
import torch
import subprocess
import os
import shutil
import glob
from datetime import datetime
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

def log(i, position_progressor, reward, odometry_subscriber):
    progress = position_progressor.get_position_progress()

    print(f"Reward step {i}: {reward}")
    print(f"Progress step {i}: {progress}")
    print(f"Actual position: {odometry_subscriber.get_actual_position()}")
    print("--------------------------")

class LearningManagement:
    def __init__(self):
        self.odometry_subscriber = OdometrySubscriber()
        self.position_progressor = PositionProgressor(self.odometry_subscriber)
        self.awarding_prizes = AwardingPrizes(self.odometry_subscriber)
        self.set_action = SetAction()
        self.state_loader = StateLoader(self.odometry_subscriber)
        self.q_network = QNetwork()
        self.m_memory = ReplayBuffer(10000)
        self.trainer = DQNTrainer(self.q_network, self.m_memory)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "camera/Qlerning/results/qnetwork_model.pth"
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                self.q_network.load_state_dict(state)
                print(f"[QNetwork] Wczytano model z {model_path}")
            except Exception as e:
                print(f"[QNetwork] Nie udało się wczytać {model_path}: {e}")
        else:
            print(f"[QNetwork] Brak pliku {model_path} — używam nowo zainicjalizowanej sieci.")
    
        self.cmd = [
            "gz", "service", "-s", "/world/Trapezoid/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--req", 'name: "saye_1" position { x: 1.0 y: 0.4 z: 0.1 } orientation { w: 1.0 }',
            "--timeout", "3000"
        ]

        self.cmd = ["gz",  "sim",  "-r",  "worlds/Trapezoid/worlds/Trapezoid.world"]
        self.cmd = [
            "gz", "service", "-s", "/world/Trapezoid/control",
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--req", "reset { all: true }",
            "--timeout", "3000",
        ]

        self.results_dir = "camera/Qlerning/results/data"
        os.makedirs(self.results_dir, exist_ok=True)
        pattern = os.path.join(self.results_dir, "replay_buffer_*.h5")
        existing = glob.glob(pattern)
        max_idx = 0
        for p in existing:
            name = os.path.basename(p)
            parts = name.replace('.h5','').split('_')
            if len(parts) >= 3:
                try:
                    idx = int(parts[-1])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        next_idx = max_idx + 1
        self.h5_path = os.path.join(self.results_dir, f"replay_buffer_{next_idx}.h5")
        self.h5_latest = os.path.join(self.results_dir, "replay_buffer_latest.h5")
        self._h5_save_count = 0
        self._h5_copy_every = 100
        # Plik logów metryk treningowych
        self.metrics_log_path = os.path.join("camera", "Qlerning", "results", "training_metrics.log")
        try:
            os.makedirs(os.path.dirname(self.metrics_log_path), exist_ok=True)
            if not os.path.exists(self.metrics_log_path):
                with open(self.metrics_log_path, 'w') as f:
                    f.write("timestamp,phase,epoch,step,total_steps,loss_total,loss_velocity,loss_steering,buffer_size,extra\n")
        except Exception as e:
            print(f"[MetricsLog] Nie udało się przygotować pliku log: {e}")
        if _HAS_H5PY:
            self._init_hdf5()
        else:
            print("[HDF5] h5py nie jest zainstalowane - pomijam zapis do pliku. Zainstaluj pakiet aby włączyć funkcję.")
    
    def teleport_and_reset(self):
        """Teleportuje model na (1.0,0.4,0.1) i jeśli tryb odometrii to 'relative_zero' zeruje logiczną odometrię.

        Sekwencja:
        1. Wywołanie usługi set_pose.
        2. Jedno spin_once aby otrzymać nową odometrię w miejscu teleporu.
        3. reset_origin (tylko w trybie relative_zero) tak aby /relative_odometry startowało od (0,0).
        """
        # self.odometry_subscriber.reset_odometry()

        move = subprocess.run(self.cmd, capture_output=True, text=True)
        rclpy.spin_once(self.odometry_subscriber, timeout_sec=0.2)
        return move



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

    def _init_hdf5(self):
        """Tworzy strukturę pliku HDF5 jeśli nie istnieje."""
        if os.path.exists(self.h5_path):
            return  # już istnieje
        with h5py.File(self.h5_path, 'w') as f:
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
        # zwiększ licznik zapisów i okresowo skopiuj plik do wersji 'latest'
        self._h5_save_count += 1
        if (new_size % 1000) == 0:
            print(f"[HDF5] Zapisano {new_size} przejść do {self.h5_path}")
        if self._h5_save_count % self._h5_copy_every == 0:
            try:
                shutil.copyfile(self.h5_path, self.h5_latest)
                print(f"[HDF5] Zaktualizowano latest: {self.h5_latest}")
            except Exception as e:
                print(f"[HDF5] Nie udało się skopiować do latest: {e}")
    

    def launch_qlearning(self):
        commander = SetAction()
        for i in range(500):
            rclpy.spin_once(commander.node, timeout_sec=0.05)
            rclpy.spin_once(self.odometry_subscriber, timeout_sec=0.0)

            raw_state = self.state_loader.get_state()
            state_tensors = self._convert_state(raw_state)

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

            raw_next_state = self.state_loader.get_state()
            next_state_tensors = self._convert_state(raw_next_state)

            done = False

            if collision or target or i % 50 == 0:
                done = True
                print("RESETTTTT")
                self.teleport_and_reset()

            self.m_memory.push(state_tensors, (best_action_velocity, best_action_angular), reward, next_state_tensors, done)
            # self._save_transition_hdf5(state_tensors, (best_action_velocity, best_action_angular), reward, next_state_tensors, done)

            # print(f"Action step {i}: vel={best_action_velocity}, steer={best_action_angular}")
            log(i, self.position_progressor, reward, self.odometry_subscriber)

            # Możliwe logowanie surowych kroków (opcjonalne odkomentowanie)
            # self._log_metric(phase="collect", epoch=-1, step=i, loss_dict=None, buffer_size=len(self.m_memory), extra=f"reward={reward}")
        commander.node.destroy_node()

    def run_qlearning_epochs(
        self,
        epochs: int = 100,
        train_steps_per_epoch: int = 200,
        save_model_each_epoch: bool = True,
        model_path: str = "camera/Qlerning/results/qnetwork_model.pth",
    ):
        """Wielokrotne uruchomienie launch_qlearning oraz trening po każdym epizodzie.

        Dla każdego epizodu:
        1. Zbieranie 500 akcji (launch_qlearning)
        2. Trening sieci na zebranych przejściach (maksymalnie train_steps_per_epoch aktualizacji)
        3. Wypisanie metryk: liczba update'ów, średnie straty, rozmiar bufora
        """
        for ep in range(epochs):
            print(f"\n[EPOCH {ep+1}/{epochs}] --- ZBIERANIE DANYCH ---")
            self.launch_qlearning()

            print(f"[EPOCH {ep+1}] Rozpoczynam trening (max {train_steps_per_epoch} kroków)")
            total_losses = []
            vel_losses = []
            str_losses = []
            updates = 0
            for t in range(train_steps_per_epoch):
                metrics = self.trainer.train_step()
                if metrics is None:  # za mało danych w buforze
                    if updates == 0:
                        print(f"[EPOCH {ep+1}] Za mało danych do treningu (buffer_size={len(self.m_memory)})")
                        self._log_metric(phase="train", epoch=ep+1, step=t, loss_dict=None, buffer_size=len(self.m_memory), extra="insufficient_data")
                    break
                total_losses.append(metrics["loss_total"])
                vel_losses.append(metrics["loss_velocity"])
                str_losses.append(metrics["loss_steering"])
                updates += 1
                # Log per-update (można wyłączyć jeśli za dużo danych):
                self._log_metric(
                    phase="train",
                    epoch=ep+1,
                    step=t,
                    loss_dict=metrics,
                    buffer_size=metrics.get("buffer_size", len(self.m_memory)),
                    extra="update"
                )
            if updates > 0:
                avg_loss = sum(total_losses) / updates
                avg_vel = sum(vel_losses) / updates
                avg_str = sum(str_losses) / updates
                print(
                    f"[EPOCH {ep+1}] Trening zakończony: updates={updates}, avg_loss={avg_loss:.4f}, avg_vel={avg_vel:.4f}, avg_str={avg_str:.4f}, buffer_size={len(self.m_memory)}"
                )
                self._log_metric(
                    phase="epoch_summary",
                    epoch=ep+1,
                    step=updates,
                    loss_dict={"loss_total": avg_loss, "loss_velocity": avg_vel, "loss_steering": avg_str},
                    buffer_size=len(self.m_memory),
                    extra="summary"
                )
            # zapis modelu po epizodzie
            if save_model_each_epoch and updates > 0:
                try:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.trainer.save(model_path)
                    print(f"[EPOCH {ep+1}] Zapisano model do {model_path}")
                    self._log_metric(
                        phase="model_save",
                        epoch=ep+1,
                        step=updates,
                        loss_dict=None,
                        buffer_size=len(self.m_memory),
                        extra="saved"
                    )
                except Exception as e:
                    print(f"[EPOCH {ep+1}] Nie udało się zapisać modelu: {e}")
                    self._log_metric(
                        phase="model_save_fail",
                        epoch=ep+1,
                        step=updates,
                        loss_dict=None,
                        buffer_size=len(self.m_memory),
                        extra=str(e)
                    )
        print("\n[RUN] Wszystkie epizody zakończone.")

    def _log_metric(self, phase: str, epoch: int, step: int, loss_dict, buffer_size: int, extra: str):
        """Zapisuje pojedynczy wpis metryki do pliku log.
        Format CSV: timestamp,phase,epoch,step,total_steps,loss_total,loss_velocity,loss_steering,buffer_size,extra
        """
        try:
            total_steps = len(self.m_memory)
            if loss_dict:
                lt = loss_dict.get("loss_total", '')
                lv = loss_dict.get("loss_velocity", '')
                ls = loss_dict.get("loss_steering", '')
            else:
                lt = lv = ls = ''
            line = f"{datetime.now().isoformat()},{phase},{epoch},{step},{total_steps},{lt},{lv},{ls},{buffer_size},{extra}\n"
            with open(self.metrics_log_path, 'a') as f:
                f.write(line)
        except Exception as e:
            print(f"[MetricsLog] Błąd zapisu: {e}")


if __name__ == '__main__':
    rclpy.init()
    manager = LearningManagement()
    # Uruchom serię epizodów z treningiem po każdym.
    manager.run_qlearning_epochs()
    # memory = manager.launch_qlearning()
    rclpy.shutdown()