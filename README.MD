
## SIAPWPA ROS2 – Jetracer + Gazebo

This project provides a simulation of an autonomous vehicle (JetRacer / `saye` model) in Gazebo Sim, integrated with ROS 2. It features a simple perception pipeline (binary track mask and spline tracking) and control (PID/continuous commands). The provided launch file starts everything: Gazebo, the `ros_gz_bridge`, image preview, mask generation, and the control node.

---

**Requirements**
- ROS 2 (e.g., Humble/Foxy) with `colcon`.
- Gazebo Sim (`gz sim`) with installed models.
- The `ros_gz_bridge` package (ROS↔Gazebo bridge).
- Optional: NVIDIA GPU and working drivers (for Devcontainer).

---

**Repository Structure**
- `jetracer/` – ROS 2 package with:
	- `launch/jetracer.launch.py` – launches `gz sim`, the bridge, and helper nodes.
	- `worlds/Trapezoid/worlds/Trapezoid.world` – main track world.
	- `models/camera.sdf` – auxiliary camera model.
	- `jetracer/jetracer/nodes/` – nodes:
		- `nodes/management_move.py` – vehicle control loop (PID, continuous control, turn decisions based on depth).
		- `nodes/make_gt.py` – generates binary track mask and visualization (publishes `/chase_camera/green_mask`).
		- `nodes/publish_image*.py`, `check_collision.py` – helper tools.
		- `control/`, `perception/`, `sensors/`, `utils/`, `qlearning/` – control and perception modules.
- `car_package/ackermann-vehicle-gzsim-ros2/` – `saye_*` packages:
	- `saye_description/models/saye/` – vehicle models for Gazebo.
	- `saye_bringup/`, `saye_control/`, `saye_behaviortree/`, `saye_localization/`, `saye_msgs/` – system components.
- `camera/` – experiments (e.g., `Qlerning/`, `splain_tracking/`).
- `simple_example/worlds/` – example worlds (`city_for_car/`).
- `docs/` – additional documentation (`ros_gz_bridge.md`, `getting_started.md`).
- `build/`, `install/`, `log/` – build artifacts and logs (created by `colcon`).

---

**Quick Start**
1) Install dependencies (ROS 2, Gazebo, ros_gz_bridge).
2) (Devcontainer) If NVIDIA GPU is not available/configured, comment out the relevant lines in `.devcontainer/compose.yaml`.
3) Build and initialize the workspace.

```bash
cd /path/to/your/workspace  # e.g. ~/ros2_ws or …/siapwpa_ros2_project
colcon clean workspace
colcon build --symlink-install
source install/setup.bash  # or: source ~/.bashrc, if you have aliases
```

---

**Running (recommended – all via Launch)**

```bash
ros2 launch jetracer jetracer.launch.py
```

What happens:
- `GZ_SIM_RESOURCE_PATH` is set (for `saye` models).
- Gazebo starts with the track: `jetracer/worlds/Trapezoid/worlds/Trapezoid.world`.
- After ~3 s, the `ros_gz_bridge parameter_bridge` starts for cameras, depth, odometry, and control.
- After ~4-6 s, `rqt_image_view` launches (preview of `/rs_front/image` and `/chase_camera/green_mask`).
- After ~10 s, the control node `management_move` starts.

---

**Manual Run (if `ros2 launch` does not work)**

1) Add the model path (adjust to your directory):
```bash
export GZ_SIM_RESOURCE_PATH="$GZ_SIM_RESOURCE_PATH:/home/dbabiarczyk/semestr9/Pojazdy_autonomiczne/siapwpa_ros2_project/car_package/ackermann-vehicle-gzsim-ros2/saye_description/models"
```

2) Start the world (from the source directory or anywhere):
```bash
gz sim -r /home/dbabiarczyk/semestr9/Pojazdy_autonomiczne/siapwpa_ros2_project/jetracer/worlds/Trapezoid/worlds/Trapezoid.world
```

3) Start the ROS↔Gazebo bridge (bidirectional for the topics below):
```bash
ros2 run ros_gz_bridge parameter_bridge \
	/rs_front/image@sensor_msgs/msg/Image@gz.msgs.Image \
	/rs_front/depth_image@sensor_msgs/msg/Image@gz.msgs.Image \
	/rs_front/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo \
	/rs_front/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked \
	/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
	/model/saye_1/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry \
	/chase_camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image
```

4) Image preview:
```bash
ros2 run rqt_image_view rqt_image_view
```

5) (Optional) Manual vehicle control (velocity + steering command):
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}, angular: {z: 0.2}}'
```

---

**Topics used by the bridge** (defined in `jetracer/launch/jetracer.launch.py`):
- `/rs_front/image` – `sensor_msgs/Image` ↔ `gz.msgs.Image`
- `/rs_front/depth_image` – `sensor_msgs/Image` ↔ `gz.msgs.Image`
- `/rs_front/camera_info` – `sensor_msgs/CameraInfo` ↔ `gz.msgs.CameraInfo`
- `/rs_front/points` – `sensor_msgs/PointCloud2` ↔ `gz.msgs.PointCloudPacked`
- `/cmd_vel` – `geometry_msgs/Twist` ↔ `gz.msgs.Twist`
- `/model/saye_1/odometry` – `nav_msgs/Odometry` ↔ `gz.msgs.Odometry`
- `/chase_camera/image_raw` – `sensor_msgs/Image` ↔ `gz.msgs.Image`

---

**Diagnostics and Tips**
- Check if the resource path contains the track and models:
```bash
echo "$GZ_SIM_RESOURCE_PATH"
```
- If the Devcontainer does not have access to NVIDIA GPU – comment out GPU lines in `.devcontainer/compose.yaml`.
- For many topics in the bridge, YAML configuration is convenient – see `docs/ros_gz_bridge.md`.
- If `rqt_image_view` does not show the image, make sure Gazebo is in "Play" mode and the bridge is running.

---

**Project Notes (what the nodes do)**
- `management_move` – fetches track lines, computes lateral error, and controls the vehicle (PID). If needed, performs a turn maneuver based on the depth image.
- `make_gt` – binarizes the track image ("green/orange" mask), fits a reference spline, and publishes the visualization and `/chase_camera/green_mask`.

---

**World sources**
- Main track: `jetracer/worlds/Trapezoid/worlds/Trapezoid.world`.
- Additional world: `jetracer/worlds/city_for_car/world.sdf`.

---

**Additional: Minor fixes and improvements made on Jetson during tests**
1) The repository on Jetson includes minor fixes:
	- Adjusted ROI for the mounted camera
	- Accelerated calculations by changing image resolution
	- Disabled unnecessary topics for faster computation
	- Minor parameter changes tailored to the vehicle
	- Code logic changes:
		* Used `compressedImage` type instead of `Image` in `get_main_lines`
		* Commented out GT calculation
		* Commented out spline and bird_view publishing
		* Commented out bird's-eye view calculation
		* Red color in binary mask calculation

2) The repository is in the `jetracer_car_project` folder

3) This part of the repository does not include Gazebo or the Ackermann vehicle model

4) Running the project on Jetson:
- Start image analysis and control publishing:
```bash
cd jetracer_car_project
export ROS_DOMAIN_ID=35
source /opt/ros/dashing/setup.bash
colcon build --symlink-install
cd SiAPwPA-reorg-gt
source ../install/setup.bash
ros2 run jetracer ros_gazebo_ex
```

- Start the topic that publishes the depth image on `/depth/image_rect_raw` and the RGB image from the camera on `/color/image_raw`. In our case, this was a ready-made package for the Realsense camera located in the `realsense_wsV2` directory. Note: this file is not in this repository, it was on Jetson in the main directory:
```bash
cd realsense2_camera
export ROS_DOMAIN_ID=35
source /opt/ros/dashing/setup.bash
source ~/realsense_wsV2/install/setup.bash
ros2 run realsense2_camera realsense2_camera_node
```

- Start receiving control commands by the vehicle. Note: this file is not in this repository, it was on Jetson in the main directory:
```bash
source /opt/ros/dashing/setup.bash
export ROS_DOMAIN_ID=35
python3 ./cmd_vel_subscriber
```
