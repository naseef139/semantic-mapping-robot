# mallbot_sim

Custom robot (dimensions match the CAD in mallbot_cad / the BOM) + mall world for **Online Object-Level Semantic Mapping During Autonomous Exploration**
(ROS 2 Humble, Ignition Gazebo Fortress).

## Robot
| Part | Value (edit in `urdf/mallbot.urdf.xacro`) |
|---|---|
| Drive | Differential, 65 mm wheels, track 0.28 m, **drive axle at the front**, 2 swivel casters at the rear corners (0.21 m behind the axle, y = ±0.08), max 0.34 m/s. `base_link` = drive axle centre |
| Chassis | 315 x 235 x 110 mm box on 345 x 265 x 5 mm plate (plate underside 47.5 mm) |
| 2D LiDAR | scan plane 0.2305 m, 0.10 m behind the axle, 360°, 0.2–12 m, 10 Hz, 720 samples |
| RGB-D camera | lens 0.60 m, 0.155 m behind the axle, level (mount tilts 0–20° up), 95° HFOV, 640x400, depth 0.2–12 m, 15 Hz |
| IMU | 100 Hz |

## Safety: cmd_vel timeout
`/cmd_vel` (teleop, Nav2) → `cmd_vel_timeout` node → `/cmd_vel_safe` → drive. If `/cmd_vel` stops for 0.7 s the robot stops.
Change it with `ros2 launch mallbot_sim sim.launch.py cmd_vel_timeout:=0.5`. 0.7 s is the default because the keyboard's auto-repeat starts after about 0.5 s; holding a teleop key must not make the robot stutter.

## ROS 2 topics
`/scan` `/camera/image` `/camera/depth_image` `/camera/camera_info` `/imu` `/odom` `/tf` `/joint_states` `/cmd_vel` `/map` (SLAM)

TF: `map → odom → base_footprint → base_link → {lidar_link, camera_link → camera_optical_frame, imu_link, wheels}`

## Build and run
```bash
sudo apt install -y ros-humble-ros-gz ros-humble-xacro ros-humble-robot-state-publisher ros-humble-slam-toolbox ros-humble-teleop-twist-keyboard
cd ~/mallbot_ws && colcon build --symlink-install && source install/setup.bash
ros2 launch mallbot_sim sim.launch.py
# new terminal
ros2 run teleop_twist_keyboard teleop_twist_keyboard
# save the map
ros2 run nav2_map_server map_saver_cli -f ~/maps/mall_v1
```

## Nav2 and autonomous exploration
```bash
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup
# explore_lite (build from source, once)
cd ~/mallbot_ws/src && git clone https://github.com/robo-friends/m-explore-ros2.git
cd ~/mallbot_ws && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select explore_lite mallbot_sim && source install/setup.bash

ros2 launch mallbot_sim sim.launch.py                  # terminal 1
ros2 launch mallbot_sim nav.launch.py                  # terminal 2: click '2D Goal Pose' in RViz
ros2 launch mallbot_sim nav.launch.py explore:=true    # or: the robot maps the mall by itself
```
The Nav2 footprint is the real outline. base_link is the front axle, and the body reaches 0.265 m behind it.

## Coloured logs
Add once to `~/.bashrc`: `source ~/mallbot_ws/install/mallbot_sim/share/mallbot_sim/mlaunch.bash`
Then use `mlaunch` instead of `ros2 launch`. Each node gets its own colour, warnings are yellow, errors red.
Filter with `MLAUNCH_ONLY=explore,bt_navigator mlaunch ...` or `MLAUNCH_HIDE=controller_server mlaunch ...`.

## mall_v2 (default): furnished for perception
Real furniture (cabinets, bookshelves, tables, café tables, trash cans), benches, potted plants, **2 standing people** (COCO *person*, must be filtered out of the notes), and **shop signs printed with the shop name** for OCR.
The furniture comes from [osrf/gazebo_models](https://github.com/osrf/gazebo_models) (© the original authors, see each `model.config`). Download it once:
```bash
python3 ~/mallbot_ws/src/mallbot_sim/tools/fetch_models.py
cd ~/mallbot_ws && colcon build --symlink-install --packages-select mallbot_sim
```
`ros2 launch mallbot_sim sim.launch.py mall:=mall_v1` still gives the plain mall (no download needed).
True positions of every object and sign: `config/mall_v2_ground_truth.csv` (for evaluation). Regenerate with `python3 tools/make_mall_v2.py`.

## Exploration (frontier_explorer, default)
`nav.launch.py explore:=true` starts MallBot's own explorer (`scripts/frontier_explorer.py`):
nearest frontier first, **short steps** (at most `max_step` = 3 m per goal, via Nav2's planned path),
**one goal at a time**: it waits for succeeded / aborted / timeout before choosing the next and never
pre-empts a running goal. Failed targets are blacklisted; when no frontier is left for 15 s it returns to the start.
Shops are not explored (keepout zones are lethal in the costmap it reads). `explorer:=explore_lite` still works.

## Keepout zones (default ON)
Shop interiors are **no-go zones**: Nav2 never plans or drives into them, and exploration never picks goals there. The robot stays in the corridor and looks into the shops through the doorways.
Mask: `config/keepout_mall_v2.yaml/.pgm` (black = no-go, **grey = no effect; never white**: white would erase all unknown space) (map frame; regenerate with `python3 tools/make_keepout_mask.py --spawn X Y YAW` if you spawn elsewhere).
Turn off with `nav.launch.py keepout:=false`. On the real robot: draw the zones over your saved map (black = no-go) and pass `keepout_mask:=/path/to/mask.yaml`.

## Change the mall
Edit `tools/make_mall_world.py` (shop names, objects), then `python3 tools/make_mall_world.py`.
It also writes `config/mall_v1_ground_truth.csv` (true object/sign positions, for evaluation).
