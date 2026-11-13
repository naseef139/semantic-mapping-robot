"""
Nav2 for MallBot (run AFTER sim.launch.py, which already runs SLAM).

  ros2 launch mallbot_sim nav.launch.py                # Nav2 only: click '2D Goal Pose' in RViz
  ros2 launch mallbot_sim nav.launch.py explore:=true  # Nav2 + frontier explorer: robot maps the mall by itself
  ros2 launch mallbot_sim nav.launch.py explore:=true max_step:=2.0   # shorter steps
  ros2 launch mallbot_sim nav.launch.py keepout:=false # allow driving into shops (default: shops are no-go zones)

Loads the standard Nav2 Humble parameters that came with nav2_bringup on this machine,
then overrides only what is specific to MallBot:
  - footprint: real outline around base_link (= front drive axle), not a circle
  - speed limits matched to the real motors (100 rpm x 65 mm wheel = 0.34 m/s max)
  - Smac State Lattice planner: checks the real footprint along the path, keeps to the middle
  - slow reverse allowed so the controller can back off a wall
The merged file is written to /tmp/mallbot_nav2_params.yaml (open it to see every value).

Output velocity goes to /cmd_vel -> cmd_vel_timeout node -> /cmd_vel_safe -> robot.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Robot outline in base_link (front drive axle) coordinates, metres, ~1.5 cm padding included.
# Chassis plate: x from +0.0825 (front) to -0.2625 (rear), y +/-0.1325; wheels reach y +/-0.1535.
FOOTPRINT = "[[0.10, 0.17], [0.10, -0.17], [-0.28, -0.17], [-0.28, 0.17]]"
MAX_V = 0.30      # m/s  (motor limit 0.34, keep a margin)
MAX_W = 1.0       # rad/s


def set_if(d, keys, value):
    """Set d[k1][k2]...[kn] = value only if the parent path exists."""
    for k in keys[:-1]:
        if not isinstance(d, dict) or k not in d:
            return False
        d = d[k]
    d[keys[-1]] = value
    return True


def make_params(context):
    nav2_share = get_package_share_directory('nav2_bringup')
    with open(os.path.join(nav2_share, 'params', 'nav2_params.yaml')) as f:
        p = yaml.safe_load(f)

    # Inflation: radius > circumscribed radius (0.33 m: rear corner swing about the front axle),
    # low scaling factor = cost fades slowly -> paths keep away from walls instead of hugging them.
    INFLATION = {'global_costmap': (0.75, 2.5), 'local_costmap': (0.55, 3.0)}
    for cm in ('local_costmap', 'global_costmap'):
        rp = p[cm][cm]['ros__parameters']
        rp.pop('robot_radius', None)
        rp['footprint'] = FOOTPRINT
        rp['footprint_padding'] = 0.0
        rp['robot_base_frame'] = 'base_link'
        if 'inflation_layer' in rp:
            rp['inflation_layer']['inflation_radius'] = INFLATION[cm][0]
            rp['inflation_layer']['cost_scaling_factor'] = INFLATION[cm][1]

    # Global planner: Smac STATE LATTICE with the differential-drive motion set.
    # It collision-checks the REAL footprint at every pose of the path, so it never plans through
    # gaps the robot can't fit (Smac 2D / NavFn treat the robot as a circle of the *inscribed*
    # radius = only 0.10 m here, because base_link sits on the front axle).
    # cost_penalty pushes paths to the middle of corridors and doorways.
    lattice = os.path.join(get_package_share_directory('nav2_smac_planner'), 'sample_primitives',
                           '5cm_resolution', '0.5m_turning_radius', 'diff', 'output.json')
    pl = p['planner_server']['ros__parameters']
    pl['planner_plugins'] = ['GridBased']
    smoother = {'max_iterations': 1000, 'w_smooth': 0.3, 'w_data': 0.2, 'tolerance': 1.0e-10}
    if os.path.exists(lattice):
        pl['GridBased'] = {
            'plugin': 'nav2_smac_planner/SmacPlannerLattice',
            'lattice_filepath': lattice,
            'allow_unknown': True,             # frontiers are at the edge of unknown space
            'tolerance': 0.5,
            'max_iterations': 1000000,
            'max_on_approach_iterations': 1000,
            'max_planning_time': 4.0,
            'analytic_expansion_ratio': 3.5,
            'analytic_expansion_max_length': 3.0,
            'reverse_penalty': 2.0,
            'change_penalty': 0.05,
            'non_straight_penalty': 1.05,
            'cost_penalty': 2.5,               # stay away from walls / inflated areas
            'rotation_penalty': 5.0,           # rotate in place only when really needed
            'retrospective_penalty': 0.015,
            'lookup_table_size': 20.0,
            'cache_obstacle_heuristic': False,
            'allow_reverse_expansion': False,
            'smooth_path': True,
            'smoother': smoother,
        }
    else:                                      # fallback (should not happen on Humble)
        pl['GridBased'] = {
            'plugin': 'nav2_smac_planner/SmacPlanner2D', 'tolerance': 0.5, 'allow_unknown': True,
            'downsample_costmap': False, 'downsampling_factor': 1, 'max_iterations': 1000000,
            'max_on_approach_iterations': 1000, 'max_planning_time': 2.0, 'cost_travel_multiplier': 2.0,
            'use_final_approach_orientation': False, 'smoother': smoother}
    # the global costmap must be 5 cm for the 5 cm lattice
    set_if(p, ['global_costmap', 'global_costmap', 'ros__parameters', 'resolution'], 0.05)

    # local costmap a bit larger: the rear swings out when turning about the front axle
    set_if(p, ['local_costmap', 'local_costmap', 'ros__parameters', 'width'], 4)
    set_if(p, ['local_costmap', 'local_costmap', 'ros__parameters', 'height'], 4)

    # controller (DWB in the default Humble file). 10 Hz is plenty at 0.3 m/s and leaves CPU for
    # Gazebo + YOLO (20 Hz gave 'Control loop missed its desired rate' warnings)
    set_if(p, ['controller_server', 'ros__parameters', 'controller_frequency'], 10.0)
    fp = ['controller_server', 'ros__parameters', 'FollowPath']
    # min_vel_x < 0: the controller may back away slowly when it is nose-to-wall and cannot turn
    for key, val in (('max_vel_x', MAX_V), ('max_speed_xy', MAX_V), ('min_vel_x', -0.08),
                     ('max_vel_theta', MAX_W), ('acc_lim_x', 0.5), ('decel_lim_x', -0.5),
                     ('acc_lim_theta', 1.5), ('decel_lim_theta', -1.5)):
        set_if(p, fp + [key], val)
    set_if(p, fp + ['BaseObstacle.scale'], 0.04)   # default 0.02: weigh obstacle cost a bit more

    # ---- KEEPOUT zones (shop interiors): Nav2 never plans or drives into them, and exploration
    # (which reads the global costmap) never picks goals there. Mask: config/keepout_mall_v2.yaml
    keepout = context.launch_configurations.get('keepout', 'true').lower() in ('true', '1', 'yes')
    mask_yaml = context.launch_configurations.get('keepout_mask', '') or os.path.join(
        get_package_share_directory('mallbot_sim'), 'config', 'keepout_mall_v2.yaml')
    if keepout:
        for cm in ('local_costmap', 'global_costmap'):
            rp = p[cm][cm]['ros__parameters']
            rp['filters'] = ['keepout_filter']
            rp['keepout_filter'] = {'plugin': 'nav2_costmap_2d::KeepoutFilter', 'enabled': True,
                                    'filter_info_topic': '/costmap_filter_info'}

    # velocity smoother (last stage before /cmd_vel)
    vs = ['velocity_smoother', 'ros__parameters']
    set_if(p, vs + ['max_velocity'], [MAX_V, 0.0, MAX_W])
    set_if(p, vs + ['min_velocity'], [-MAX_V, 0.0, -MAX_W])
    set_if(p, vs + ['max_accel'], [0.5, 0.0, 1.5])
    set_if(p, vs + ['max_decel'], [-0.5, 0.0, -1.5])

    out = '/tmp/mallbot_nav2_params.yaml'
    with open(out, 'w') as f:
        yaml.safe_dump(p, f, sort_keys=False)

    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_share, 'launch', 'navigation_launch.py')),
        launch_arguments={'use_sim_time': 'true', 'params_file': out, 'autostart': 'true'}.items())]

    if keepout:
        actions += [
            # publishes the mask image as an OccupancyGrid on /keepout_filter_mask
            Node(package='nav2_map_server', executable='map_server', name='filter_mask_server',
                 parameters=[{'use_sim_time': True, 'yaml_filename': mask_yaml,
                              'topic_name': '/keepout_filter_mask', 'frame_id': 'map'}], output='screen'),
            # tells the costmaps which mask to use and that it is a keepout mask (type 0)
            Node(package='nav2_map_server', executable='costmap_filter_info_server',
                 name='costmap_filter_info_server',
                 parameters=[{'use_sim_time': True, 'type': 0, 'filter_info_topic': '/costmap_filter_info',
                              'mask_topic': '/keepout_filter_mask', 'base': 0.0, 'multiplier': 1.0}],
                 output='screen'),
            Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                 name='lifecycle_manager_costmap_filters',
                 parameters=[{'use_sim_time': True, 'autostart': True,
                              'node_names': ['filter_mask_server', 'costmap_filter_info_server']}],
                 output='screen'),
            LogInfo(msg=f'keepout zones ON: {mask_yaml}'),
        ]

    # ---- optional: frontier exploration ----
    # explorer:=mallbot (default): our frontier_explorer - nearest frontier, short steps (max_step),
    #   ONE goal at a time, waits for the result, never pre-empts a running goal, waits for the
    #   costmap by itself.  explorer:=explore_lite: the m-explore-ros2 node (re-targets continuously).
    if context.launch_configurations.get('explore', 'false').lower() in ('true', '1', 'yes'):
        which = context.launch_configurations.get('explorer', 'mallbot')
        if which == 'explore_lite':
            try:
                get_package_share_directory('explore_lite')
            except Exception:
                return actions + [LogInfo(msg='explorer:=explore_lite but explore_lite is not built.')]
            explore_yaml = os.path.join(get_package_share_directory('mallbot_sim'), 'config', 'explore.yaml')
            delay = float(context.launch_configurations.get('explore_delay', '20.0'))
            actions.append(LogInfo(msg=f'explore_lite starts in {delay:.0f} s'))
            actions.append(TimerAction(period=delay, actions=[Node(
                package='explore_lite', executable='explore', name='explore_node',
                parameters=[explore_yaml, {'use_sim_time': True}], output='screen')]))
        else:
            step = float(context.launch_configurations.get('max_step', '3.0'))
            actions.append(Node(
                package='mallbot_sim', executable='frontier_explorer.py', name='frontier_explorer',
                parameters=[{'use_sim_time': True, 'max_step': step, 'min_frontier_size': 0.5,
                             'goal_timeout': 60.0, 'done_after_s': 15.0, 'return_to_start': True,
                             'costmap_topic': '/global_costmap/costmap', 'robot_base_frame': 'base_link'}],
                output='screen'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('explore', default_value='false',
                              description='true = also start explore_lite (autonomous frontier exploration)'),
        DeclareLaunchArgument('explorer', default_value='mallbot',
                              description='mallbot (one short goal at a time) or explore_lite'),
        DeclareLaunchArgument('max_step', default_value='3.0',
                              description='mallbot explorer: max distance (m) of one navigation goal'),
        DeclareLaunchArgument('explore_delay', default_value='20.0',
                              description='explore_lite only: seconds to wait before it starts'),
        DeclareLaunchArgument('keepout', default_value='true',
                              description='true = shop interiors are no-go zones (robot stays in the corridor)'),
        DeclareLaunchArgument('keepout_mask', default_value='',
                              description='keepout mask yaml (default: config/keepout_mall_v2.yaml)'),
        OpaqueFunction(function=make_params),
    ])
