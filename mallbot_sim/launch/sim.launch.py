"""
MallBot simulation launch.

  ros2 launch mallbot_sim sim.launch.py              # Gazebo + robot + SLAM + RViz (mall_v2)
  ros2 launch mallbot_sim sim.launch.py mall:=mall_v1  # plain mall (no downloaded models needed)
  ros2 launch mallbot_sim sim.launch.py slam:=false  # no SLAM (RViz fixed frame -> odom)

Starts:
  1. Ignition Gazebo (Fortress) with the mall world
  2. robot_state_publisher (URDF from xacro)
  3. Spawns MallBot into the world
  4. ros_gz_bridge for clock, cmd_vel_safe, odom, tf, joint states, scan, imu, camera
  4b. cmd_vel_timeout node (stops the robot 0.7 s after the last /cmd_vel)
  5. slam_toolbox (online async)            [slam:=true]
  6. RViz2 with a ready-made layout         [rviz:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('mallbot_sim')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    slam_toolbox = get_package_share_directory('slam_toolbox')

    world = LaunchConfiguration('world')
    slam = LaunchConfiguration('slam')
    rviz = LaunchConfiguration('rviz')
    x, y, yaw = LaunchConfiguration('x'), LaunchConfiguration('y'), LaunchConfiguration('yaw')

    declare_args = [
        # mall_v2 = furnished + text signs (run tools/fetch_models.py once); mall_v1 = plain boxes
        DeclareLaunchArgument('mall', default_value='mall_v2'),
        DeclareLaunchArgument('world', default_value=[os.path.join(pkg, 'worlds', ''), LaunchConfiguration('mall'), '.sdf']),
        DeclareLaunchArgument('slam', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('x', default_value='-10.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('cmd_vel_timeout', default_value='0.7'),
    ]

    robot_description = ParameterValue(
        Command(['xacro ', os.path.join(pkg, 'urdf', 'mallbot.urdf.xacro')]),
        value_type=str)

    # 1. Gazebo (-r = start running immediately)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world]}.items())

    # 2. Robot state publisher
    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen')

    # 3. Spawn robot
    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-topic', 'robot_description', '-name', 'mallbot',
                   '-x', x, '-y', y, '-z', '0.02', '-Y', yaw],
        output='screen')

    # 4. Bridge Gazebo <-> ROS 2  ( [ = GZ->ROS,  ] = ROS->GZ )
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/cmd_vel_safe@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/imu@sensor_msgs/msg/Imu[ignition.msgs.IMU',
            '/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen')

    # 4b. cmd_vel timeout: /cmd_vel -> /cmd_vel_safe, publishes zero if /cmd_vel goes quiet
    cmd_timeout = Node(
        package='mallbot_sim', executable='cmd_vel_timeout.py',
        parameters=[{'timeout': ParameterValue(LaunchConfiguration('cmd_vel_timeout'), value_type=float)}],
        output='screen')

    # 5. SLAM
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox, 'launch', 'online_async_launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'slam_params_file': os.path.join(pkg, 'config', 'slam_toolbox.yaml'),
        }.items(),
        condition=IfCondition(slam))

    # 6. RViz
    rviz_node = Node(
        package='rviz2', executable='rviz2',
        arguments=['-d', os.path.join(pkg, 'config', 'mallbot.rviz')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(rviz),
        output='screen')

    # let Gazebo find model://cabinet, model://mall_signs, ... in share/mallbot_sim/models
    models_path = [AppendEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', os.path.join(pkg, 'models')),
                   AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'models'))]

    return LaunchDescription(declare_args + models_path + [gazebo, rsp, spawn, bridge, cmd_timeout, slam_launch, rviz_node])
