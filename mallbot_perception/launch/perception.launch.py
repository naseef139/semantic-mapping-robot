"""
Object notes for MallBot (run with sim.launch.py, and nav.launch.py explore:=true).

  ros2 launch mallbot_perception perception.launch.py
  ros2 launch mallbot_perception perception.launch.py model:=yolov8n.pt      # plain COCO YOLO
  ros2 launch mallbot_perception perception.launch.py device:=0              # NVIDIA GPU
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def node(context):
    cfg = os.path.join(get_package_share_directory('mallbot_perception'), 'config', 'perception.yaml')
    over = {}
    for k in ('model', 'device', 'save_path'):
        v = context.launch_configurations.get(k, '')
        if v:
            over[k] = v
    return [Node(package='mallbot_perception', executable='semantic_mapper', name='semantic_mapper',
                 parameters=[cfg, over], output='screen')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('model', default_value=''),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument('save_path', default_value=''),
        OpaqueFunction(function=node),
    ])
