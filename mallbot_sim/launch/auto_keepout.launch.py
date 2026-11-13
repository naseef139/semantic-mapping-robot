"""Automatic keepout zones: closes doorways and shop fronts found in the SLAM map.

Start it together with sim.launch.py and nav.launch.py (see AUTO_KEEPOUT.md).
  ros2 launch mallbot_sim auto_keepout.launch.py [door_max:=2.5] [use_signs:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('mallbot_sim')
    params = os.path.join(pkg, 'config', 'auto_keepout.yaml')
    sign_topic = PythonExpression([
        "'/semantic_notes' if '", LaunchConfiguration('use_signs'),
        "'.lower() == 'true' else '/auto_keepout/signs_disabled'"])
    return LaunchDescription([
        DeclareLaunchArgument('door_max', default_value='2.5',
                              description='widest doorway (m) closed without a sign'),
        DeclareLaunchArgument('use_signs', default_value='true',
                              description='use shop signs from the semantic mapper for wide shop fronts'),
        Node(
            package='mallbot_sim',
            executable='auto_keepout.py',
            name='auto_keepout',
            output='screen',
            emulate_tty=True,
            parameters=[params, {
                'door_max': ParameterValue(LaunchConfiguration('door_max'), value_type=float),
                'sign_topic': sign_topic,
            }],
        ),
    ])
