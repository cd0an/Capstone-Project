from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='capstone_brain',
            executable='detector_node',
            name='detector_node',
            output='screen',
        ),
        Node(
            package='capstone_brain',
            executable='tracking_node',
            name='tracking_node',
            output='screen',
        ),
        Node(
            package='capstone_brain',
            executable='fsm_node',
            name='fsm_node',
            output='screen',
        ),
    ])
