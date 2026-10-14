from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
import os


def generate_launch_description():
    package_share = get_package_share_directory('my_car_astar')
    params_file = os.path.join(package_share, 'config', 'astar_drive.yaml')

    return LaunchDescription([
        Node(
            package='my_car_astar',
            executable='astar_drive',
            name='astar_drive',
            output='screen',
            parameters=[params_file],
        ),
    ])
