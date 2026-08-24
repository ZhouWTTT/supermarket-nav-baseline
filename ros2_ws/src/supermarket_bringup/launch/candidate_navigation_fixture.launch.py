from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("supermarket_bringup")
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "candidate_navigation.launch.py")
        )
    )
    return LaunchDescription(
        [
            Node(
                package="supermarket_bringup",
                executable="fake_public_server",
                output="screen",
            ),
            navigation,
        ]
    )
