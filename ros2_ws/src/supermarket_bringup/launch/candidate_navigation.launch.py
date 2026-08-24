from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    share = get_package_share_directory("supermarket_bringup")
    params_file = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    configured = RewrittenYaml(
        source_file=params_file,
        root_key="",
        param_rewrites={"yaml_filename": map_file},
        convert_types=True,
    )

    nav_nodes = [
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[configured],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[configured],
        ),
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[configured],
            remappings=[("cmd_vel", "/motion/nav_cmd_vel")],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[configured],
            remappings=[("cmd_vel", "/motion/nav_cmd_vel")],
        ),
        Node(
            package="nav2_velocity_smoother",
            executable="velocity_smoother",
            name="velocity_smoother",
            output="screen",
            parameters=[configured],
            remappings=[
                ("cmd_vel", "/motion/selected_cmd_vel"),
                ("cmd_vel_smoothed", "/motion/smoothed_cmd_vel"),
            ],
        ),
        Node(
            package="nav2_collision_monitor",
            executable="collision_monitor",
            name="collision_monitor",
            output="screen",
            parameters=[configured],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[configured],
        ),
    ]

    application_nodes = [
        Node(
            package="supermarket_navigation",
            executable="sensor_adapter",
            name="sensor_adapter",
            output="screen",
        ),
        Node(
            package="supermarket_navigation",
            executable="motion_arbiter",
            name="motion_arbiter",
            output="screen",
        ),
        Node(
            package="supermarket_navigation",
            executable="footprint_manager",
            name="footprint_manager",
            output="screen",
        ),
        Node(
            package="supermarket_navigation",
            executable="confirmed_obstacle_tracker",
            name="confirmed_obstacle_tracker",
            output="screen",
        ),
        Node(
            package="supermarket_navigation",
            executable="navigation_session",
            name="navigation_session",
            output="screen",
        ),
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=os.path.join(share, "config", "nav2_candidate.yaml"),
            ),
            DeclareLaunchArgument(
                "map",
                default_value=os.path.join(share, "maps", "supermarket.yaml"),
            ),
            *application_nodes,
            *nav_nodes,
        ]
    )
