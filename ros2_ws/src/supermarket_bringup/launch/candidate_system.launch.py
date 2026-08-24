from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os


def generate_launch_description():
    share = get_package_share_directory("supermarket_bringup")
    repo_root = os.environ.get("SUPERMARKET_REPO_ROOT", "/workspace/baseline")
    examples = os.path.join(repo_root, "examples", "supermarket_sorting")
    perception = os.path.join(examples, "persistent_perception.py")
    weights = os.environ.get(
        "SUPERMARKET_BASELINE_WEIGHTS",
        os.path.join(examples, "perception", "checkpoints", "best.pt"),
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(share, "launch", "candidate_navigation.launch.py")
        )
    )
    return LaunchDescription(
        [
            navigation,
            ExecuteProcess(
                cmd=[
                    "python3", perception,
                    "--weights", weights,
                    "--confidence", os.environ.get("SUPERMARKET_CONFIDENCE", "0.45"),
                    "--max-inference-hz", os.environ.get("SUPERMARKET_INFERENCE_HZ", "12"),
                    "--device", os.environ.get("SUPERMARKET_DEVICE", "cpu"),
                    "--ready-file", "/tmp/supermarket_mission_perception.ready",
                ],
                output="screen",
            ),
            Node(
                package="supermarket_mission",
                executable="manipulation_adapter",
                output="screen",
            ),
            Node(
                package="supermarket_mission",
                executable="mission_executive",
                output="screen",
            ),
        ]
    )
