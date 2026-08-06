"""Run the vision-guided shelf approach, grasp, lift, and retreat baseline only."""

import rclpy

from supermarket_sorting_client import PickPlaceClient


def main():
    rclpy.init()
    node = PickPlaceClient(finish_after_retreat=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
