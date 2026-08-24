from glob import glob
from setuptools import find_packages, setup


package_name = "supermarket_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Supermarket Team",
    maintainer_email="team@example.invalid",
    description="Single-owner navigation and motion safety.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "confirmed_obstacle_tracker = supermarket_navigation.confirmed_obstacle_node:main",
            "footprint_manager = supermarket_navigation.footprint_manager:main",
            "motion_arbiter = supermarket_navigation.motion_arbiter:main",
            "navigation_session = supermarket_navigation.navigation_session_node:main",
            "sensor_adapter = supermarket_navigation.sensor_adapter:main",
        ],
    },
)
