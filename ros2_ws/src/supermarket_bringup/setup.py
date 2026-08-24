from glob import glob
from setuptools import find_packages, setup


package_name = "supermarket_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/maps", glob("maps/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Supermarket Team",
    maintainer_email="team@example.invalid",
    description="Nav2 candidate launch and configuration.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "generate_map = supermarket_bringup.map_generator:main",
            "fake_public_server = supermarket_bringup.fake_public_server:main",
        ]
    },
)
