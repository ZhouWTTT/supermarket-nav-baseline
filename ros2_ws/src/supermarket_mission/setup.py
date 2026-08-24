from setuptools import find_packages, setup


package_name = "supermarket_mission"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Supermarket Team",
    maintainer_email="team@example.invalid",
    description="Long-lived supermarket mission execution.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "manipulation_adapter = supermarket_mission.manipulation_adapter:main",
            "mission_executive = supermarket_mission.mission_executive:main",
        ],
    },
)
