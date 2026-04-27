from setuptools import find_packages, setup

package_name = "pick_place_motion_planning"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (
            f"share/{package_name}",
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="michael",
    maintainer_email="M.Eggleston0001@gmail.com",
    description="Motion planning package for the pick and place robot project.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "panda_moveit_planner = pick_place_motion_planning.panda_moveit_planner:main",
        ],
    },
)