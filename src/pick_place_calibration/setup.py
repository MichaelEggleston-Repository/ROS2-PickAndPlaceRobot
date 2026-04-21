from setuptools import find_packages, setup

package_name = "pick_place_calibration"

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
    maintainer="Michael",
    maintainer_email="you@example.com",
    description="Vision and camera acquisition package for the pick and place robot project.",
    license="TODO",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "calibration_pose_sweep = pick_place_calibration.calibration_pose_sweep:main",
        ],
    },
)