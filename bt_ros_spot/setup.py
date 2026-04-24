
from glob import glob
import os
from setuptools import find_packages, setup

PACKAGE_NAME = "spot_bt_test"

setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
	(os.path.join("share", PACKAGE_NAME, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "py_trees", "py_trees_ros"],
    zip_safe=True,
    maintainer="sundtlejon",
    maintainer_email="sundtfredrik@gmail.com",
    description="Testing Spot ROS BT",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        'console_scripts': [
		'spot_standwalksit_demo = spot_bt_test.stand_walk_sit_bt:main',
		'spot_arm_demo = spot_bt_test.demo_arm:main',
        'spot_graphnav_demo = spot_bt_test.graphnav_test:main',
        ],
    },
)
