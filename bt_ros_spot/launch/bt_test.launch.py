"""ROS 2 Launch file for spot_bt_test"""
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchContext
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

def get_demo_type_node(context: LaunchContext) -> list[Node]:
    """Get demo type from launch configuration"""
    demo_type = LaunchConfiguration("demo_type").perform(context)
    if demo_type.lower() not in [
        "arm",
        "graphnav",
        "standwalksit",
    ]:
        raise ValueError(
            "The only demo_types supported are: arm, graphnav, standwalksit"
        )

    return [
        Node(
            package="spot_bt_test",
            executable=f"spot_{demo_type}_demo",
            output="screen",
        )
    ]


def generate_launch_description():
    config_file = LaunchConfiguration("config_file", default="")
    has_arm = LaunchConfiguration("has_arm", default="True")

    # Include Spot driver launch file
    spot_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("spot_driver"), "launch"),
                "/spot_driver.launch.py",
            ]
        ),
        launch_arguments={
            "config_file": config_file
        }.items(),  # , "has_arm": has_arm}.items(),
    )
    # Include spot_bt_ros_node movement controller launch file
    spot_controller_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("spot_bt_ros_bringup"), "launch"),
                "/controller.launch.py",
            ]
        ),
        launch_arguments={
            "config_file": config_file
        }.items(),  # , "has_arm": has_arm}.items(),
    )
    # Include spot_bt_ros_node simple planner launch file
    spot_planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(get_package_share_directory("spot_bt_ros_bringup"), "launch"),
                "/planner.launch.py",
            ]
        ),
        launch_arguments={
            "config_file": config_file
        }.items(),  # , "has_arm": has_arm}.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                description="Path to config file for the driver.",
                default_value="",
            ),
            DeclareLaunchArgument(
                "has_arm",
                description="Wheter spot has an arm",
                default_value="True",
            ),
            spot_driver_launch,
            spot_controller_launch,
            spot_planner_launch,
            OpaqueFunction(function=get_demo_type_node),
        ]
    )
