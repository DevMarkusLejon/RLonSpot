"""spot_bt_ros_py Arm Demonstration."""

from __future__ import annotations

import sys

from geometry_msgs.msg import PoseStamped

from py_trees import logging
from py_trees.blackboard import Blackboard
from py_trees.common import Access
from py_trees.composites import Sequence
from py_trees.display import render_dot_tree

from py_trees_ros.exceptions import TimedOutError
from py_trees_ros.trees import BehaviourTree

import rclpy
from rclpy.executors import ExternalShutdownException

import synchros2.process as ros_process

from spot_bt_ros_py.actions.arm import ArmStow
from spot_bt_ros_py.actions.arm import ArmTrajectory
from spot_bt_ros_py.actions.arm import ArmUnstow
from spot_bt_ros_py.actions.arm import CloseGripper
from spot_bt_ros_py.actions.arm import OpenGripper
from spot_bt_ros_py.actions.movement import ComputePathToFiducial
from spot_bt_ros_py.actions.movement import MoveToTarget
from spot_bt_ros_py.actions.movement import ComputeNewWaypoint
from spot_bt_ros_py.composites.selector import create_lease_claim_selector
from spot_bt_ros_py.composites.selector import create_power_off_selector
from spot_bt_ros_py.composites.selector import create_power_on_selector
from spot_bt_ros_py.composites.selector import create_lease_release_selector
from spot_bt_ros_py.composites.selector import create_sitting_selector
from spot_bt_ros_py.composites.selector import create_standing_selector
from spot_bt_ros_py.composites.selector import create_generic_fiducial_selector
from spot_bt_ros_py.tick import generic_pre_tick_handler
from spot_bt_ros_py.utils import create_mission_blackboard
from spot_bt_ros_py.utils import create_status_blackboard

def create_test_behavior() -> Sequence:
    """Create first test of own bt."""
    behavior = Sequence("Simple test bt", memory=True) #What does memory do?
    behavior.add_children(
        [
            #create_generic_fiducial_selector(no_dock=True),
	    #ComputePathToFiducial(name="Compute path to fiducial"),
	    ComputeNewWaypoint(name="Get random search traj"),
	    MoveToTarget(name="Move to target goal"),
        ]
    )
    return behavior

def create_arm_motion_behavior() -> Sequence:
    """Create a simple arm motion behavior."""
    behavior = Sequence("Simple arm motion", memory=True)
    behavior.add_children(
        [
            ArmUnstow(name="Unstow arm"),
            #ArmTrajectory(name="Move arm to postion"),
            OpenGripper(name="Open gripper"),
            CloseGripper(name="Close gripper"),
            ArmStow(name="Stow arm"),
        ]
    )

    return behavior


def create_root() -> Sequence:
    """Create the root for the Autonomy capability."""
    root = Sequence("Demo arm", memory=True)
    root.add_children(
        [
            create_lease_claim_selector(),
            create_power_on_selector(),
            create_standing_selector(),
	    #create_arm_motion_behavior(),

 	    create_test_behavior(),

	    create_sitting_selector(),
            create_power_off_selector(),
            create_lease_release_selector(),
        ]
    )

    render_dot_tree(root)

    return root


@ros_process.main()
def main():
    """Create demo behavior tree and run."""

    # Set BT debug level and activity stream
    logging.level = logging.Level.DEBUG
    Blackboard.enable_activity_stream(maximum_size=100)

    # Create status blackboard
    status_blackboard = create_status_blackboard(docked=True)

    # Create mission blackboard
    mission_blackboard = create_mission_blackboard(dock_id=549)
    mission_blackboard.register_key(key="fiducials", access=Access.WRITE)
    mission_blackboard.fiducials = None
    mission_blackboard.register_key(key="target", access=Access.WRITE)
    mission_blackboard.target = PoseStamped()
    mission_blackboard.target.pose.position.x = 1.0
    mission_blackboard.target.pose.position.y = 0.25
    mission_blackboard.target.pose.position.z = 0.0
    mission_blackboard.target.pose.orientation.w = 1.0

    # Enable tree stewardship
    root = create_root()
    tree = BehaviourTree(root)
    tree.add_pre_tick_handler(generic_pre_tick_handler)

    # Setup the behavior tree
    try:
        # tree.setup(timeout=15)  # by default if node=None, it will create a new node.
        tree.setup(node=main.node, timeout=15)
    except TimedOutError as e:
        main.node.get_logger().error(f"Failed to setup the tree, aborting [{e}]")
        main.node.destroy_node()
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)
    except KeyboardInterrupt:
        # not a warning, nor error, usually a user-initiated shutdown
        main.node.get_logger().error("Tree setup interrupted!")
        main.node.destroy_node()
        tree.shutdown()
        rclpy.try_shutdown()
        sys.exit(1)

    tree.tick_tock(period_ms=1000.0)

    # Execute the behavior tree
    try:
        rclpy.spin(tree.node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        tree.shutdown()
        main.node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
