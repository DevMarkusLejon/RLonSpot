from __future__ import annotations

import sys

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

#Import selectors, actions, etc
from spot_bt_ros_py.composites.selector import create_lease_claim_selector
from spot_bt_ros_py.composites.selector import create_power_on_selector
from spot_bt_ros_py.composites.selector import create_power_off_selector
from spot_bt_ros_py.composites.selector import create_lease_release_selector
from spot_bt_ros_py.composites.selector import create_standing_selector
from spot_bt_ros_py.composites.selector import create_sitting_selector

from spot_bt_ros_py.tick import generic_pre_tick_handler
from spot_bt_ros_py.utils import create_mission_blackboard
from spot_bt_ros_py.utils import create_status_blackboard

from spot_bt_test.tree_nodes.test_graphnav_action import UploadGraphNavGraph
from spot_bt_test.tree_nodes.test_graphnav_action import WaitForGraphNavReady
from spot_bt_test.tree_nodes.test_graphnav_action import LocalizeInGraphNav
from spot_bt_test.tree_nodes.test_graphnav_action import NavigateToWaypointInGraphNav
from spot_bt_test.tree_nodes.test_graphnav_action import SetGraphNavWaypoint
from spot_bt_test.tree_nodes.test_graphnav_action import GetNextGraphNavWaypoint


def create_graphnav_behavior() -> Sequence:
    """"""
    behavior = Sequence("GraphNav test behavior", memory=True)
    behavior.add_children(
        [
            UploadGraphNavGraph(name="Upload GraphNav map"),
            #WaitForGraphNavReady(name="Wait for GraphNav ready"),
            SetGraphNavWaypoint(name="Set localize waypoint", waypoint_id=""),
            LocalizeInGraphNav(name="Set GraphNav localization"),
            SetGraphNavWaypoint(name="Set WP1", waypoint_id="wimpy-emu-uD6cjojKhSWPIf4V5cHnig=="), #wp3
            NavigateToWaypointInGraphNav(name="Navigate to GraphNav waypoint"),
            SetGraphNavWaypoint(name="Set WP2", waypoint_id="moire-hornet-5.hKWSBlGAr22Qd+ZXv7GA=="), #wp12
            NavigateToWaypointInGraphNav(name="Navigate to GraphNav waypoint"),
        ]
    )
    return behavior

def create_root() -> Sequence:
    """Create the root for GraphNav behvaior"""
    root = Sequence("GraphNav test root", memory=True)
    root.add_children(
        [
            create_lease_claim_selector(),
            create_power_on_selector(),
            create_standing_selector(),

            #Graph here
            create_graphnav_behavior(),

            create_sitting_selector(),
            create_power_off_selector(),
            create_lease_release_selector(),
        ]
    )

    render_dot_tree(root)
    return root

@ros_process.main()
def main():
    """Create test GraphNav bt tree and run."""

    # Set BT debug level and activity stream
    logging.level = logging.Level.DEBUG
    Blackboard.enable_activity_stream(maximum_size=100)

    # Create status blackboard
    status_blackboard = create_status_blackboard(docked=False)

    # Create mission blackboard
    mission_blackboard = create_mission_blackboard(dock_id=549)
    mission_blackboard.register_key(key="graph_nav_map_path", access=Access.WRITE)
    mission_blackboard.register_key(key="graph_nav_waypoint_id", access=Access.WRITE)
    mission_blackboard.register_key(key="waypoint_list", access=Access.WRITE)
    mission_blackboard.register_key(key="graph_nav_localization_method", access=Access.WRITE)

    mission_blackboard.graph_nav_map_path = "/home/sundt/thesis/colcon_ws/src/my_spot_thesis/spot_bt_test/graphnavs"
    mission_blackboard.waypoint_list = []
    #mission_blackboard.graph_nav_waypoint_id = "moire-hornet-5.hKWSBlGAr22Qd+ZXv7GA=="
    mission_blackboard.graph_nav_localization_method = "fiducial" #or "waypoint"

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
