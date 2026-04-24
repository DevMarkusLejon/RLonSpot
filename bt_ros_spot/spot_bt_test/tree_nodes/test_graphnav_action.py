""""Spot GraphNav actions test."""

from __future__ import annotations

from py_trees.behaviour import Behaviour
from py_trees.blackboard import Client
from py_trees.common import Access
from py_trees.common import Status

from rclpy.action import ActionClient
from rclpy.client import Client as ServiceClient
from rclpy.action.server import ServerGoalHandle
from rclpy.node import Node
from rclpy.task import Future

from spot_bt_ros_py.utils import SpotBTActionClientMixin

from spot_msgs.srv import GraphNavUploadGraph
from spot_msgs.srv import GraphNavSetLocalization
from spot_msgs.srv import ListGraph
from spot_msgs.srv import GraphNavGetLocalizationPose
from spot_msgs.action import NavigateTo


class UploadGraphNavGraph(Behaviour):
    """Upload GraphNav graph to the robot."""
    def __init__(self, name: str):
        super().__init__(name)
        self.blackboard: Client = None
        self.client: ServiceClient = None
        self.future: Future = None
        self.node: Node = None
        self._result: GraphNavUploadGraph.Response | None = None

    def setup(self, **kwargs):
        """Setup ROS 2 connection to spot_driver"""
        self.logger.debug(f"{self.qualified_name}.setup()") 
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            error_msgs = f"didn't find 'node' in setup's kwargs [{self.qualified_name}]"
            raise KeyError(error_msgs) from e
        
        # Create service client
        self.client = self.node.create_client(GraphNavUploadGraph, "graph_nav_upload_graph")

    def initialise(self):
        """Initialize variables and perform service behavior for first tick."""
        self.logger.debug(f"  {self.name} [UploadGraphNav::initialise()]")
        self.blackboard = self.attach_blackboard_client("mission")
        self.blackboard.register_key(key="graph_nav_map_path", access=Access.READ)

        self.future = None
        self._result = None

    def update(self) -> Status:
        
        self.logger.debug(f"  {self.name} [UploadGraphNav::update()]")
        
        if not self.client.service_is_ready():
            self.logger.debug(f"  {self.name} [UploadGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        if self.future is None:
            request = GraphNavUploadGraph.Request()
            request.upload_filepath = self.blackboard.graph_nav_map_path
            self.future = self.client.call_async(request)
            self.logger.debug(f"  {self.name} [UploadGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        if not self.future.done():
            self.logger.debug(f"  {self.name} [UploadGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        self._result = self.future.result()
        if self._result is None:
            self.logger.error("graph_nav_upload_graph returned no response")
            return Status.FAILURE

        if self._result.success:
            self.node.get_logger().info(f"Graph uploaded: {self.blackboard.graph_nav_map_path}")
            return Status.SUCCESS
        
        self.logger.error(self._result.message)
        return Status.FAILURE 

class WaitForGraphNavReady(Behaviour):
    """Wait for GraphNav to load map"""
    def __init__(self, name):
        super().__init__(name)
        self.blackboard: Client = None
        self.client: ServiceClient = None
        self.future: Future = None
        self.node: Node = Node
        self._result: GraphNavGetLocalizationPose.Response | None = None

    def setup(self, **kwargs):
        """Setup ROS 2 connection to spot_driver"""
        self.logger.debug(f"{self.qualified_name}.setup()") 
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            error_msgs = f"didn't find 'node' in setup's kwargs [{self.qualified_name}]"
            raise KeyError(error_msgs) from e
        
        # Create service client
        self.client = self.node.create_client(GraphNavGetLocalizationPose, "graph_nav_get_localization_pose")

    def initialise(self):
        """Initialize variables and perform service behavior for first tick."""
        self.logger.debug(f"  {self.name} [WaitForGraphNavReady::initialise()]")
        #self.blackboard = self.attach_blackboard_client("mission")
        #self.blackboard.register_key(key="graph_nav_map_path", access=Access.READ)

        self.future = None
        self._result = None

    def update(self) -> Status:
        self.logger.debug(f"  {self.name} [WaitForGraphNavReady::update()]")

        if not self.client.service_is_ready():
            self.logger.debug(f"  {self.name} [WaitForGraphNavReady::update()][RUNNING Service not ready]")
            return Status.RUNNING

        if self.future is None:
            request = GraphNavGetLocalizationPose.Request()
            self.future = self.client.call_async(request)
            self.logger.debug(f"  {self.name} [WaitForGraphNavReady::update()][RUNNING Future is none]")
            return Status.RUNNING
        
        if not self.future.done():
            return Status.RUNNING
        
        self._result = self.future.result()
        self.future = None
        
        if self._result is None:
            self.logger.debug(f"  {self.name} [WaitForGraphNavReady::update()][RUNNING Result is none]")
            return Status.RUNNING
        
        self.node.get_logger().info(f"GraphNav ready (localized={self._result.success})")
        return Status.SUCCESS

class LocalizeInGraphNav(Behaviour):
    """Set GraphNav localization via fiducial or waypoint"""
    def __init__(self, name):
        super().__init__(name)
        self.blackboard: Client = None
        self.client: ServiceClient = None
        self.future: Future = None
        self.node: Node = None
        self._result: GraphNavSetLocalization.Response | None = None

    def setup(self, **kwargs):
        """Setup ROS 2 connection to spot_driver"""
        self.logger.debug(f"{self.qualified_name}.setup()") 
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            error_msgs = f"didn't find 'node' in setup's kwargs [{self.qualified_name}]"
            raise KeyError(error_msgs) from e
        
        # Create service client
        self.client = self.node.create_client(GraphNavSetLocalization, "graph_nav_set_localization")

    
    def initialise(self):
        """Initialize variables and perform service behavior for first tick."""
        self.logger.debug(f"  {self.name} [LocalizeInGraphNav::initialise()]")
        self.blackboard = self.attach_blackboard_client("mission")
        self.blackboard.register_key(key="graph_nav_localization_method", access=Access.READ)
        self.blackboard.register_key(key="graph_nav_waypoint_id", access=Access.READ)

        self.future = None
        self._result = None
    
    def update(self) -> Status:
        self.logger.debug(f"  {self.name} [LocalizeInGraphNav::update()]")

        if not self.client.service_is_ready():
            self.logger.debug(f"  {self.name} [LocalizeInGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        if self.future is None:
            request = GraphNavSetLocalization.Request()
            request.method = self.blackboard.graph_nav_localization_method
            request.waypoint_id = self.blackboard.graph_nav_waypoint_id
            self.future = self.client.call_async(request)
            self.logger.debug(f"  {self.name} [LocalizeInGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        if not self.future.done():
            self.logger.debug(f"  {self.name} [LocalizeInGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        self._result = self.future.result()
        if self._result is None:
            self.logger.error("graph_nav_set_localization returned no response")
            return Status.FAILURE
        
        if self._result.success:
            self.node.get_logger().info("GraphNav successfully localized")
            return Status.SUCCESS
        
        self.logger.error(self._result.message)
        return Status.FAILURE

class NavigateToWaypointInGraphNav(Behaviour, SpotBTActionClientMixin):
    """Navigate to a waypoint in the loaded GraphNav map"""
    def __init__(self, name):
        super().__init__(name)
        self.blackboard: Client = None
        self.client: ActionClient = None
        self.future: Future = None
        self.node: Node = None
        self._goal_handle: ServerGoalHandle = None
        self._get_result_future: Future = None
        self._result: NavigateTo.Result | None = None
        self._result_status: int | None = None

    def setup(self, **kwargs):
        """Setup ROS 2 connection to spot_driver"""
        self.logger.debug(f"{self.qualified_name}.setup()") 
        try:
            self.node = kwargs["node"]
        except KeyError as e:
            error_msgs = f"didn't find 'node' in setup's kwargs [{self.qualified_name}]"
            raise KeyError(error_msgs) from e
        
        # Create action client
        self.client = ActionClient(self.node, NavigateTo, "navigate_to")

    def initialise(self):
        """Initialize variables and perform service behavior for first tick."""
        self.logger.debug(f"  {self.name} [NavigateToWaypointInGraphNav::initialise()]")
        self.blackboard = self.attach_blackboard_client("mission")
        self.blackboard.register_key(key="graph_nav_waypoint_id", access=Access.READ)
        self._result_status = None
        self._result = None

        goal_msg = NavigateTo.Goal()
        goal_msg.waypoint_id = self.blackboard.graph_nav_waypoint_id
        self.future = self.client.send_goal_async(goal_msg, feedback_callback=self._feedback)
        self.future.add_done_callback(self._goal_response_callback)

    def update(self) -> Status:
        self.logger.debug(f"  {self.name} [NavigateToWaypointInGraphNav::update()]")
        
        if self._result_status is None:
            self.logger.debug(f"  {self.name} [NavigateToWaypointInGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        if not self._get_result_future.done():
            self.node.get_logger().warn(
                f"--- Got result, but future not complete --> [{self.qualified_name}]"
            )
            self.logger.debug(f"  {self.name} [NavigateToWaypointInGraphNav::update()][RUNNING]")
            return Status.RUNNING
        
        if self.future.done():
            if self._result.success:
                self.node.get_logger().info("GraphNav successfully navigated")
                return Status.SUCCESS
            
            self.logger.error(self._result.message)
            return Status.FAILURE
        
        self.logger.debug(f"  {self.name} [NavigateToWaypointInGraphNav::update()][RUNNING]")
        return Status.RUNNING
    
    def terminate(self, new_status: str):
        """Terminate behavior and save information."""
        self.logger.debug(
            f"  {self.name} [NavigateToWaypointInGraphNav::terminate()]"
            f"[{self.status}->{new_status}]"
        )

        if self.status == Status.RUNNING and new_status == Status.INVALID:
            self._send_cancel_request()

    def _feedback(self, feedback_msg):
        """Log action feedback for MoveToGoal."""
        self.node.get_logger().info(
            f"Navigating...",
            throttle_duration_sec=1.0
        )
        

class SetGraphNavWaypoint(Behaviour):
    def __init__(self, name: str, waypoint_id: str):
        super().__init__(name)
        self.blackboard: Client = None
        self.waypoint_id = waypoint_id

    def initialise(self):
        """Initialize variables and perform service behavior for first tick."""
        self.logger.debug(f"  {self.name} [SetGraphNavWaypoint::initialise()]")

        self.blackboard = self.attach_blackboard_client("mission")
        self.blackboard.register_key(key="graph_nav_waypoint_id", access=Access.WRITE)

    def update(self) -> Status:
        self.logger.debug(f"  {self.name} [SetGraphNavWaypoint::update()]")

        self.blackboard.graph_nav_waypoint_id = self.waypoint_id
        self.logger.debug(f"Set waypoint -> {self.waypoint_id}")
        return Status.SUCCESS
    

class GetNextGraphNavWaypoint(Behaviour):
    def __init__(self, name):
        super().__init__(name)
        self.blackboard: Client = None

    def initialise(self):
        """Initialize variables and perform service behavior for first tick."""
        self.logger.debug(f"  {self.name} [GetNextGraphNavWaypoint::initialise()]")

        self.blackboard = self.attach_blackboard_client("mission")
        self.blackboard.register_key(key="waypoint_list", access=Access.READ)
        self.blackboard.register_key(key="graph_nav_waypoint_id", access=Access.READ)

    def update(self) -> Status:
        self.logger.debug(f"  {self.name} [GetNextGraphNavWaypoint::update()]")

        if len(self.blackboard.waypoint_list) == 0:
            self.logger.info("No more waypoints")
            return Status.FAILURE
        
        next_wp = self.blackboard.waypoint_list.pop(0)
        self.blackboard.graph_nav_waypoint_id = next_wp
        self.logger.info(f"Next waypoint: {next_wp}")
        return Status.SUCCESS