# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass
from operator import add, mul
from threading import Event
from typing import List

import numpy as np
import onnxruntime as ort
import orbit.observations as ob
from bosdyn.api import robot_command_pb2
from bosdyn.api.robot_command_pb2 import JointControlStreamRequest
from bosdyn.api.robot_state_pb2 import RobotStateStreamResponse
from bosdyn.util import seconds_to_timestamp, set_timestamp_from_now, timestamp_to_sec
from orbit.orbit_configuration import OrbitConfig
from orbit.orbit_constants import ordered_joint_names_orbit
from spot.constants import DEFAULT_K_Q_P, DEFAULT_K_QD_P, ordered_joint_names_bosdyn
from utils.dict_tools import dict_to_list, find_ordering, reorder

@dataclass
class OnnxControllerContext:
    """data class to hold runtime data needed by the controller"""

    event = Event()
    latest_state = None
    velocity_cmd = [0, 0, 0]
    count = 0


class StateHandler:
    """Class to be used as callback for state stream to put state date
    into the controllers context
    """

    def __init__(self, context: OnnxControllerContext) -> None:
        self._context = context

    def __call__(self, state: RobotStateStreamResponse):
        """make class a callable and handle incoming state stream when called

        arguments
        state -- proto msg from spot containing most recent data on the robots state"""
        self._context.latest_state = state
        self._context.event.set()


def print_observations(observations: List[float]):
    """debug function to print out the observation data used as model input

    arguments
    observations -- list of float values ready to be passed into the model
    """
    print("base_linear_velocity:", observations[0:3])
    print("base_angular_velocity:", observations[3:6])
    print("projected_gravity:", observations[6:9])
    print("commanded_vel:", observations[9:12])
    print("joint_positions:", observations[12:31])
    print("joint_velocity:", observations[31:50])
    print("last_action:", observations[50:69])


class OnnxStandWalkCommandGenerator:
    """class to be used as generator for spots command stream that executes
    an onnx model and converts the output to a spot command"""

    def __init__(
        self, 
        context: OnnxControllerContext, 
        stand_config: OrbitConfig, 
        walk_config: OrbitConfig,
        stand_policy_file_name: os.PathLike, 
        walk_policy_file_name: os.PathLike,
        verbose: bool
    ):
        self._context = context
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose

        self._stand_config = stand_config
        self._walk_config = walk_config
        self._stand_session = ort.InferenceSession(stand_policy_file_name)
        self._walk_session = ort.InferenceSession(walk_policy_file_name)

        self._last_action = [0] * 19
        self._shifted_action = [0] * 19

        self._policy_mode = None

        # own variable for logging
        self._log_file = open("/home/spot/spot-rl-deployment/spot-rl-example/python/log/observation_log.txt")

        # mask which joint should hold init state
        # true -> hold at init pos, false -> moves with policy
        self._hold_mask = [
            False, False, False, # front left
            False, False, False, # front right
            False, False, False, # hind left
            False, False, False, # hind right
            True, True, True, True, True, True, True, # arm 
        ]

    def __call__(self):
        """makes class a callable and computes model output for latest controller context

        return proto message to be used in spots command stream
        """

        # cache initial joint position when command stream starts
        if self._init_pos is None:
            self._init_pos = self._context.latest_state.joint_states.position
            self._init_load = self._context.latest_state.joint_states.load


        # extract observation data from latest spot state data
        # chosen config should no matter since init state should be the same
        input_list = self.collect_inputs(self._context.latest_state, self._walk_config)
        
        # execute model from onnx file
        input = [np.array(input_list).astype("float32")]
        
        #Check conditions for policy
        self._policy_mode = self.choose_policy_mode(input_list[9:12], input_list[0:3], input_list[3:6])
        if self._policy_mode == "standing":
            config = self._stand_config
            output = self._stand_session.run(None, {"obs": input})[0].tolist()[0]

        elif self._policy_mode == "walking":
            config = self._walk_config
            output = self._walk_session.run(None, {"obs": input})[0].tolist()[0]
        else:
            raise ValueError("_policy_mode is neither 'walking' or 'standing'")

        #apply action scaling and offset to output
        shifted_output = self.scale_and_shift_output(output, config)
        
        orbit_to_spot = find_ordering(ordered_joint_names_orbit, ordered_joint_names_bosdyn)
        reordered_output = reorder(shifted_output, orbit_to_spot)

        #Create proto message from target joint positions
        proto = self.create_proto(reordered_output, config)

        #cache data for history and logging
        self._last_action = output
        self._shifted_action = shifted_output
        self._count += 1
        self._context.count += 1

        return proto

    def collect_inputs(self, state: JointControlStreamRequest, config: OrbitConfig):
        """extract observation data from spots current state and format for onnx

        arguments
        state -- proto msg with spots latest state
        config -- model configuration data from orbit

        return list of float values ready to be passed into the model
        """
        observations = []
        observations += ob.get_base_linear_velocity(state)
        observations += ob.get_base_angular_velocity(state)
        observations += ob.get_projected_gravity(state)
        observations += self._context.velocity_cmd
        observations += ob.get_joint_positions(state, config)
        observations += ob.get_joint_velocity(state)
        observations += self._last_action
        
        if self.verbose and self._count%25==0:
            print_observations(observations)
            #print("[INFO] cmd", self._context.velocity_cmd)
            self.log_observations_to_file(observations)

        return observations

    def create_proto(self, pos_command: List[float], config: OrbitConfig):
        """generate a proto msg for spot with a given pos_command for dual policy

        arguments
        pos_command -- list of joint positions see spot.constants for order
        config -- config for policy

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = dict_to_list(config.kp, ordered_joint_names_bosdyn)
        k_qd_p = dict_to_list(config.kd, ordered_joint_names_bosdyn)

        N_DOF = len(pos_command)
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = pos_command[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = 0

        # Fill in gains the first dt
        if self._count == 1:
            update_proto.joint_command.gains.k_q_p.extend(k_q_p)
            update_proto.joint_command.gains.k_qd_p.extend(k_qd_p)

        update_proto.joint_command.position.extend(pos_cmd)
        update_proto.joint_command.velocity.extend(vel_cmd)
        update_proto.joint_command.load.extend(load_cmd)

        observation_time = self._context.latest_state.joint_states.acquisition_timestamp
        end_time = seconds_to_timestamp(timestamp_to_sec(observation_time) + 0.1)
        update_proto.joint_command.end_time.CopyFrom(end_time)

        # Let it extrapolate the command a little
        update_proto.joint_command.extrapolation_duration.nanos = int(5 * 1e6)

        # Set user key for latency tracking
        update_proto.joint_command.user_command_key = self._count
        return update_proto

    #TODO: should we log what policy we are using
    def log_observations_to_file(self, observations: List[float]):
        """feature to log observations into text file.
        
        arguments
        observations -- list of float values ready to be passed into the model
        """
        lines = [
            f"base_linear_velocity: {observations[0:3]}",
            f"base_angular_velocity: {observations[3:6]}",
            f"projected_gravity: {observations[6:9]}",
            f"commanded_vel: {observations[9:12]}",
            f"joint_positions: {observations[12:31]}",
            f"joint_velocity: {observations[31:50]}",
            f"last_action: {observations[50:69]}",
            f"shifted_action: {self._shifted_action}"
        ]

        for line in lines:
            self._log_file.write(line + "\n")

        self._log_file.flush()

    def close_logger_file(self):
        """Close text log file."""
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()

        
    def scale_and_shift_output(self, output: List[float], config: OrbitConfig) -> List[float]:
        """Apply action scale and default joint offset to output."""
        test_scale = min(0.1 * self._count, 1)

        scaled_output = list(map(mul, [config.action_scale] * 19, output))
        test_scaled = list(map(mul, [test_scale] * 19, scaled_output))

        default_joints = dict_to_list(config.default_joints, ordered_joint_names_orbit)
        shifted_output = list(map(add, test_scaled, default_joints))

        return shifted_output
    
    def choose_policy_mode(self, cmd_vel: List[float], base_lin_vel: List[float], base_ang_vel):
        """Helper function for choosing which policy to use, standing or walking.
        
        input
        cmd_vel - commanded velocity
        base_lin_vel - robot base velocity
        base_ang_vel

        return policy mode

        """
        cmd_norm = np.linalg.norm(cmd_vel)
        base_lin_norm = np.linalg.norm(base_lin_vel)
        base_ang_norm = np.linalg.norm(base_ang_vel)

        if base_lin_norm > 1  and cmd_norm < 1:
            print(f"base_speed is to high {base_lin_vel} while cmd is below threshold {cmd_vel}, can't use standing")
        
        can_use_standing = (
            cmd_norm < 1
            and base_lin_norm < 1
            and base_ang_norm < 1
        )

        must_use_walking = (
            cmd_norm > 1
            or base_lin_norm > 1
            or base_ang_norm > 1
        )

        mode_to_use = None
        if must_use_walking:
            mode_to_use = "walking"
        elif can_use_standing:
            mode_to_use = "standing"
        else:
            mode_to_use = "walking"

        if self._policy_mode != mode_to_use:
            print(f"Switching policy mode. Currently using {self._policy_mode}, now switching to {mode_to_use}")
        
        self._policy_mode = mode_to_use
        return mode_to_use