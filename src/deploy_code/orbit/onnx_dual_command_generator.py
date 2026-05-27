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


class OnnxDualCommandGenerator:
    """class to be used as generator for spots command stream that executes
    an onnx model and converts the output to a spot command"""

    def __init__(
        self, 
        context: OnnxControllerContext, 
        body_config: OrbitConfig, 
        arm_config: OrbitConfig,
        body_policy_file_name: os.PathLike, 
        arm_policy_file_name: os.PathLike,
        verbose: bool
    ):
        self._context = context
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose

        self._body_config = body_config
        self._arm_config = arm_config
        self._body_session = ort.InferenceSession(body_policy_file_name)
        self._arm_session = ort.InferenceSession(arm_policy_file_name)

        self._last_action = [0] * 19
        self._body_last_action = [0] * 19
        self._arm_last_action = [0] * 19

        self._shifted_action = [0] * 19

        self._body_dof = 12
        self._arm_dof = 7
        self._total_dof = 19

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

        #TODO: Find out what policy wants. Its own latest action or the one sent to the robot.

        body_input_list, arm_input_list = self.collect_dual_inputs(self._context.latest_state, self._body_config, self._arm_config, self._body_last_action, self._arm_last_action)
        
        body_input = [np.array(body_input_list).astype("float32")]
        body_output = self._body_session.run(None, {"obs": body_input})[0].tolist()[0]
        body_shifted_output = self.postprocess_output(body_output, self._body_config)

        arm_input = [np.array(arm_input_list).astype("float32")]
        arm_output = self._arm_session.run(None, {"obs": arm_input})[0].tolist()[0]
        arm_shifted_output = self.postprocess_output(arm_output, self._arm_config)

        """
        #Body
        body_input_list = self.collect_inputs(self._context.latest_state, self._body_config, self._body_last_action)
        body_input = [np.array(body_input_list).astype("float32")]
        body_output = self._body_session.run(None, {"obs": body_input})[0].tolist()[0]
        
        body_shifted_output = self.postprocess_output(body_output, self._body_config)
        
        #Arm
        arm_input_list = self.collect_inputs(self._context.latest_state, self._arm_config, self._arm_last_action)
        arm_input = [np.array(arm_input_list).astype("float32")]
        arm_output = self._arm_session.run(None, {"obs": arm_input})[0].tolist()[0]
        
        arm_shifted_output = self.postprocess_output(arm_output, self._arm_config)
        """
        #Merge
        merged_shifted_output = self.merge_policy_output(body_shifted_output, arm_shifted_output)
        
        #Reorder
        orbit_to_spot = find_ordering(ordered_joint_names_orbit, ordered_joint_names_bosdyn)
        merged_reordered_output = reorder(merged_shifted_output, orbit_to_spot)

        #Create proto msg
        proto = self.create_proto_dual(merged_reordered_output, self._body_config, self._arm_config)

        
        self._body_last_action = body_output
        self._arm_last_action = arm_output
        self._shifted_action = merged_shifted_output
        self._count += 1
        self._context.count += 1

        return proto

    def collect_inputs(self, state: JointControlStreamRequest, config: OrbitConfig, last_action: List[float]):
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
        observations += last_action
        
        if self.verbose and self._count%25==0:
            print_observations(observations)
            #print("[INFO] cmd", self._context.velocity_cmd)
            self.log_observations_to_file(observations)

        return observations
    
    def collect_dual_inputs(self, state: JointControlStreamRequest, body_config: OrbitConfig, arm_config: OrbitConfig, body_action: List[float], arm_action: List[float]):
        """extract observation data from spots current state and format for onnx

        arguments
        state -- proto msg with spots latest state
        config -- model configuration data from orbit

        return two list of float values ready to be passed into the corresponding model
        """
        body_observations = []
        arm_observations = []

        base_lin_vel = ob.get_base_linear_velocity(state)
        base_ang_vel = ob.get_base_angular_velocity(state)
        projected_gravity = ob.get_projected_gravity(state)
        commanded_vel = self._context.velocity_cmd
        joint_vel = ob.get_joint_velocity(state)

        body_observations += base_lin_vel
        body_observations += base_ang_vel
        body_observations += projected_gravity
        body_observations += commanded_vel
        body_observations += ob.get_joint_positions(state, body_config)
        body_observations += joint_vel
        body_observations += body_action

        arm_observations += base_lin_vel
        arm_observations += base_ang_vel
        arm_observations += projected_gravity
        arm_observations += commanded_vel
        arm_observations += ob.get_joint_positions(state, arm_config)
        arm_observations += joint_vel
        arm_observations += arm_action

        if self.verbose and self._count%25==0:
            print_observations(body_observations)
            print_observations(arm_observations)
            #print("[INFO] cmd", self._context.velocity_cmd)
            #self.log_observations_to_file(observations)
        return body_observations, arm_observations



    def create_proto_dual(self, pos_command: List[float], body_config: OrbitConfig, arm_config: OrbitConfig):
        """generate a proto msg for spot with a given pos_command for dual policy

        arguments
        pos_command -- list of joint positions see spot.constants for order
        body_config -- config for body policy
        arm_config -- config for arm policy

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        body_kp = dict_to_list(body_config.kp, ordered_joint_names_bosdyn)
        body_kd = dict_to_list(body_config.kd, ordered_joint_names_bosdyn)

        arm_kp = dict_to_list(arm_config.kp, ordered_joint_names_bosdyn)
        arm_kd = dict_to_list(arm_config.kd, ordered_joint_names_bosdyn)

        k_q_p = body_kp[:self._body_dof] + arm_kp[self._body_dof:]
        k_qd_p = body_kd[:self._body_dof] + arm_kd[self._body_dof:]

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

    #TODO: The logging must be fixed, cant simply log last action with dual policy
    def log_observations_to_file(self, observations: List[float], body_action: List[float], arm_action: List[float]):
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
            #f"last_action: {observations[50:69]}",
            f"body_action: {body_action}",
            f"arm_action: {arm_action}",
            f"shifted_action: {self._shifted_action}"
        ]

        for line in lines:
            self._log_file.write(line + "\n")

        self._log_file.flush()

    def close_logger_file(self):
        """Close text log file."""
        if hasattr(self, "_log_file") and self._log_file:
            self._log_file.close()

        
    def postprocess_output(self, output: List[float], config: OrbitConfig) -> List[float]:
        """Apply action scale and default joint offset to output."""
        test_scale = min(0.1 * self._count, 1)

        scaled_output = list(map(mul, [config.action_scale] * 19, output))
        test_scaled = list(map(mul, [test_scale] * 19, scaled_output))

        default_joints = dict_to_list(config.default_joints, ordered_joint_names_orbit)
        shifted_output = list(map(add, test_scaled, default_joints))

        return shifted_output
    
    def merge_policy_output(self, body_output: List[float], arm_output: List[float]) -> List[float]:
        """Merge two 19-DOF Orbit-order outputs.

        First 12 joints come from body policy.
        Last 7 joints come from arm policy.
        """
        return body_output[:self._body_dof] + arm_output[self._body_dof:]