# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.
import re
import os
from dataclasses import dataclass
from operator import add, mul
from threading import Event
from typing import List
from datetime import datetime

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


def print_observations(observations: List[float], USED_DOF: int = 19):
    """debug function to print out the observation data used as model input

    arguments
    observations -- list of float values ready to be passed into the model
    """
    
    print("Joint order names:", ordered_joint_names_orbit)
    print("base_linear_velocity:", observations[0:3])
    print("base_angular_velocity:", observations[3:6])
    print("projected_gravity:", observations[6:9])
    print("commanded_vel:", observations[9:12])
    print("joint_positions:", observations[12:12+USED_DOF])
    print("joint_velocity:", observations[12+USED_DOF:12+USED_DOF*2])
    print("last_action:", observations[12+USED_DOF*2:12+USED_DOF*3])

    # print("joint_positions:", observations[12:31])
    # print("joint_velocity:", observations[31:50])
    # print("last_action:", observations[50:69])

class OnnxCommandGenerator:
    """class to be used as generator for spots command stream that executes
    an onnx model and converts the output to a spot command"""

    def __init__(
        self, context: OnnxControllerContext, config: OrbitConfig, policy_file_name: os.PathLike, verbose: bool
    ):
        self._context = context
        self._config = config
        self._inference_session = ort.InferenceSession(policy_file_name)
        self._last_action = [0] * 19
        self._shifted_action = [0] * 19
        self._count = 1
        self._init_pos = None
        self._init_load = None
        self.verbose = verbose
      
        # open a .txt file for logging, finds the next number in order used for the file name
        log_dir = "/home/spot/spot-rl-deployment/spot-rl-example/python/logs/"
        log_prefix = "logfile_"
        log_suffix = ".txt"
        file_number = self.get_next_log_file_number(log_dir, log_prefix, log_suffix)
        self.filename = log_dir + log_prefix + str(file_number) + log_suffix
        self._log_file = open(self.filename, "w")
        print("Saving log in file: ", self.filename)

        # own mask variable used in create proto blend defined in spot/bosdyn order
        # if true holds the init/fixed pose, if false move with pos_command from policy
        self._hold_mask = [
            False, False, False, # fl: hx, hy, kn
            False, False, False, # fr: hx, hy, kn
            False, False, False, # hl: hx, hy, kn
            False, False, False, # hr: hx, hy, kn
            True, True, True, True, True, True, True, # arm 
        ]
        self._TOTAL_DOF = 19
        self._USED_DOF = 12
        
        policy_input_len = self._inference_session.get_inputs()[0].shape[-1] #len of first input list for the policy
        policy_input_dof = 12 if policy_input_len == 48 else 19 if policy_input_len == 69 else None
        print(f"Policy len: {policy_input_len}. Policy DOF: {policy_input_dof}.")
        print(f"Full DOF: {self._TOTAL_DOF}. Used DOF: {self._USED_DOF}.")
        print(f"USED_DOF and Policy DOF  "
              f"{'MATCH, GO AHEAD AND RUN!' if self._USED_DOF == policy_input_dof else 'DO NOT MATCH, FIX BEFORE RUNNING!'}")


    def __call__(self):
        """makes class a callable and computes model output for latest controller context

        return proto message to be used in spots command stream
        """

        # cache initial joint position when command stream starts
        if self._init_pos is None:
            self._init_pos = self._context.latest_state.joint_states.position
            self._init_load = self._context.latest_state.joint_states.load
            print("Spot init pos in spot order, printed in onnx __call: ", self._init_pos) # own print
            print("Spot init load in spot order, printed in onnx __call: ", self._init_load) # own print
            spot_to_orbit = find_ordering(ordered_joint_names_bosdyn, ordered_joint_names_orbit)
            print("Spot init pos in orbit order, printed in onnx __call: ", reorder(self._init_pos, spot_to_orbit)) # own print
            print("Spot init load in orbit order, printed in onnx __call: ", reorder(self._init_load, spot_to_orbit)) # own print


        # extract observation data from latest spot state data
        input_list = self.collect_inputs(self._context.latest_state, self._config)
        # print("observations", input_list)

        # execute model from onnx file
        input = [np.array(input_list).astype("float32")]
        output = self._inference_session.run(None, {"obs": input})[0].tolist()[0]
       
        # post process model output apply action scaling and return to spots
        # joint order and offset
        test_scale = min(0.1 * self._count, 1) #0.1 is standard value

        scaled_output = list(map(mul, [self._config.action_scale] * self._USED_DOF, output))
        test_scaled = list(map(mul, [test_scale] * self._USED_DOF, scaled_output))

        default_joints = dict_to_list(self._config.default_joints, ordered_joint_names_orbit)[:self._USED_DOF]
        shifted_output = list(map(add, test_scaled, default_joints))
        print("shifted output in orbit order: \n", shifted_output) # own print 

        orbit_to_spot = find_ordering(ordered_joint_names_orbit, ordered_joint_names_bosdyn)[:self._USED_DOF]
        reordered_output = reorder(shifted_output, orbit_to_spot)
        print("shifted output in bosdyn order: \n", reordered_output) # own print
        
        # generate proto message from target joint positions
        #proto = self.create_proto(reordered_output)
        #proto = self.create_proto_hold()
        #proto = self.create_proto_blend_hold(reordered_output)

        proto = self.create_proto_blend_fixed(reordered_output)
        # if self._count < 100:
        #     proto = self.create_proto_fixed()
        # else:
        #     proto = self.create_proto_blend_fixed(reordered_output)
        

        # cache data for history and logging
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
        observations += ob.get_joint_positions(state, config)[:self._USED_DOF]
        observations += ob.get_joint_velocity(state)[:self._USED_DOF]
        observations += self._last_action[:self._USED_DOF]

        load = ob.get_joint_torques(state)

        if self._count % 50 == 0:
            print_observations(observations, self._USED_DOF)
            print(f"shifted_action: {self._shifted_action[:self._USED_DOF]}")
            print(f"Joint load: {load}")
            #print("[INFO] cmd", self._context.velocity_cmd)
        self.log_observations_txt(observations)

        return observations

    def log_observations_txt(self, observations):
        """save observations into txt file.

        arguments
        observations -- list of float values ready to be passed into the model
        """
        lines = [
            f"base_linear_velocity: {observations[0:3]}",
            f"base_angular_velocity: {observations[3:6]}",
            f"projected_gravity: {observations[6:9]}",
            f"commanded_vel: {observations[9:12]}",
            f"joint_positions: {observations[12:12+self._USED_DOF]}",
            f"joint_velocity: {observations[12+self._USED_DOF:12+self._USED_DOF*2]}",
            f"last_action: {observations[12+self._USED_DOF*2:12+self._USED_DOF*3]}",
            f"shifted_action: {self._shifted_action[:self._USED_DOF]}"
        ]
            #f"joint_positions: {observations[12:31]}",
            #f"joint_velocity: {observations[31:50]}",
            #f"last_action: {observations[50:69]}",

        for line in lines:
            self._log_file.write(line + "\n")
        self._log_file.flush()

    def close_log_file(self):
        """Close the log txt file."""
        print("Closing log file with name: ", self.filename)
        self._log_file.close()
    
    def get_next_log_file_number(self, log_dir: str, prefix: str = "logfile_", suffix: str = ".txt") -> str:
        os.makedirs(log_dir, exist_ok=True)
        pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}")
        
        existing_numbers = []
        for filename in os.listdir(log_dir):
            match = pattern.match(filename)
            if match:
                existing_numbers.append(int(match.group(1)))
            
        return max(existing_numbers, default=0) + 1  


    def create_proto(self, pos_command: List[float]):
        """generate a proto msg for spot with a given pos_command

        arguments
        pos_command -- list of joint positions see spot.constants for order

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = dict_to_list(self._config.kp, ordered_joint_names_bosdyn)
        k_qd_p = dict_to_list(self._config.kd, ordered_joint_names_bosdyn)

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

    def create_proto_hold(self):
        """generate a proto msg that holds spots current pose useful for debugging

        return proto message to send in spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        update_proto.Clear()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        
        N_DOF = 19 #OG 12 CHANGED TO 19 TO INCLUDE ARM

        k_q_p = DEFAULT_K_Q_P[0:N_DOF] # 
        k_qd_p = DEFAULT_K_QD_P[0:N_DOF] #

        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            pos_cmd[joint_ind] = self._init_pos[joint_ind]
            vel_cmd[joint_ind] = 0
            load_cmd[joint_ind] = self._init_load[joint_ind]
        
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

    def create_proto_blend_hold(self, pos_command: List[float]):
        """generate a proto msg for spot with a blend of current pose and pos_command

        arguments
        pos_command -- list of joint positions see spot.constants for order

        return proto message to send i spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        N_DOF = len(pos_command)

        config_kp = dict_to_list(self._config.kp, ordered_joint_names_bosdyn)
        config_kd = dict_to_list(self._config.kd, ordered_joint_names_bosdyn)

        default_kp = DEFAULT_K_Q_P[0:N_DOF] #
        default_kd = DEFAULT_K_QD_P[0:N_DOF] #

        k_q_p = [0] * N_DOF
        k_qd_p = [0] * N_DOF

        for i in range (N_DOF):
            if self._hold_mask[i]:
                # use def gains
                k_q_p[i] = default_kp[i]
                k_qd_p[i] = default_kd[i]
            else:
                # use conf gains
                k_q_p[i] = config_kp[i]
                k_qd_p[i] = config_kd[i]


        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

        for joint_ind in range(N_DOF):
            if self._hold_mask[joint_ind]:
                # if true hold at init pos
                pos_cmd[joint_ind] = self._init_pos[joint_ind]
                load_cmd[joint_ind] = self._init_load[joint_ind]
            else:
                # if false use policy output
                pos_cmd[joint_ind] = pos_command[joint_ind]
                load_cmd[joint_ind] = 0

            # vel is zero for both
            vel_cmd[joint_ind] = 0

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

    def create_proto_fixed(self):
        """generate a proto msg that holds spot at a fixed pose

        return proto message to send to spot command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = dict_to_list(self._config.kp, ordered_joint_names_bosdyn)
        k_qd_p = dict_to_list(self._config.kd, ordered_joint_names_bosdyn)

        N_DOF = 19
        
        pos_cmd = [
                0.1, 0.9, -1.5,
                -0.1, 0.9, -1.5,
                0.1, 1.1, -1.5,
                -0.1, 1.1, -1.5,
                0, -3.14, 3.14, 1.57, 0, -1.57, 0]
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF

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

    def create_proto_blend_fixed(self, pos_command: List[float]):
        """generate a proto msg for spot with a blend of fixed pose and pos_command

        arguments
        pos_command -- list of joint positions see spot.constants for order

        return proto message to send i spots command stream
        """
        update_proto = robot_command_pb2.JointControlStreamRequest()
        set_timestamp_from_now(update_proto.header.request_timestamp)
        update_proto.header.client_name = "rl_example_client"

        k_q_p = dict_to_list(self._config.kp, ordered_joint_names_bosdyn)
        k_qd_p = dict_to_list(self._config.kd, ordered_joint_names_bosdyn)

        N_DOF = 19
        
        pos_cmd_fixed = [
                0.1, 0.9, -1.5,
                -0.1, 0.9, -1.5,
                0.1, 1.1, -1.5,
                -0.1, 1.1, -1.5,
                0, -3.14, 3.14, 1.57, 0, -1.57, 0]
        
        pos_cmd = [0] * N_DOF
        vel_cmd = [0] * N_DOF
        load_cmd = [0] * N_DOF
        
        for joint_ind in range(N_DOF):
            if self._hold_mask[joint_ind] or joint_ind >= len(pos_command):
                #if true hold at fixed pose
                pos_cmd[joint_ind] = pos_cmd_fixed[joint_ind]
            else:
                pos_cmd[joint_ind] = pos_command[joint_ind]


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
