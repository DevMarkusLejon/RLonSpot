# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates keyboard-driven policy inference for Spot locomotion policies.

The script loads a JIT policy checkpoint, spawns a single Spot robot, and overrides the existing
``base_velocity`` command term from keyboard input so the walking policy can be tested
interactively.

Keyboard controls:

* ``W`` / ``S``: move forward / backward
* ``Q`` / ``E``: strafe left / right
* ``A`` / ``D``: yaw left / right

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for a Spot locomotion policy.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)
parser.add_argument(
    "--env-config",
    type=str,
    choices=("feet", "leg_only"),
    default="feet",
    help="Which Spot locomotion environment config to use. Keep 'feet' for the existing use case.",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=0,
    help="Optional number of simulation steps to run before exiting. Use 0 to run until the app closes.",
)
parser.add_argument(
    "--arm-mode",
    type=str,
    choices=("auto", "hold_default", "free"),
    default="auto",
    help="How to handle the arm. 'auto' holds the default arm pose in leg_only mode and leaves feet mode unchanged.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os

import carb
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.manager_based.sundtlejon.feet_spot_locomotion_env_cfg import SpotLocomotionEnvCfg_Play as feet
from isaaclab_tasks.manager_based.sundtlejon.spot_leg_only_locomotion_env_cfg import (
    SpotLegOnlyLocomotionEnvCfg_Play as leg_only,
)


class SpotKeyboardController:
    """Translate keyboard state into Spot base-velocity commands."""

    def __init__(self, env: ManagerBasedRLEnv):
        self._env = env
        self._device = env.device
        self._active_keys: set[str] = set()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        cmd_cfg = env.cfg.commands.base_velocity
        self._lin_x = max(abs(cmd_cfg.ranges.lin_vel_x[0]), abs(cmd_cfg.ranges.lin_vel_x[1]))
        self._lin_y = max(abs(cmd_cfg.ranges.lin_vel_y[0]), abs(cmd_cfg.ranges.lin_vel_y[1]))
        self._ang_z = max(abs(cmd_cfg.ranges.ang_vel_z[0]), abs(cmd_cfg.ranges.ang_vel_z[1]))

        self._desired_command = torch.zeros(env.scene.num_envs, 3, device=self._device)
        self._command_tensor = env.command_manager.get_command("base_velocity")

    def _on_keyboard_event(self, event):
        #key_name = event.input.name
        key_name = str(event.input).split(".")[-1]

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._active_keys.add(key_name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._active_keys.discard(key_name)
        return True

    def _compute_command(self) -> torch.Tensor:
        command = torch.zeros(3, device=self._device)

        if "W" in self._active_keys or "UP" in self._active_keys:
            command[0] += self._lin_x
        if "S" in self._active_keys or "DOWN" in self._active_keys:
            command[0] -= self._lin_x
        if "Q" in self._active_keys:
            command[1] += self._lin_y
        if "E" in self._active_keys:
            command[1] -= self._lin_y
        if "A" in self._active_keys or "LEFT" in self._active_keys:
            command[2] += self._ang_z
        if "D" in self._active_keys or "RIGHT" in self._active_keys:
            command[2] -= self._ang_z

        return command

    def apply(self):
        command = self._compute_command()
        self._desired_command[:] = command
        self._command_tensor.copy_(self._desired_command)


class ZeroCommandController:
    """Keep the command at zero for headless smoke tests."""

    def __init__(self, env: ManagerBasedRLEnv):
        self._command_tensor = env.command_manager.get_command("base_velocity")

    def apply(self):
        self._command_tensor.zero_()


class ArmDefaultPoseController:
    """Continuously hold the arm at the robot's configured default joint pose."""

    def __init__(self, env: ManagerBasedRLEnv):
        self._robot = env.scene["robot"]
        self._arm_joint_ids, self._arm_joint_names = self._robot.find_joints(["arm0.*"], preserve_order=True)
        self._arm_joint_pos_target = self._robot.data.default_joint_pos[:, self._arm_joint_ids].clone()

    @property
    def joint_names(self) -> list[str]:
        return list(self._arm_joint_names)

    def apply(self):
        self._robot.set_joint_position_target(self._arm_joint_pos_target, joint_ids=self._arm_joint_ids)


def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = feet() if args_cli.env_config == "feet" else leg_only()
    if args_cli.env_config == "leg_only":
        leg_joint_names = list(env_cfg.actions.joint_pos.joint_names)
        env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names = leg_joint_names
        env_cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names = leg_joint_names
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain.terrain_type = "plane"

    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 99999
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    keyboard_controller = ZeroCommandController(env) if args_cli.headless else SpotKeyboardController(env)
    hold_arm_default = args_cli.arm_mode == "hold_default" or (
        args_cli.arm_mode == "auto" and args_cli.env_config == "leg_only"
    )
    arm_controller = ArmDefaultPoseController(env) if hold_arm_default else None

    if args_cli.headless:
        print("Headless mode detected: holding zero base-velocity command.")
    else:
        print("Keyboard controls: W/S forward-back, Q/E strafe, A/D yaw.")
    print(f"Environment config: {args_cli.env_config}")
    if arm_controller is not None:
        print(f"Holding default arm pose for joints: {arm_controller.joint_names}")


    # run inference with the policy
    obs, _ = env.reset()
    keyboard_controller.apply()
    if arm_controller is not None:
        arm_controller.apply()
    obs = env.observation_manager.compute()
    step_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            keyboard_controller.apply()
            if arm_controller is not None:
                arm_controller.apply()
            obs = env.observation_manager.compute()
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)
#            print("joint names: ", env.action_manager.get_term("joint_pos")._joint_names)
#            print("leg targets (processed):", env.action_manager.get_term("joint_pos").processed_actions)
            if "arm_pos" in env.action_manager._terms:
                print("arm targets (processed):", env.action_manager.get_term("arm_pos").processed_actions)

            step_count += 1
            if args_cli.num_steps > 0 and step_count >= args_cli.num_steps:
                break




if __name__ == "__main__":
    main()
    simulation_app.close()
