# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates keyboard-driven policy inference for Spot with two policies.

It keeps a single locomotion-style environment alive, but switches between a locomotion
policy and a standing policy based on the magnitude of the commanded base velocity.

Keyboard controls:

* ``W`` / ``S``: move forward / backward
* ``Q`` / ``E``: strafe left / right
* ``A`` / ``D``: yaw left / right
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Spot with locomotion and standing policies.")
parser.add_argument(
    "--locomotion-checkpoint",
    type=str,
    default="logs/rsl_rl/spot_SL_locomotion_flat/2026-04-01_12-53-55/exported/policy.pt",
    help="Path to the locomotion policy exported as TorchScript/JIT.",
)
parser.add_argument(
    "--standing-checkpoint",
    type=str,
    default="logs/rsl_rl/spot_SL_locomotion_flat/2026-04-02_09-55-04/exported/policy.pt",
    help="Path to the standing policy exported as TorchScript/JIT.",
)
parser.add_argument("--standing-threshold", type=float, default=0.35, help="Enter standing mode below this command norm.")
parser.add_argument(
    "--standing-exit-threshold",
    type=float,
    default=0.25,
    help="Leave standing mode above this command norm. Keep this above --standing-threshold.",
)
parser.add_argument("--base-speed-threshold", type=float, default=0.91, help="Enter standing mode only when body XY speed is below this threshold.")
parser.add_argument(
    "--base-speed-exit-threshold",
    type=float,
    default= 0.15*8,
    help="Leave standing mode when body XY speed rises above this threshold.",
)
parser.add_argument("--base-yaw-rate-threshold", type=float, default=0.15, help="Enter standing mode only when body yaw rate is below this threshold.")
parser.add_argument(
    "--base-yaw-rate-exit-threshold",
    type=float,
    default=0.3,
    help="Leave standing mode when body yaw rate rises above this threshold.",
)
parser.add_argument(
    "--accelerate-command",
    action="store_true",
    help="Ramp keyboard base-velocity commands toward their targets instead of applying them instantly.",
)
parser.add_argument(
    "--command-acceleration",
    type=float,
    default=8.0,
    help="Maximum commanded-velocity change per second when --accelerate-command is enabled.",
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
from isaaclab_tasks.manager_based.sundtlejon.feet_spot_locomotion_env_cfg import SpotLocomotionEnvCfg_Play


class SpotKeyboardController:
    """Translate keyboard state into Spot base-velocity commands."""

    def __init__(self, env: ManagerBasedRLEnv, accelerate_command: bool = False, command_acceleration: float = 8.0):
        self._env = env
        self._device = env.device
        self._step_dt = env.unwrapped.step_dt
        self._accelerate_command = accelerate_command
        self._command_acceleration = max(command_acceleration, 0.0)
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

    @property
    def command(self) -> torch.Tensor:
        return self._command_tensor[0]

    def _on_keyboard_event(self, event):
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
        target_command = self._compute_command()
        self._desired_command[:] = target_command
        if self._accelerate_command:
            max_delta = self._command_acceleration * self._step_dt
            delta = torch.clamp(target_command - self._command_tensor[0], min=-max_delta, max=max_delta)
            command = self._command_tensor[0] + delta
        else:
            command = target_command
        self._command_tensor[:] = command


def _load_jit_policy(policy_path: str, device: str) -> torch.jit.ScriptModule:
    policy_path = os.path.abspath(policy_path)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    return torch.jit.load(file, map_location=device)


def main():
    """Main function."""
    locomotion_policy = _load_jit_policy(args_cli.locomotion_checkpoint, args_cli.device)
    standing_policy = _load_jit_policy(args_cli.standing_checkpoint, args_cli.device)

    # Use the locomotion env interface as the shared runtime contract.
    env_cfg = SpotLocomotionEnvCfg_Play()

    # defualt values are 
    # lin_vel_x=(-2.5, 3.0)
    # lin_vel_y=(-2.5, 2.5)
    # ang_vel_z=(-1.5, 1.5)
    env_cfg.commands.base_velocity.debug_vis = True
    #env_cfg.commands.base_velocity.resampling_time_range = (0.0,0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-1.0*3,1.0*3)#(-2.0*2.5, 2.0*2.5)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-1.0*3,1.0*3)#(-2.0*2.5, 2.0*2.5)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0*3,1.0*3)#(-1.2*2.5, 1.2*1.5)
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.4
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.observations.policy.enable_corruption = True#False
    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 999999
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    env = ManagerBasedRLEnv(cfg=env_cfg)
    keyboard_controller = SpotKeyboardController(
        env,
        accelerate_command=args_cli.accelerate_command,
        command_acceleration=args_cli.command_acceleration,
    )

    action_index = 0
    print("Policy action order:")
    for term_name, term in env.action_manager._terms.items():
        joint_names = getattr(term, "_joint_names", None)
        if joint_names is None:
            print(f"  {term_name}: {term.action_dim} dims")
            action_index += term.action_dim
            continue
        for joint_name in joint_names:
            print(f"  [{action_index:02d}] {joint_name} ({term_name})")
            action_index += 1

    print("Keyboard controls: W/S forward-back, Q/E strafe, A/D yaw.")
    print(
        f"Switching to standing policy below command norm {args_cli.standing_threshold:.3f}, body speed "
        f"{args_cli.base_speed_threshold:.3f}, and yaw rate {args_cli.base_yaw_rate_threshold:.3f}."
    )
    print(
        f"Returning to locomotion above command norm {args_cli.standing_exit_threshold:.3f}, body speed "
        f"{args_cli.base_speed_exit_threshold:.3f}, or yaw rate {args_cli.base_yaw_rate_exit_threshold:.3f}."
    )

    obs, _ = env.reset()
    keyboard_controller.apply()
    obs = env.observation_manager.compute()
    use_standing = False
    last_mode = None

    with torch.inference_mode():
        while simulation_app.is_running():
            keyboard_controller.apply()
            obs = env.observation_manager.compute()

            cmd_norm = torch.linalg.norm(keyboard_controller.command).item()
            if(cmd_norm < 0.1):
                cmd = env.command_manager.get_command("base_velocity")[0]
                cmd_norm = torch.linalg.norm(cmd).item()
            robot = env.scene["robot"]
            base_speed = torch.linalg.norm(robot.data.root_lin_vel_b[0, :2]).item()
            base_yaw_rate = torch.abs(robot.data.root_ang_vel_b[0, 2]).item()
            if(base_speed > args_cli.base_speed_threshold and cmd_norm == 0):
                print("base_speed is above threshold for entering standing but no command norm is given",base_speed)

            can_enter_standing = (
                cmd_norm < args_cli.standing_threshold
                and base_speed < args_cli.base_speed_threshold
                and base_yaw_rate < args_cli.base_yaw_rate_threshold
            )
            must_leave_standing = (
                cmd_norm > args_cli.standing_exit_threshold
                or base_speed > args_cli.base_speed_exit_threshold
                or base_yaw_rate > args_cli.base_yaw_rate_exit_threshold
            )

            if use_standing:
                use_standing = not must_leave_standing
            else:
                use_standing = can_enter_standing

            mode = "standing" if use_standing else "locomotion"
            if mode != last_mode:
                print(
                    f"Policy mode: {mode} (command norm: {cmd_norm:.3f}, body speed: {base_speed:.3f}, yaw rate: {base_yaw_rate:.3f})"
                )
                last_mode = mode

            if use_standing:
                #print("im using standing")
                action = standing_policy(obs["policy"])
            else:
                action = locomotion_policy(obs["policy"])

            env.action_manager.process_action(action)
            desired_joint_pos = torch.cat(
                [term.processed_actions for term in env.action_manager._terms.values()], dim=1
            )
            #print(desired_joint_pos)
            
            obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
