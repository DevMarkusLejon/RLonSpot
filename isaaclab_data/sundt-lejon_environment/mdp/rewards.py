# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab.utils import math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


#This function is stolen from git https://github.com/isaac-sim/IsaacLab/compare/main...mschweig:IsaacLab:feat/add-spot-arm?diff=unified&w
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
#This function is also stolen from git https://github.com/isaac-sim/IsaacLab/compare/main...mschweig:IsaacLab:feat/add-spot-arm?diff=unified&w
def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_default_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.15,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward staying close to the articulation default joint pose during stand commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    rms_error = torch.sqrt(torch.mean(torch.square(joint_error), dim=1))
    reward = torch.exp(-rms_error / std)

    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, reward, torch.zeros_like(reward))


# ------------------ Codex generated reward functions ----------------

def _is_stand_command(env: ManagerBasedRLEnv, command_name: str, cmd_threshold: float) -> torch.Tensor:
    """Mask envs where commanded base velocity is near zero."""
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(cmd, dim=1)
    return cmd_mag < cmd_threshold


def base_xy_position_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize XY drift from each env origin while stand command is active."""
    asset: RigidObject = env.scene[asset_cfg.name]
    xy_err = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    penalty = torch.linalg.norm(xy_err, dim=1)
    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, penalty, torch.zeros_like(penalty))


def yaw_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    desired_yaw: float = 0.0,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize yaw drift from desired_yaw while stand command is active."""
    asset: RigidObject = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w  # assumed (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    yaw_err = yaw - desired_yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))  # wrap to [-pi, pi]
    penalty = torch.abs(yaw_err)

    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, penalty, torch.zeros_like(penalty))


def base_height_error_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize deviation from target base height while stand command is active."""
    asset: RigidObject = env.scene[asset_cfg.name]
    z_err = asset.data.root_pos_w[:, 2] - target_height
    penalty = torch.abs(z_err)

    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, penalty, torch.zeros_like(penalty))


def feet_contact_count_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward keeping feet in contact while stand command is active."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history  # [N, H, B, 3]
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    contact_count = torch.sum(is_contact.float(), dim=1)

    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, contact_count, torch.zeros_like(contact_count))


def stand_gated_air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    sensor_cfgs: list[SceneEntityCfg] | tuple[SceneEntityCfg, ...] | None = None,
    mode_time: float = 0.3,
    velocity_threshold: float = 0.5,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Disable air-time reward when the commanded base velocity is near zero."""
    reward = spot_mdp.air_time_reward(
        env=env,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
        sensor_cfgs=sensor_cfgs,
        mode_time=mode_time,
        velocity_threshold=velocity_threshold,
    )
    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, torch.zeros_like(reward), reward)


class StandGatedGaitReward(ManagerTermBase):
    """Wrap Spot gait reward and disable it during stand commands."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name = cfg.params.get("command_name", "base_velocity")
        self.cmd_threshold = cfg.params.get("cmd_threshold", 0.1)
        self._gait_reward = spot_mdp.GaitReward(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg | None = None,
        sensor_cfgs: list[SceneEntityCfg] | tuple[SceneEntityCfg, ...] | None = None,
    ) -> torch.Tensor:
        reward = self._gait_reward(
            env=env,
            std=std,
            max_err=max_err,
            velocity_threshold=velocity_threshold,
            synced_feet_pair_names=synced_feet_pair_names,
            asset_cfg=asset_cfg,
            sensor_cfg=sensor_cfg,
            sensor_cfgs=sensor_cfgs,
        )
        stand_mask = _is_stand_command(env, self.command_name, self.cmd_threshold)
        return torch.where(stand_mask, torch.zeros_like(reward), reward)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 1.0,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint motion and scale it up when a stand command is active."""
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = torch.linalg.norm(asset.data.joint_vel[:, asset_cfg.joint_ids], dim=1)
    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, stand_still_scale * penalty, penalty)


def alive_bonus(env: ManagerBasedRLEnv, value: float = 1.0) -> torch.Tensor:
    """Small constant positive reward every step."""
    return torch.full((env.num_envs,), value, device=env.device)


#Codex did this
def _lleg_contact_masks(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 5.0,
    foot_z_max: float = -0.28,
    foot_xy_radius: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify lower-leg contacts into desired (foot-zone) and undesired (non-foot-zone)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if sensor.data.force_matrix_w_history is None or sensor.data.contact_pos_w is None:
        raise RuntimeError(
            f"Sensor '{sensor_cfg.name}' must enable filtered contacts and track_contact_points for foot-zone checks."
        )
    if sensor.data.pos_w is None or sensor.data.quat_w is None:
        raise RuntimeError(f"Sensor '{sensor_cfg.name}' must enable track_pose for foot-zone checks.")

    # Single-leg sensors are configured as one body vs filtered ground shapes.
    contact_force = torch.linalg.norm(sensor.data.force_matrix_w_history[:, 0, 0, :, :], dim=-1)
    has_force = contact_force > force_threshold

    pts_w = sensor.data.contact_pos_w[:, 0, :, :]
    valid_pts = ~torch.isnan(pts_w[..., 0])
    is_contact = has_force & valid_pts

    leg_pos_w = sensor.data.pos_w[:, 0, :]
    leg_quat_w = sensor.data.quat_w[:, 0, :]
    rel_w = pts_w - leg_pos_w.unsqueeze(1)
    num_envs, num_pts, _ = rel_w.shape

    rel_l = math_utils.quat_apply_inverse(
        leg_quat_w.unsqueeze(1).expand(-1, num_pts, -1).reshape(-1, 4),
        rel_w.reshape(-1, 3),
    ).reshape(num_envs, num_pts, 3)

    in_foot_zone = (rel_l[..., 2] < foot_z_max) & (torch.linalg.norm(rel_l[..., :2], dim=-1) < foot_xy_radius)
    desired = torch.any(is_contact & in_foot_zone, dim=1)
    undesired = torch.any(is_contact & ~in_foot_zone, dim=1)
    return desired, undesired


#Codex did this
def feet_in_foot_zone_reward(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float = 5.0,
    foot_z_max: float = -0.28,
    foot_xy_radius: float = 0.06,
) -> torch.Tensor:
    """Reward lower-leg contacts occurring in the distal foot-like zone."""
    per_leg = []
    for sensor_cfg in sensor_cfgs:
        desired, _ = _lleg_contact_masks(env, sensor_cfg, force_threshold, foot_z_max, foot_xy_radius)
        per_leg.append(desired.float())
    return torch.stack(per_leg, dim=1).sum(dim=1)


#Codex did this
def lleg_non_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float = 5.0,
    foot_z_max: float = -0.28,
    foot_xy_radius: float = 0.06,
) -> torch.Tensor:
    """Penalize lower-leg contacts outside the distal foot-like zone."""
    per_leg = []
    for sensor_cfg in sensor_cfgs:
        _, undesired = _lleg_contact_masks(env, sensor_cfg, force_threshold, foot_z_max, foot_xy_radius)
        per_leg.append(undesired.float())
    return torch.stack(per_leg, dim=1).sum(dim=1)


#Codex did this
def any_lleg_non_foot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float = 5.0,
    foot_z_max: float = -0.28,
    foot_xy_radius: float = 0.06,
) -> torch.Tensor:
    """Terminate when any lower leg makes non-foot-zone contact with the ground."""
    per_leg = []
    for sensor_cfg in sensor_cfgs:
        _, undesired = _lleg_contact_masks(env, sensor_cfg, force_threshold, foot_z_max, foot_xy_radius)
        per_leg.append(undesired)
    return torch.stack(per_leg, dim=1).any(dim=1)


def filtered_contact_force_exceeds(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when filtered contact force exceeds threshold on selected bodies.

    Unlike ``illegal_contact`` (which uses unfiltered net forces), this function uses
    ``force_matrix_w_history`` so the result respects ``filter_prim_paths_expr``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    filtered_forces = contact_sensor.data.force_matrix_w_history
    if filtered_forces is None:
        raise RuntimeError(
            f"Sensor '{sensor_cfg.name}' must define filter_prim_paths_expr to use filtered_contact_force_exceeds."
        )

    body_ids = sensor_cfg.body_ids
    if isinstance(body_ids, list) and len(body_ids) == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Shape after norm: [num_envs, history, num_selected_bodies, num_filtered_targets]
    force_norm = torch.norm(filtered_forces[:, :, body_ids, :, :], dim=-1)
    if force_norm.shape[-1] == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Max over filtered targets and history, then any over selected bodies.
    max_over_targets = torch.max(force_norm, dim=3)[0]
    max_over_history = torch.max(max_over_targets, dim=1)[0]
    return torch.any(max_over_history > threshold, dim=1)


def drawer_opened_above_threshold(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when the drawer joint position reaches the configured open threshold."""
    cabinet: Articulation = env.scene[asset_cfg.name]
    drawer_pos = cabinet.data.joint_pos[:, asset_cfg.joint_ids[0]]
    return drawer_pos >= threshold


def backward_base_when_drawer_pulled(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    cabinet_cfg: SceneEntityCfg,
    pull_threshold: float = 0.02,
    full_open_pos: float = 0.30,
    target_backward_speed: float = 0.15,
) -> torch.Tensor:
    """Reward backward base motion once the drawer has started opening.

    The term is zero before the drawer reaches ``pull_threshold`` and increases
    with both drawer opening progress and backward base speed.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    cabinet: Articulation = env.scene[cabinet_cfg.name]

    drawer_pos = cabinet.data.joint_pos[:, cabinet_cfg.joint_ids[0]]
    base_vx_body = robot.data.root_lin_vel_b[:, 0]
    backward_speed = torch.clamp(-base_vx_body, min=0.0)

    open_progress = torch.clamp((drawer_pos - pull_threshold) / max(full_open_pos - pull_threshold, 1.0e-6), 0.0, 1.0)
    speed_score = torch.clamp(backward_speed / max(target_backward_speed, 1.0e-6), 0.0, 1.0)

    active = drawer_pos > pull_threshold
    reward = open_progress * speed_score
    return torch.where(active, reward, torch.zeros_like(reward))


def backward_ee_when_drawer_pulled(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    cabinet_cfg: SceneEntityCfg,
    ee_frame_name: str = "ee_frame",
    pull_threshold: float = 0.02,
    full_open_pos: float = 0.30,
    retreat_start_x: float = 0.22,
    retreat_span: float = 0.18,
) -> torch.Tensor:
    """Reward end-effector retreat in base frame after the drawer starts opening.

    This variant is intended for fixed-base runs where root backward motion is impossible.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    cabinet: Articulation = env.scene[cabinet_cfg.name]

    drawer_pos = cabinet.data.joint_pos[:, cabinet_cfg.joint_ids[0]]
    ee_tcp_pos_w = env.scene[ee_frame_name].data.target_pos_w[..., 0, :]

    ee_rel_w = ee_tcp_pos_w - robot.data.root_pos_w[:, :3]
    ee_pos_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, ee_rel_w)

    open_progress = torch.clamp((drawer_pos - pull_threshold) / max(full_open_pos - pull_threshold, 1.0e-6), 0.0, 1.0)
    retreat_score = torch.clamp((retreat_start_x - ee_pos_b[:, 0]) / max(retreat_span, 1.0e-6), 0.0, 1.0)

    active = drawer_pos > pull_threshold
    reward = open_progress * retreat_score
    return torch.where(active, reward, torch.zeros_like(reward))


def base_motion_stand_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    cmd_threshold: float = 0.12,
    lin_scale: float = 1.0,
    ang_scale: float = 0.5,
) -> torch.Tensor:
    """Penalize planar base motion when the commanded motion is near zero."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_xy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
    penalty = lin_scale * lin_xy + ang_scale * yaw_rate

    stand_mask = _is_stand_command(env, command_name, cmd_threshold)
    return torch.where(stand_mask, penalty, torch.zeros_like(penalty))
