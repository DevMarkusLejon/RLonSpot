# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot feet manipulation task built on the standing environment.

This variant removes the standing task's two-phase curriculum and adds a random
end-effector position command in front of the robot while keeping balance rewards.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as reach_mdp

from . import mdp
from .feet_spot_standing_env_cfg import SpotCommandsCfg as StandingCommandsCfg
from .feet_spot_standing_env_cfg import (
    SpotFeetStandingEnvCfg,
    SpotObservationsCfg as StandingObservationsCfg,
    SpotRewardsCfg as StandingRewardsCfg,
)


@configclass
class SpotFeetManipualtionCommandsCfg(StandingCommandsCfg):
    """Commands for balanced random-point gripper reaching."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="arm0_link_fngr",
        resampling_time_range=(2.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35*1, 0.70*3),
            pos_y=(-0.25*1, 0.25*3),
            pos_z=(0.00, 0.35*3),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SpotFeetManipualtionObservationsCfg(StandingObservationsCfg):
    """Observations for standing + end-effector target tracking."""

    @configclass
    class PolicyCfg(StandingObservationsCfg.PolicyCfg):
        velocity_commands = None
        ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotFeetManipualtionRewardsCfg(StandingRewardsCfg):
    """Standing rewards plus end-effector position tracking."""

    end_effector_position_tracking = RewTerm(
        func=reach_mdp.position_command_error,
        weight=-0.40*4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm0_link_fngr"),
            "command_name": "ee_pose",
        },
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=reach_mdp.position_command_error_tanh,
        weight=1.00*4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="arm0_link_fngr"),
            "std": 0.15,
            "command_name": "ee_pose",
        },
    )


@configclass
class SpotFeetManipualtionCurriculumCfg:
    """Curriculum with terrain progression only (no two-phase switching)."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)


@configclass
class SpotFeetManipualtionEnvCfg(SpotFeetStandingEnvCfg):
    """Balanced Spot manipulation environment with random front-of-body gripper targets."""

    commands: SpotFeetManipualtionCommandsCfg = SpotFeetManipualtionCommandsCfg()
    observations: SpotFeetManipualtionObservationsCfg = SpotFeetManipualtionObservationsCfg()
    rewards: SpotFeetManipualtionRewardsCfg = SpotFeetManipualtionRewardsCfg()
    curriculum: SpotFeetManipualtionCurriculumCfg = SpotFeetManipualtionCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Keep this a standing + manipulation task (no base locomotion command).
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.debug_vis = False
        self.commands.ee_pose.debug_vis = True
        self.scene.terrain.debug_vis = False
        self.scene.terrain.visual_material = None

        # Allow arm motion toward targets (do not anchor arm to stow pose).
        self.rewards.joint_deviation_arm = None


class SpotFeetManipualtionEnvCfg_Play(SpotFeetManipualtionEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 6.0

        # Spawn robots randomly in the grid instead of fixed terrain levels.
        self.scene.terrain.max_init_terrain_level = None

        # Reduce terrain count for interactive play.
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False


# Backward-compatible aliases using the correct "Manipulation" spelling.
SpotFeetManipulationEnvCfg = SpotFeetManipualtionEnvCfg
SpotFeetManipulationEnvCfg_Play = SpotFeetManipualtionEnvCfg_Play



