# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import ViewerCfg

from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

from isaaclab_tasks.manager_based.sundtlejon.sundtlejon_env_cfg import SundtlejonEnvCfg

##
# Pre-defined configs
##
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

body_name = "base" # base for sim's 
body_name = "body" # body for white


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25), #1.0 to keep it flat 100% of the time
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.75, noise_range=(0.02, 0.07), noise_step=0.03, border_width=0.25
        ),
    },
)

PHASE_2_START_STEPS = 2500 * 24


def _set_value_after_steps(env, env_ids, data, value, num_steps):
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE


##
# MDP settings
##


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "fl_hx", "fr_hx", "hl_hx", "hr_hx", "fl_hy", "fr_hy", "hl_hy", "hr_hy", "fl_kn", "fr_kn", "hl_kn", "hr_kn" #oribt order
        ],
        scale=0.2,
        use_default_offset=True,
    )

    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm0.*"], #THIS USUALLY SAYS ARM0
        scale=0.2,
        use_default_offset=True,
    )




@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3, 9.0),
        rel_standing_envs=0.15, # og 0.1 how often should the sample be stay stil
        rel_heading_envs=0.9, # how often should he head in the direction? I think
        heading_command=False, # Change this to make him head in the direction of the command? 
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            #lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-0.5, 0.5)
            lin_vel_x=(-2.5, 2.5), lin_vel_y=(-2.5, 2.5), ang_vel_z=(-1.0, 1.0)
            #lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        #here is the issue
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.25, 1.25),
            "dynamic_friction_range": (0.25, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=body_name),
            "mass_distribution_params": (-2.5, 4.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=body_name),
            "force_range": (-15.0, 15.0),
            "torque_range": (-4.0, 4.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            # "velocity_range": {
            #     "x": (-1.5, 1.5),
            #     "y": (-1.0, 1.0),
            #     "z": (-0.5, 0.5),
            #     "roll": (-0.7, 0.7),
            #     "pitch": (-0.7, 0.7),
            #     "yaw": (-1.0, 1.0),
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (-1.5, 1.5),
                "y": (-0.75, 0.75),
                "z": (-0.25, 0.25),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.7, 0.7),
            },
        },

    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.2, 0.2),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
        },
    )


@configclass
class SpotRewardsCfg:
    # -- task
    air_time = RewTerm(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    base_angular_velocity = RewTerm(
        func=spot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewTerm(
        func=spot_mdp.base_linear_velocity_reward,
        weight=8.0,
        params={"std": 0.5, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
     
    foot_clearance = RewTerm(
        func=spot_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=20.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewTerm(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewTerm(
        func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    stand_height_error = RewTerm(
            func=mdp.base_height_error_penalty,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.65},
        )

    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    joint_acc = RewTerm(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewTerm(
        func=spot_mdp.joint_position_penalty,
        weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
        },
    )
    joint_torques = RewTerm(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewTerm(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )

    #------ ADDED ARM SHIT ------ 
    # arm position in stow state
    joint_deviation_arm = RewTerm(
        func=mdp.joint_deviation_l1,
        weight= -7.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "arm0_sh1",
                    "arm0_el0",
                    "arm0_el1",
                    "arm0_sh0",
                    "arm0_wr0",
                    "arm0_wr1",
                    "arm0_f1x",
                ],
            )
        },
    )

    joint_vel_arm = RewTerm(
        func=mdp.joint_velocity_penalty,
        weight=-5.0e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_h[xy]",
                    "arm0_sh1",
                    "arm0_el0",
                    "arm0_el1",
                    "arm0_sh0",
                    "arm0_wr0",
                    "arm0_wr1",
                    "arm0_f1x",
                ],
            ),
        },
    )

    joint_acc = RewTerm(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_h[xy]",
                    "arm0_sh1",
                    "arm0_el0",
                    "arm0_el1",
                    "arm0_sh0",
                    "arm0_wr0",
                    "arm0_wr1",
                    "arm0_f1x",
                ],
            )
        },
    )


@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[body_name, ".*uleg", "arm0.*"]), "threshold": 5.0},
    )
    
    non_foot_lleg_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lleg"), "threshold": 5.0},
    )

    terrain_out_of_bounds = DoneTerm(
        func=vel_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class SpotCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=vel_mdp.terrain_levels_vel)

    # Two-phase curriculum (phase switch at step 2500 * 24):
    # phase 1 uses the default values above, then these terms set phase 2 values.
    phase2_cmd_resample = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.resampling_time_range",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": (4.0, 10.0), "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_cmd_lin_vel_x = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.lin_vel_x",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": (-2.5, 3.0), "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_cmd_ang_vel_z = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.base_velocity.ranges.ang_vel_z",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": (-1.5, 1.5), "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_base_ang_std = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.base_angular_velocity.params.std",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": 2.0 / 1.5, "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_base_lin_std = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.base_linear_velocity.params.std",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": 1.0, "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_base_lin_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.base_linear_velocity.weight",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": 10.0, "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_gait_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.gait.weight",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": 12.0, "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_base_orientation_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.base_orientation.weight",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": -3.5, "num_steps": PHASE_2_START_STEPS},
        },
    )
    phase2_stand_height_weight = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.stand_height_error.weight",
            "modify_fn": _set_value_after_steps,
            "modify_params": {"value": -3.0, "num_steps": PHASE_2_START_STEPS},
        },
    )
    
    phase2_disable_non_foot_lleg_contact = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
        "address": "terminations.non_foot_lleg_contact.params.threshold",
        "modify_fn": _set_value_after_steps,
        "modify_params": {"value": float("inf"), "num_steps": PHASE_2_START_STEPS},
    },
)




##
# Environment configuration
##


@configclass
class SpotLocomotionEnvCfg(SundtlejonEnvCfg):
    # Basic settings
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()
    observations: SpotObservationsCfg = SpotObservationsCfg()
    events: SpotEventCfg = SpotEventCfg()
    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    curriculum: SpotCurriculumCfg = SpotCurriculumCfg()
    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")
    
    # Post initialization
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        # terrain physics settings
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt


        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level= 10, #COBBLESTONE_ROAD_CFG.num_rows - 1, #change this to small value that can then be increased by the terrain_levels_vel
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
  

class SpotLocomotionEnvCfg_Play(SpotLocomotionEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0
        # spawn the robot randomly in the grid (instead of their terrain levels) 
        self.scene.terrain.max_init_terrain_level = None
        

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
