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
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .. import mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

from isaaclab_tasks.manager_based.sundtlejon.sundtlejon_env_cfg import SundtlejonEnvCfg

##
# Pre-defined configs
##
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.75, noise_range=(0.02, 0.07), noise_step=0.03, border_width=0.25
        ),
    },
)

body_name = "base" # base for sim's
body_name = "body" # body for white
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
            "fl_hx", "fl_hy", "fl_kn",
            "fr_hx", "fr_hy", "fr_kn",
            "hl_hx", "hl_hy", "hl_kn",
            "hr_hx", "hr_hy", "hr_kn",
        ],
        scale=0.3,
        use_default_offset=True,
    )
    arm_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm0.*"],
        scale=0.1,
        use_default_offset=True,
    )



@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5, 10.0),
        rel_standing_envs=1.0, # og 0.1
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
       ),
    )


@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        #ee_to_handle = ObsTerm(func=mdp.relative)


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
            "static_friction_range": (0.2, 1.4),
            "dynamic_friction_range": (0.15, 1.2),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 128,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=body_name),
            "mass_distribution_params": (-4, 4),
            "operation": "add",
        },
    )

    

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=body_name),
            "force_range": (-5,5),#(-15.0, 15.0),
            "torque_range": (-4.0, 4.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            #"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),"z":(0.07,0.1), "yaw": (-3.14, 3.14)},
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5),"z":(0.07,0.1), "yaw": (-3.14/3, 3.14/3)},
            "velocity_range": {
                # "x": (-1.5, 1.5),
                # "y": (-1.0, 1.0),
                # "z": (-0.5, 0.5),
                # "roll": (-0.7, 0.7),
                # "pitch": (-0.7, 0.7),
                # "yaw": (-1.0, 1.0),
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.05, 0.05),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.2, 0.2),
            "asset_cfg": SceneEntityCfg("robot",joint_names=[".*_kn",".*_hx",".*_hy"]),
        },
    )
    

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
        },
    )


@configclass
class SpotRewardsCfg:
    # -- task
    keep_alive = RewTerm(
        func= mdp.is_alive,
        weight= 3.0,
    )

    keep_alive_ramp = RewTerm(
        func = mdp.is_alive_ramp,
        weight=1.0,
        params={"base_value": 0.0, "growth_per_second": 0.01, "max_value": 0.2},
    )

    stand_feet_contact = RewTerm(
        func=mdp.feet_contact_count_reward,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lleg")},
    )

    #Codex did this
    foot_zone_contact = RewTerm(
        func=mdp.feet_in_foot_zone_reward,
        weight=1.5,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("contact_fl_lleg"),
                SceneEntityCfg("contact_fr_lleg"),
                SceneEntityCfg("contact_hl_lleg"),
                SceneEntityCfg("contact_hr_lleg"),
            ],
            "force_threshold": 5.0,
            "foot_z_max": -0.28,
            "foot_xy_radius": 0.06,
        },
    )

    joint_pos_default_tracking = RewTerm(
        func=mdp.joint_pos_default_tracking_reward,
        weight=6.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "std": 0.15,
        },
    )


    #mdp.base_height_l2

    # -- penalties

    stand_xy_drift = RewTerm(
        func=mdp.base_xy_position_drift_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stand_yaw_drift = RewTerm(
        func=mdp.yaw_drift_penalty,
        weight=-0.75,
        params={"asset_cfg": SceneEntityCfg("robot"), "desired_yaw": 0.0},
    )
    stand_height_error = RewTerm(
        func=mdp.base_height_error_penalty,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.6}, #previous height target was 0.55 
    )

    lleg_non_foot_contact = RewTerm(
        func=mdp.lleg_non_foot_contact_penalty,
        weight=-4.0,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("contact_fl_lleg"),
                SceneEntityCfg("contact_fr_lleg"),
                SceneEntityCfg("contact_hl_lleg"),
                SceneEntityCfg("contact_hr_lleg"),
            ],
            "force_threshold": 5.0,
            "foot_z_max": -0.28,
            "foot_xy_radius": 0.06,
        },
    )

    terminating = RewTerm(func=mdp.is_terminated,weight=-250.0)

    action_smoothness = RewTerm(func=spot_mdp.action_smoothness_penalty, weight=-0.5)

    base_motion = RewTerm(
        func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewTerm(
        func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_lleg"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lleg"),
            "threshold": 1.0,
        },
    )
    joint_pos = RewTerm(
        func=spot_mdp.joint_position_penalty,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.10,
        },
    )
    joint_torques = RewTerm(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
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
    joint_vel = RewTerm(
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
        weight=-2.0e-4,
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
    #Codex did this
    non_foot_lleg_contact = DoneTerm(
        func=mdp.any_lleg_non_foot_contact,
        params={
            "sensor_cfgs": [
                SceneEntityCfg("contact_fl_lleg"),
                SceneEntityCfg("contact_fr_lleg"),
                SceneEntityCfg("contact_hl_lleg"),
                SceneEntityCfg("contact_hr_lleg"),
            ],
            "force_threshold": 2.0,
            "foot_z_max": -0.28,
            "foot_xy_radius": 0.06,
        },
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


##
# Environment configuration
##


@configclass
class SpotStandingEnvCfg(SundtlejonEnvCfg):
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
        self.episode_length_s = 5.0
        #self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.05, 0.10), "yaw": (0.0, 0.0)}
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

        self.scene.contact_fl_lleg.update_period = self.sim.dt
        self.scene.contact_fr_lleg.update_period = self.sim.dt
        self.scene.contact_hl_lleg.update_period = self.sim.dt
        self.scene.contact_hr_lleg.update_period = self.sim.dt

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level= 10, #COBBLESTONE_ROAD_CFG.num_rows - 1
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

@configclass
class SpotStandingEnvCfg_Play(SpotStandingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event


@configclass
class SpotStandingEnvCfg_flat_conditon_test_Play(SpotStandingEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        for actuator_cfg in self.scene.robot.actuators.values():
            if hasattr(actuator_cfg, "min_delay"):
                actuator_cfg.min_delay = 0
            if hasattr(actuator_cfg, "max_delay"):
                actuator_cfg.max_delay = 0

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 15

        # switch to a flat plane so validation is done without terrain disturbances
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # keep the standing task at a true zero-velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0

        # disable observation corruption and randomization for a clean baseline
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # tighten reset disturbances so each episode starts near nominal
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.08, 0.08), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        #self.events.reset_robot_joints

        #self.events.reset_arm_wr1
        #self.events.reset_arm_el1

