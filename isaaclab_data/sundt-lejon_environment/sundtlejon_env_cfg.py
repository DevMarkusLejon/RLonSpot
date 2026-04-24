# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ViewerCfg

from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp

##
# Pre-defined configs
##

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.spot import SPOT_CFG, SPOT_ARM_CFG, SPOT_ARM_BDAIFEET_CFG # isort: skip


##
# Scene definition
##

body_name = "base" # base for sim's 
body_name = "body" # body for white  


@configclass
class SundtlejonSceneCfg(InteractiveSceneCfg):
    """Configuration for a Sundt-Lejon scene."""

    # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )

    #ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=1,
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
        debug_vis=False,
    )
    #spot
    robot: ArticulationCfg = SPOT_ARM_BDAIFEET_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # set up contact sensors on all joints
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=0, track_air_time=True)
    # one sensor per lower leg (required for filtered contacts to be reliable)
    contact_fl_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fl_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        # old: filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        # For generator terrains, collisions are on /World/ground/terrain/mesh.
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )
    contact_fr_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        # old: filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        # For generator terrains, collisions are on /World/ground/terrain/mesh.
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )
    contact_hl_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hl_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        # old: filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        # For generator terrains, collisions are on /World/ground/terrain/mesh.
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )
    contact_hr_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hr_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        # old: filter_prim_paths_expr=["/World/ground/terrain/GroundPlane/CollisionPlane"],
        # For generator terrains, collisions are on /World/ground/terrain/mesh.
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

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
        joint_names=["arm0.*"],
        scale=0.2,
        use_default_offset=True,
    )


@configclass
class CommandCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.7,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # --- base state ---
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05))

        # --- joint state ---
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05))

        # --- last action (helps learning stability) ---
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    #reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    # reset_scene = EventTerm(func=mdp.reset_joints_by_offset, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: 

    # (4) Shaping tasks: 

    # (5) Shaping tasks: 


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Collosion
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=[body_name, ".*uleg", "arm0.*"] #"body" here 
            ),
            "threshold": 1.0
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    #terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class SundtlejonEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: SundtlejonSceneCfg = SundtlejonSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    commands: CommandCfg = CommandCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt

        self.scene.contact_fl_lleg.update_period = self.sim.dt
        self.scene.contact_fr_lleg.update_period = self.sim.dt
        self.scene.contact_hl_lleg.update_period = self.sim.dt
        self.scene.contact_hr_lleg.update_period = self.sim.dt


@configclass
class SundtlejonEnvCfg_Play(SundtlejonEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
