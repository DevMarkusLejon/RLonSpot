# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sketch environment for Spot-with-arm opening a drawer.

This file intentionally provides a *starting point* and not a fully tuned task.
It reuses the cabinet manipulation MDP and swaps in Spot-with-arm so you can
iterate quickly on whole-body control and frame/joint calibration.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp as cabinet_mdp
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
    CabinetEnvCfg,
    CabinetSceneCfg,
)

from isaaclab_assets.robots.spot import SPOT_ARM_BDAIFEET_CFG  # isort: skip

from . import mdp as spot_mdp


GROUND_FILTER_PATHS = ["/World/GroundPlane", "/World/ground/terrain/mesh"]


@configclass
class SpotOpenDrawerSceneCfg(CabinetSceneCfg):
    """Cabinet scene variant with Spot contact sensors for robust terminations."""

    # Reverted to broad contact sensor because this matched your previous drawer-contact behavior.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # Per lower-leg sensors needed by any_lleg_non_foot_contact.
    contact_fl_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fl_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=GROUND_FILTER_PATHS,
    )
    contact_fr_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=GROUND_FILTER_PATHS,
    )
    contact_hl_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hl_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=GROUND_FILTER_PATHS,
    )
    contact_hr_lleg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/hr_lleg",
        history_length=3,
        track_air_time=True,
        track_pose=True,
        track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=GROUND_FILTER_PATHS,
    )

    # Detect self-contact between arm links and robot body.
    contact_arm_to_body = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm0.*",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/body"],
    )


@configclass
class SpotOpenDrawerTerminationsCfg:
    """Spot-specific termination terms for drawer opening."""

    time_out = DoneTerm(func=cabinet_mdp.time_out, time_out=True)

    drawer_opened_good = DoneTerm(
        func=spot_mdp.drawer_opened_above_threshold,
        params={
            "threshold": 0.30,
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
        },
    )

    #Restored previous contact behavior (drawer/cabinet contacts can terminate).
    # body_contact = DoneTerm(
    #     func=cabinet_mdp.illegal_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*uleg", "arm0.*"]),
    #         "threshold": 5.0,
    #     },
    # )

    # Added explicit fall checks so falling over terminates reliably.
    bad_orientation = DoneTerm(
        func=cabinet_mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    low_base_height = DoneTerm(
        func=cabinet_mdp.root_height_below_minimum,
        params={"minimum_height": 0.24, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Extra locomotion-style safeguard.
    non_foot_lleg_contact = DoneTerm(
        func=spot_mdp.any_lleg_non_foot_contact,
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

    # Terminate if arm makes self-contact with robot body.
    arm_self_collision = DoneTerm(
        func=spot_mdp.filtered_contact_force_exceeds,
        params={
            "threshold": 3.0,
            "sensor_cfg": SceneEntityCfg("contact_arm_to_body", body_names=["arm0.*"]),
        },
    )


@configclass
class SpotOpenDrawerEnvCfg(CabinetEnvCfg):
    """Spot + cabinet drawer task scaffold based on CabinetEnvCfg."""

    scene: SpotOpenDrawerSceneCfg = SpotOpenDrawerSceneCfg(num_envs=512, env_spacing=3.0)
    terminations: SpotOpenDrawerTerminationsCfg = SpotOpenDrawerTerminationsCfg()

    def __post_init__(self):
        # Post init of parent (cabinet scene + generic drawer rewards/obs/events).
        super().__post_init__()

        # Use Spot with arm (BDAI-feet URDF variant).
        self.scene.robot = SPOT_ARM_BDAIFEET_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Needed so arm-vs-body contacts are physically realized and can be penalized.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True

        # Start a bit farther back than fixed-base Franka.
        self.scene.robot.init_state.pos = (-0.67, 0.0, 0.65)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Joint-space arm control (first pass scaffold).
        self.actions.arm_action = cabinet_mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"],
            scale=0.25*0.1,
            use_default_offset=True,
        )
        self.actions.gripper_action = cabinet_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm0_f1x"],
            open_command_expr={"arm0_f1x": 0.0},
            close_command_expr={"arm0_f1x": -1.2},
        )

        # Required frame ordering:
        # 0 -> ee_tcp, 1 -> tool_leftfinger, 2 -> tool_rightfinger
        # NOTE: Spot URDF here is single-finger. We use pseudo left/right offsets on
        # the same fingertip link so cabinet reward code can run unchanged.
        # TODO(user): Replace with true gripper fingertip links if available.
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/body",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/SpotEndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/arm0_link_wr1",
                    name="ee_tcp",
                    offset=OffsetCfg(pos=(0.12, 0.0, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/arm0_link_fngr",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.02, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/arm0_link_fngr",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, -0.02, 0.0)),
                ),
            ],
        )

        # Reward term parameters that depend on gripper model.
        self.rewards.approach_gripper_handle.params["offset"] = 0.03
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.0
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["arm0_f1x"]

        # Penalize arm joint velocities to reduce erratic arm motion.
        self.rewards.arm_joint_vel = RewTerm(
            func=cabinet_mdp.joint_vel_l2,
            weight=-2.5e-5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["arm0_sh0", "arm0_sh1", "arm0_el0", "arm0_el1", "arm0_wr0", "arm0_wr1"],
                )
            },
        )

        # Encourage base retreat as the drawer is pulled open.
        self.rewards.base_backward_on_pull = RewTerm(
            func=spot_mdp.backward_base_when_drawer_pulled,
            weight=1.5,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
                "pull_threshold": 0.02,
                "full_open_pos": 0.30,
                "target_backward_speed": 0.15,
            },
        )

        # Penalize non-timeout terminations (timeouts are excluded by is_terminated_term).
        self.rewards.non_timeout_termination = RewTerm(
            func=cabinet_mdp.is_terminated_term,
            weight=-50.0,
            params={
                "term_keys": [
                    #"body_contact",
                    "bad_orientation",
                    "low_base_height",
                    "non_foot_lleg_contact",
                    #"arm_self_collision",
                ]
            },
        )

        # Keep contact sensors synchronized with physics step for reliable terminations.
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.contact_fl_lleg.update_period = self.sim.dt
        self.scene.contact_fr_lleg.update_period = self.sim.dt
        self.scene.contact_hl_lleg.update_period = self.sim.dt
        self.scene.contact_hr_lleg.update_period = self.sim.dt
        self.scene.contact_arm_to_body.update_period = self.sim.dt

        # Sketch defaults: longer episode for mobile manipulation.
        self.episode_length_s = 12.0
        self.viewer.eye = (-3.0, 2.0, 2.2)
        self.viewer.lookat = (0.8, 0.0, 0.65)


@configclass
class SpotOpenDrawerEnvCfg_PLAY(SpotOpenDrawerEnvCfg):
    """Play variant of SpotOpenDrawerEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


@configclass
class SpotOpenDrawerFixedBaseEnvCfg(SpotOpenDrawerEnvCfg):
    """Fixed-base variant for arm-focused drawer opening."""

    def __post_init__(self):
        super().__post_init__()
        # Anchor the root so the policy can focus on manipulation instead of balance.
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # Keep URDF conversion settings aligned with fixed-base intent.
        self.scene.robot.spawn.fix_base = True
        # In fixed-base mode, swap base retreat for end-effector retreat.
        self.scene.robot.init_state.joint_pos ={
            # otbit order
            "fl_hx": 0.1, # 
            "fr_hx": -0.1, 
            "hl_hx": 0.1, 
            "hr_hx": -0.1, 
            "fl_hy": 0.9, 
            "fr_hy": 0.9, 
            "hl_hy": 1.1, 
            "hr_hy": 1.1, 
            "fl_kn": -1.5, 
            "fr_kn": -1.5, 
            "hl_kn": -1.5, 
            "hr_kn":-1.5,
            "arm0_sh0": 0, #-150°/180°  /pi
            "arm0_sh1": -0.9, # -180°/30° 
            "arm0_el0": 1.8, # 0°/180° 
            "arm0_el1": 0,#0, # -160°/160° 
            "arm0_wr0": -0.9, # -105°/105°
            "arm0_wr1": 0,#0, # -165°/165°
            "arm0_f1x": 0,#-1.54, #-90°/0°
        }

        self.rewards.base_backward_on_pull = RewTerm(
            func=spot_mdp.backward_ee_when_drawer_pulled,
            weight=1.5,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "cabinet_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
                "ee_frame_name": "ee_frame",
                "pull_threshold": 0.02,
                "full_open_pos": 0.30,
                "retreat_start_x": 0.22,
                "retreat_span": 0.18,
            },
        )

        # Slightly shorter episodes can speed up early curriculum/debug loops.
        self.episode_length_s = 10.0


@configclass
class SpotOpenDrawerFixedBaseEnvCfg_PLAY(SpotOpenDrawerFixedBaseEnvCfg):
    """Play variant of SpotOpenDrawerFixedBaseEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
