# Reward functions in the program
# arm position in stow state
    joint_deviation_hip = RewardTermCfg(
        func=spot_mdp.joint_deviation_l1,
        weight=-5,
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

joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
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

joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
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


# Reward functions in the rewards script
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

