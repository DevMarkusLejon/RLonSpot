import torch

src = "logs/rsl_rl/spot_SL_standing_flat/2026-03-04_17-08-08/model_799.pt"
dst = "logs/rsl_rl/spot_SL_standing_flat/2026-03-04_17-08-08/model_799_obs69.pt"

ckpt = torch.load(src, map_location="cpu", weights_only=False)

# New obs order added 3 command dims after first 9 dims:
# [base_lin_vel(3), base_ang_vel(3), projected_gravity(3), velocity_commands(3), ...]
for k in ["actor.0.weight", "critic.0.weight"]:
    w = ckpt["model_state_dict"][k]  # [32, 66]
    z = torch.zeros((w.shape[0], 3), dtype=w.dtype)
    ckpt["model_state_dict"][k] = torch.cat([w[:, :9], z, w[:, 9:]], dim=1)  # [32, 69]

# Important: old Adam state has wrong tensor shapes after input change.
ckpt["optimizer_state_dict"]["state"] = {}

# Optional: reset iteration counter if you want full new max_iterations from zero.
# ckpt["iter"] = 0

torch.save(ckpt, dst)
print("saved:", dst)
