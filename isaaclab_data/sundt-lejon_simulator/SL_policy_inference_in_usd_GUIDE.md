# Spot Policy Inference Guide

This guide explains how to use `scripts/tutorials/03_envs/SL_policy_inference_in_usd.py`
for all currently supported configurations.

The script loads a TorchScript/JIT policy checkpoint, creates a single Spot environment,
and runs policy inference either interactively with the keyboard or headless for smoke
tests.

## Script Path

```bash
scripts/tutorials/03_envs/SL_policy_inference_in_usd.py
```

## What The Script Supports

The script currently supports two environment contracts:

- `feet`
  Uses `SpotLocomotionEnvCfg_Play` from
  `isaaclab_tasks.manager_based.sundtlejon.feet_spot_locomotion_env_cfg`
- `leg_only`
  Uses `SpotLegOnlyLocomotionEnvCfg_Play` from
  `isaaclab_tasks.manager_based.sundtlejon.spot_leg_only_locomotion_env_cfg`

It also supports three arm handling modes:

- `auto`
  Holds the default arm pose in `leg_only` mode and leaves `feet` mode unchanged
- `hold_default`
  Always holds the arm at the robot default pose
- `free`
  Does not explicitly control the arm

## CLI Arguments

### Script-specific arguments

- `--checkpoint PATH`
  Required. Path to a TorchScript/JIT policy file.
- `--env-config {feet,leg_only}`
  Selects which environment contract the checkpoint expects.
  Default: `feet`
- `--num-steps N`
  Optional. Run exactly `N` environment steps and then exit.
  Default: `0`, which means run until the app closes.
- `--arm-mode {auto,hold_default,free}`
  Controls how the arm is handled while the policy runs.
  Default: `auto`

### Common AppLauncher arguments

These come from Isaac Lab's `AppLauncher` and are available to this script too:

- `--headless`
  Run without GUI.
- `--device {cpu,cuda,cuda:N}`
  Choose the simulation device.
  Default in Isaac Lab is typically `cuda:0`.
- `--livestream {0,1,2}`
  Enable WebRTC streaming.
- `--enable_cameras`
  Needed if your environment uses camera sensors.
- `--experience PATH`
  Override the Isaac Sim experience `.kit` file.
- `--kit_args "..."`  
  Pass raw Omniverse Kit arguments.

## Keyboard Controls

In non-headless mode, the script maps keyboard input to the Spot velocity command:

- `W` / `S`: forward / backward
- `Q` / `E`: strafe left / right
- `A` / `D`: yaw left / right
- Arrow keys also work for forward/back and yaw

## Core Runtime Behavior

The script makes a few choices regardless of mode:

- It always creates `1` environment
- It forces the terrain to `plane`
- It disables command resampling by setting a very large resampling window
- It disables observation corruption
- It disables several randomization events for cleaner playback

This makes it a policy playback script, not a training script.

## Configuration Matrix

### 1. `feet` + interactive GUI

Use this when:

- Your checkpoint was trained for the `feet` environment contract
- You want to drive the robot with the keyboard
- Your checkpoint expects the full feet/arm action-observation contract used by the
  current `feet_spot_locomotion_env_cfg`

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/feet_policy.pt \
  --env-config feet
```

Arm behavior:

- `--arm-mode auto` leaves the existing `feet` behavior alone
- `--arm-mode hold_default` forces the arm to stay at the default pose
- `--arm-mode free` leaves the arm unconstrained by this helper

### 2. `feet` + headless smoke test

Use this when:

- You want to verify that a `feet` checkpoint loads and steps
- You do not need the GUI
- You want a short sanity check in CI or from SSH

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/feet_policy.pt \
  --env-config feet \
  --headless \
  --num-steps 10
```

Notes:

- In headless mode the script holds the base command at zero
- The robot will not receive keyboard input in this mode

### 3. `leg_only` + interactive GUI

Use this when:

- Your checkpoint was trained for a 12-action leg-only contract
- The arm is present on the robot asset, but the policy does not output arm actions
- You want to drive the robot with the keyboard

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/leg_only_policy.pt \
  --env-config leg_only
```

Important behavior in this mode:

- The script narrows `joint_pos` and `joint_vel` observations to the 12 leg joints
- With `--arm-mode auto`, the script continuously reapplies the default arm pose so the
  arm does not spring outward

### 4. `leg_only` + headless smoke test

Use this when:

- You want a quick check that a leg-only checkpoint still loads and steps
- You are running remotely or testing without a viewer

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/leg_only_policy.pt \
  --env-config leg_only \
  --headless \
  --num-steps 10
```

This is the best mode for fast validation after making environment or policy-loader
changes.

## Arm Handling Modes

### `--arm-mode auto`

Recommended for most uses.

Behavior:

- In `leg_only`, holds the arm at the default robot pose every step
- In `feet`, does not add extra arm handling

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/policy.pt \
  --env-config leg_only \
  --arm-mode auto
```

### `--arm-mode hold_default`

Use this when:

- You want the arm parked even if the environment is `feet`
- You want deterministic arm posture during evaluation

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/policy.pt \
  --env-config feet \
  --arm-mode hold_default
```

### `--arm-mode free`

Use this when:

- You do not want the script to inject any helper arm target
- Your policy or environment already manages the arm correctly

Example:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/policy.pt \
  --env-config leg_only \
  --arm-mode free
```

Warning:

- In `leg_only`, this may let the arm drift or extend because the robot still has active
  arm actuators but the policy does not command them

## Recommended Commands

### NVIDIA leg-only checkpoint

This is the checkpoint currently located at:

```bash
/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/sundtlejon/nvidias_policy/spot_policy\ \(1\).pt
```

Recommended interactive command:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/sundtlejon/nvidias_policy/'spot_policy (1).pt' \
  --env-config leg_only \
  --arm-mode auto
```

Recommended headless test:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/sundtlejon/nvidias_policy/'spot_policy (1).pt' \
  --env-config leg_only \
  --arm-mode auto \
  --headless \
  --num-steps 10 \
  --device cpu
```

### Existing feet-style checkpoint

If you have a checkpoint that matches the current `feet` observation/action contract:

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/feet_checkpoint.pt \
  --env-config feet
```

## How To Choose `env-config`

Choose `--env-config leg_only` when:

- The policy outputs only the 12 leg joints
- The policy was not trained with an arm action head
- You are using the NVIDIA policy under `nvidias_policy/`

Choose `--env-config feet` when:

- The policy was trained with the current feet locomotion contract
- The checkpoint expects the full `feet` observation and action structure

If you choose the wrong one, you will usually get a TorchScript matrix multiply shape
error such as:

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

## Compatibility Notes

Not all old Spot checkpoints are interchangeable.

A checkpoint must match:

- the action dimension
- the observation dimension
- the ordering of observation terms
- the ordering of action joints

In practice, this means:

- `leg_only` checkpoints usually need `leg_only`
- some older `feet` checkpoints may not match the current `feet` runtime if they were
  trained with a different observation layout

One example we already observed is an older feet-style exported policy that expected an
`ee_pose_command` observation term instead of the current `velocity_commands` term.

## Typical Failure Modes

### 1. Arm springs outward in `leg_only`

Cause:

- The robot still has arm actuators, but the policy does not command the arm

Fix:

- Use `--arm-mode auto` or `--arm-mode hold_default`

### 2. TorchScript shape mismatch

Cause:

- The checkpoint does not match the selected environment contract

Fix:

- Try the correct `--env-config`
- Verify the checkpoint was trained with the same observation layout

### 3. No keyboard response

Cause:

- You are running with `--headless`

Fix:

- Run without `--headless` for interactive teleoperation

### 4. CPU is slow

Cause:

- Isaac Sim is running physics on CPU

Fix:

- Prefer `--device cuda:0` if available

## Quick Recipes

### Interactive NVIDIA leg-only run

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/sundtlejon/nvidias_policy/'spot_policy (1).pt' \
  --env-config leg_only \
  --arm-mode auto
```

### Headless NVIDIA leg-only validation

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/sundtlejon/nvidias_policy/'spot_policy (1).pt' \
  --env-config leg_only \
  --arm-mode auto \
  --headless \
  --num-steps 20
```

### Interactive feet run

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/feet_checkpoint.pt \
  --env-config feet
```

### Force the arm to stay parked in any mode

```bash
./isaaclab.sh -p scripts/tutorials/03_envs/SL_policy_inference_in_usd.py \
  --checkpoint /absolute/path/to/policy.pt \
  --env-config feet \
  --arm-mode hold_default
```

## Future Extensions

If we add more policy families later, the clean pattern is:

- add a new `--env-config` option
- adapt the observation contract for that policy family
- document the expected checkpoint type in this guide

