import ast
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_JOINT_POSITION = np.array([
    0.1, -0.1, 0.1, -0.1, 
    0.9, 0.9, 1.1, 1.1,
    -1.5, -1.5, -1.5, -1.5,
    0, -3.14, 3.14, 1.56, 0, -1.56, 0 
])

REQUIRED_KEYS = [
    "joint_positions:",
    "shifted_action:",
    "base_linear_velocity:",
    "commanded_vel:"
]
EXPECTED_VECTOR_LENGHT = {
    "joint_positions:": 19,
    "shifted_action:": 19,
    "base_linear_velocity:": 3,
    "commanded_vel:": 3
}

# This is in orbit order
JOINT_LABELS = ["fl_hx", "fr_hx","hl_hx", "hr_hx", 
          "fl_hy", "fr_hy", "hl_hy", "hr_hy",
          "fl_kn", "fr_kn", "hl_kn", "hr_kn",
          "a0_sh0", "a0_sh1", "a0_el0", "a0_el1", "a0_wr0", "a0_wr1", "a0_f1x",]
# label -> idx matching
JOINT_LABEL_TO_IDX = {label: i for i, label in enumerate(JOINT_LABELS)}

JOINT_GROUPS = {
    "FL": ["fl_hx", "fl_hy", "fl_kn"],
    "FR": ["fr_hx", "fr_hy", "fr_kn"],
    "HL": ["hl_hx", "hl_hy", "hl_kn"],
    "HR": ["hr_hx", "hr_hy", "hr_kn"],
    "Arm joints": ["a0_sh0", "a0_sh1", "a0_el0", "a0_el1", "a0_wr0", "a0_wr1", "a0_f1x"]
}

# JOINT_COLOR_MAP = {
#     "fl_hx": "tab:blue",   "fl_hy": "tab:cyan",   "fl_kn": "tab:purple",
#     "fr_hx": "tab:orange", "fr_hy": "tab:brown",  "fr_kn": "tab:pink",
#     "hl_hx": "tab:green",  "hl_hy": "tab:olive",  "hl_kn": "tab:gray",
#     "hr_hx": "tab:red",    "hr_hy": "tab:blue",   "hr_kn": "tab:orange",
#     "a0_sh0": "tab:blue", "a0_sh1": "tab:orange", "a0_el0": "tab:green", "a0_el1": "tab:red", "a0_wr0": "tab:purple", "a0_wr1": "tab:brown", "a0_f1x": "tab:pink",
# }
JOINT_COLOR_MAP = {
    "fl_hx": "tab:blue",    "fl_hy": "tab:red",   "fl_kn": "tab:green",
    "fr_hx": "tab:blue",    "fr_hy": "tab:red",   "fr_kn": "tab:green",
    "hl_hx": "tab:blue",    "hl_hy": "tab:red",   "hl_kn": "tab:green",
    "hr_hx": "tab:blue",    "hr_hy": "tab:red",   "hr_kn": "tab:green",
    "a0_sh0": "tab:blue", "a0_sh1": "tab:orange", "a0_el0": "tab:green", "a0_el1": "tab:red", "a0_wr0": "tab:purple", "a0_wr1": "tab:brown", "a0_f1x": "tab:pink",
}

VEL_LABELS = ["vx", "vy", "vz"]

VEL_COLOR_MAP = {
    "vx": "tab:green", "vy": "tab:blue", "vz": "tab:red"
}

# ----- DATA LOADING -----
def load_data(filename):
    """Read text file with data and store in dict of numpy arrays."""
    data = {}
    with open(filename, "r") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            
            # Find where the vector starts
            bracket_index = line.find("[")
            if bracket_index == -1:
                print(f"WARNING: line {line_num} has no vector and will be skipped")
                continue

            # Split into label and vector string, skip labels that are not required
            key = line[:bracket_index].strip()
            vector_str = line[bracket_index:].strip()
            if key not in REQUIRED_KEYS:
                #print(f"Key: '{key}' not required, skipping.")
                continue

            # Convert string to actual list
            try:
                vector = ast.literal_eval(vector_str)
            except Exception as e:
                print(f"WARNING: could not parse vector on line {line_num}: {e}")
                continue

            # Create list for this key if not already present
            if key not in data:
                data[key] = []
            
            data[key].append(vector)

    # Convert all lists to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data

def validate_data(data):
    """Ensure all required signals have the same amount of samples (data points) and the correct dimension."""
    sample_count = {key: data[key].shape[0] for key in REQUIRED_KEYS}

    if len(set(sample_count.values())) != 1:
        error_msg = "Mismatch in number of samples:\n"
        for key, count in sample_count.items():
            error_msg += f"  {key} {count}\n"
        raise ValueError(error_msg)

    

    for key, expected_len in EXPECTED_VECTOR_LENGHT.items():
        if data[key].shape[1] != expected_len:
            raise ValueError(
                f"Wrong vector lenght for '{key}'\n"
                f"  Expected: {expected_len}\n"
                f"  Got: {data[key].shape[1]}"
            )

def print_summary(data):
    """Print loaded data."""
    print("Required keys:")
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
        else:
            print(f"  {key}")

    print("\nLoaded keys:")
    for key, arr in data.items():
        print(f"  {key} {arr.shape}")


# ----- PLOTTING -----
def plot_joint_subplot(ax, x, joint_names, group_name, joint_pos, shifted_action):
    """Plot one joint group on ax."""
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]

        ax.plot(
            x,
            joint_pos[:, idx] + DEFAULT_JOINT_POSITION[idx],
            color=color,
            linestyle="-",
            linewidth=2,
            label=joint
        )
        ax.plot(
            x,
            shifted_action[:, idx],
            color=color,
            linestyle="--",
            linewidth=2,
        )
        ax.hlines(
            y=DEFAULT_JOINT_POSITION[idx],
            xmin=0,
            xmax=max(x),
            color=color,
            linestyle=":",
            linewidth=2,
        )

    ax.set_ylabel("Joint angle [rad]")
    ax.set_title(f"{group_name} joints. (Solid - Position, Dashed - Action, Dotted - Default)")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=1, loc="upper left")


def plot_vel_subplot(ax, x, base_vel, cmd_vel):
    """Plot vel comparison on ax."""
    for i, vel_name in enumerate(VEL_LABELS):
        vel_color = VEL_COLOR_MAP[vel_name] 

        ax.plot(
            x,
            base_vel[:, i],
            color=vel_color,
            linestyle="-",
            linewidth=2,
            label=vel_name
        )

        ax.plot(
            x,
            cmd_vel[:, i],
            color=vel_color,
            linestyle="--",
            linewidth=2,
        )

    ax.set_xlabel("Timestep [-]")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Base linear velocity vs commanded. (Solid - Base, Dashed - Commanded)")
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper left")


def plot_all_groups(data):
    """Create all group plots."""
    joint_pos = data["joint_positions:"]
    shifted_action = data["shifted_action:"]
    base_vel = data["base_linear_velocity:"]
    cmd_vel = data["commanded_vel:"]

    x = np.arange(joint_pos.shape[0])

    for group_name, joint_names in JOINT_GROUPS.items():
        fig, (joint_ax, vel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3,1]}) 
        plot_joint_subplot(joint_ax, x, joint_names, group_name, joint_pos, shifted_action)
        plot_vel_subplot(vel_ax, x, base_vel, cmd_vel)
        plt.tight_layout()
    plt.show()

# ----- MAIN -----
def main():
    filename = "/home/sundt/thesis/colcon_ws/src/my_spot_thesis/spot_deploy_data/spot_joint_values.txt"
    data = load_data(filename)
    validate_data(data)
    print_summary(data)
    plot_all_groups(data)

if __name__ == "__main__":
    main()