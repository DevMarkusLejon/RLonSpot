import ast
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
#fl_hx, fr_hx, hl_hx, hr_hx, fl_hy, fr_hy, hl_hy, hr_hy, fl_kn, fr_kn, hl_kn, hr_kn
DEFAULT_JOINT_POSITION = np.array([
    0.1, -0.1, 0.1, -0.1, 
    0.9, 0.9, 1.1, 1.1,
    -1.5, -1.5, -1.5, -1.5,
    0, -3.14, 3.14, 1.56, 0, -1.56, 0 
])
UPPER_JOINT_BOUND = np.array([
    0.79, 0.79, 0.79, 0.79,
    2.3, 2.3, 2.3, 2.3,
    -0.25, -0.25, -0.25, -0.25,
    3.14, 0.52, 3.14, 2.79, 1.83, 2.88, 0
])
LOWER_JOINT_BOUND = np.array([
    -0.79, -0.79, -0.79, -0.79,
    -0.9, -0.9, -0.9, -0.9,
    -2.79, -2.79, -2.79, -2.79,
    -2.62, -3.14, 0, -2.79, -1.83, -2.88, -1.57
])

REQUIRED_KEYS = [
    "joint_positions:",
    "joint_velocity:",
    "shifted_action:",
    "base_linear_velocity:",
    "commanded_vel:",
    "joint_torques:",
]

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
    "Arm": ["a0_sh0", "a0_sh1", "a0_el0", "a0_el1", "a0_wr0", "a0_wr1", "a0_f1x"]
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

VEL_LABELS = ["X", "Y", "YAW"]

VEL_COLOR_MAP = {
    "X": "tab:green", "Y": "tab:blue", "YAW": "tab:red"
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
def plot_jointpos_subplot(ax, x, joint_names, group_name, joint_pos, shifted_action, show_ylabel=True, show_legend=True):
    """Plot joint pos groupwise on ax."""
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
        # ax.hlines(
        #     y=DEFAULT_JOINT_POSITION[idx],
        #     xmin=0,
        #     xmax=max(x),
        #     color=color,
        #     linestyle=":",
        #     linewidth=2,
        # )
        ax.hlines(
            y=[UPPER_JOINT_BOUND[idx], LOWER_JOINT_BOUND[idx]],
            xmin=0,
            xmax=max(x),
            color=color,
            linestyle="-.",
            linewidth=2,
        )
    if show_ylabel:
        ax.set_ylabel("Joint angle [rad]")
    if show_legend:
        ax.legend(fontsize=8, ncol=1, loc="upper left")
    ax.set_title(f"{group_name} joints. (Solid - Position, Dashed - Action)")#, Dotted - Default)")
    ax.grid(True)

def plot_jointvel_subplot(ax, x, joint_names, group_name, joint_vel, show_ylabel=True, show_legend=True):
    "Plot joint vel groupwise on ax."
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]

        ax.plot(
            x,
            joint_vel[:, idx],
            color=color,
            linestyle="-",
            linewidth=2,
            label=joint
        )
        # ax.plot(
        #     x,
        #     shifted_action[:, idx],
        #     color=color,
        #     linestyle="--",
        #     linewidth=2,
        # )

    if show_ylabel:
        ax.set_ylabel("Joint vel [rad/s]")
    if show_legend:
        ax.legend(fontsize=8, ncol=1, loc="upper left")
    ax.set_title(f"{group_name} joint vels. (Solid - Vel)")
    ax.grid(True)


def plot_basevel_subplot(ax, x, base_vel, cmd_vel, show_legend=True):
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

    ax.set_xlabel("Timestep at 56 Hz")
    ax.set_ylabel("Velocity magnitude [m/s] or [rad/s]")
    ax.set_title("Base linear velocity vs commanded. (Solid - Base, Dashed - Commanded)")
    ax.grid(True)
    if show_legend:
        ax.legend(fontsize=8, loc="upper left")

def plot_jointtorque_subplot(ax, x, joint_names, group_name, joint_torque, show_ylabel=True, show_legend=True):
    """Plot joint torque groupwise on ax."""
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]

        ax.plot(
            x,
            joint_torque[:, idx],
            color=color,
            linestyle="-",
            linewidth=2,
            label=joint
        )

    if show_ylabel:
        ax.set_ylabel("Joint torque [Nm]")
    if show_legend:
        ax.legend(fontsize=8, ncol=1, loc="upper left")
    ax.set_title(f"{group_name} joint torques.")#, Dotted - Default)")
    ax.grid(True)

def plot_all_groups(data):
    """Create all group plots."""
    joint_pos = data["joint_positions:"]
    joint_vel = data["joint_velocity:"]
    shifted_action = data["shifted_action:"]
    base_vel = data["base_linear_velocity:"]
    cmd_vel = data["commanded_vel:"]
    joint_torques = data["joint_torques:"]
    x = np.arange(joint_pos.shape[0])
    figures = []

    used_dof = joint_pos.shape[1]
    active_joint_groups = {
        name: joints 
        for name, joints in JOINT_GROUPS.items() 
        if all(JOINT_LABEL_TO_IDX[joint] < used_dof for joint in joints) 
    }
    # Plot joint pos
    # for group_name, joint_names in active_joint_groups.items():
    #     fig, (jointpos_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2,1]}) 
    #     plot_jointpos_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action)
    #     plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel)
    #     plt.tight_layout()
    #     figures.append((f"{group_name}_joint_pos", fig))

    # Plot joint vels
    # for group_name, joint_names in JOINT_GROUPS.items():
    #     fig, (jointvel_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3,1]}) 
    #     plot_jointvel_subplot(jointvel_ax, x, joint_names, group_name, joint_vel)
    #     plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel)
    #     plt.tight_layout()
    #     figures.append((f"{group_name}_joint_vel", fig))
    
    # Plot joint torques with pose and cmd vels
    for group_name, joint_names in active_joint_groups.items():
        fig, (jointpos_ax, base_vel_ax, joint_torques_ax) = plt.subplots(3, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [2,1,1]})
        plot_jointpos_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action)
        plot_basevel_subplot(base_vel_ax, x, base_vel, cmd_vel)
        plot_jointtorque_subplot(joint_torques_ax, x, joint_names, group_name, joint_torques)
        plt.tight_layout()
        figures.append((f"{group_name}_joint_torque", fig))

    # Plot togheter
    fig, mosaic_ax = plt.subplot_mosaic([["FL", "FR"], ["HL", "HR"], ["VEL", "VEL"]], figsize=(12,8), sharex=True) 
    for group_name in ["FL", "FR", "HL",  "HR"]:
        plot_jointpos_subplot(mosaic_ax[group_name], x, JOINT_GROUPS[group_name], group_name, joint_pos, shifted_action, show_ylabel=(group_name in ["FL", "HL"]), show_legend=(group_name=="FL"))
    plot_basevel_subplot(mosaic_ax["VEL"], x, base_vel, cmd_vel)
    plt.tight_layout()
    figures.append(("all_joint_pos", fig))

    return figures


def figure_saver(figures: list[tuple[str, plt.Figure]], save_path: str, save_dir: str, save_ending: str, dpi: int = 300):
    base_path = Path(save_path)
    path = base_path / save_dir
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory not found: {path.parent}")
    path.mkdir(exist_ok=True)
    
    for name, fig in figures:
        filename = f"{name}{save_ending}"
        fig.savefig(path / filename, dpi=dpi, bbox_inches="tight")
def extract_single_count_for_key(data, key: str = "shifted_action:", idx: int = 0):
    key_data = data[key]
    val = []
    if key == "joint_positions:":
        val = key_data[idx][:] + DEFAULT_JOINT_POSITION[:len(key_data[idx])]
    else:
        val = [key_data[idx]]
    return val
def print_jp_sa_values(data, index):
    jp_data = data["joint_positions:"]
    sa_data = data["shifted_action:"]

    jp_value = np.array(jp_data[index][:]+DEFAULT_JOINT_POSITION[:len(jp_data[index])])
    sa_value = np.array(sa_data[index])

    print(f"Joint position for idx {index} is:\n {jp_value} rads and:\n {np.degrees(jp_value)} degs")
    print(f"Shifted action for idx {index} is:\n {sa_value} rads and:\n {np.degrees(sa_value)} degs")
    print(f"Delta (SA-JP) for idx {index} is:\n {sa_value-jp_value} rads and:\n {np.degrees(sa_value)-np.degrees(jp_value)} degs")


# ----- MAIN -----
def main():
    # Add filepath to file in logs directly?
    filename = "graph_code\spot_joint_values.txt"#"/home/sundt/thesis/my_spot_thesis/graph_code/spot_joint_values.txt"
    save_path = "/home/sundt/thesis//my_spot_thesis/graph_code/plots_deployment/"
    save_ending = "_deployment_plot.png"
    
    data = load_data(filename)
    validate_data(data)
    print_summary(data)

    #jp = np.array(extract_single_count_for_key(data, key="joint_positions:", idx=3000))
    #sa = np.array(extract_single_count_for_key(data, key="shifted_action:", idx=3000))
    #print_jp_sa_values(data, 1000)
    figures = plot_all_groups(data)

    flag = input("Do you want to save the plots? (y/n): ")
    if flag == "y":
        save_dir = input("Input the name of the directory inside plots_deployment to save the plots in: ").strip()
        figure_saver(figures, save_path, save_dir, save_ending)

    plt.show()



if __name__ == "__main__":
    main()