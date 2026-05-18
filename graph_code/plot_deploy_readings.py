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
    "base_linear_velocity:",
    "joint_positions:",
    "joint_velocity:",
    "shifted_action:",
    "commanded_vel:",
]
OPTIONAL_KEYS = [
    "base_angular_velocity:",
    "projected_gravity:",
    "last_action:",
    "joint_torques:",
]

# This is in orbit order
JOINT_LABELS = [
    "FL_HX", "FR_HX","HL_HX", "HR_HX", 
    "FL_HY", "FR_HY", "HL_HY", "HR_HY",
    "FL_KN", "FR_KN", "HL_KN", "HR_KN",
    "A0_SH0", "A0_SH1", "A0_EL0", "A0_EL1", "A0_WR0", "A0_WR1", "A0_F1X",]
# label -> idx matching
JOINT_LABEL_TO_IDX = {label: i for i, label in enumerate(JOINT_LABELS)}

JOINT_GROUPS = {
    "FL": ["FL_HX", "FL_HY", "FL_KN"],
    "FR": ["FR_HX", "FR_HY", "FR_KN"],
    "HL": ["HL_HX", "HL_HY", "HL_KN"],
    "HR": ["HR_HX", "HR_HY", "HR_KN"],
    "Arm": ["A0_SH0", "A0_SH1", "A0_EL0", "A0_EL1", "A0_WR0", "A0_WR1", "A0_F1X"]
}

GROUP_DISPLAY_NAMES = {
    "FL": "Front Left",
    "FR": "Front Right",
    "HL": "Hind Left",
    "HR": "Hind Right",
    "Arm": "Arm"
}


JOINT_COLOR_MAP = {
    "FL_HX": "tab:blue",    "FL_HY": "tab:red",   "FL_KN": "tab:green",
    "FR_HX": "tab:blue",    "FR_HY": "tab:red",   "FR_KN": "tab:green",
    "HL_HX": "tab:blue",    "HL_HY": "tab:red",   "HL_KN": "tab:green",
    "HR_HX": "tab:blue",    "HR_HY": "tab:red",   "HR_KN": "tab:green",
    "A0_SH0": "tab:blue", "A0_SH1": "tab:orange", "A0_EL0": "tab:green", "A0_EL1": "tab:red", "A0_WR0": "tab:purple", "A0_WR1": "tab:brown", "A0_F1X": "tab:pink",
}

VEL_LABELS = ["X", "Y", "Yaw"]

VEL_COLOR_MAP = {
    "X": "tab:green", "Y": "tab:blue", "Yaw": "tab:red"
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
            if key not in REQUIRED_KEYS and key not in OPTIONAL_KEYS:
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
    for key in REQUIRED_KEYS:
        if key in data:
            sample_count[key] = data[key].shape[0]

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

    print("\nOptional keys:")
    for key in OPTIONAL_KEYS:
        if key in data:
            print(f"  {key} (loaded)")
        else:
            print(f"  {key} (not found)")

    print("\nLoaded keys:")
    for key, arr in data.items():
        print(f"  {key} {arr.shape}")

# ----- PLOTTING -----
def plot_jointpos_subplot(ax, x, joint_names, group_name, joint_pos, shifted_action, show_ylabel=True, show_xlabel=True, show_legend=True):
    """Plot joint pos groupwise on ax."""
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]
        display_name = GROUP_DISPLAY_NAMES[group_name]

        ax.plot(
            x,
            joint_pos[:, idx] + DEFAULT_JOINT_POSITION[idx],
            color=color,
            linestyle="-",
            label=joint
        )
        ax.plot(
            x,
            shifted_action[:, idx],
            color=color,
            linestyle="--",
        )
        # ax.hlines(
        #     y=DEFAULT_JOINT_POSITION[idx],
        #     xmin=0,
        #     xmax=max(x),
        #     color=color,
        #     linestyle=":",
        # )
        # ax.hlines(
        #     y=[UPPER_JOINT_BOUND[idx], LOWER_JOINT_BOUND[idx]],
        #     xmin=0,
        #     xmax=max(x),
        #     color=color,
        #     linestyle="-.",
        # )
    if show_ylabel:
        ax.set_ylabel("Joint angle [rad]")
    if show_xlabel:
        ax.set_xlabel("Timestep at 56 Hz")
    if show_legend:
        ax.legend(loc="upper left")
    ax.set_title(f"{display_name} Joint Positions")# (Solid - Position, Dashed - Action)")#, Dotted - Default)")

def plot_jointvel_subplot(ax, x, joint_names, group_name, joint_vel, show_ylabel=True, show_xlabel=True, show_legend=True):
    "Plot joint vel groupwise on ax."
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]
        display_name = GROUP_DISPLAY_NAMES[group_name]
        ax.plot(
            x,
            joint_vel[:, idx],
            color=color,
            linestyle="-",
            label=joint
        )
        # ax.plot(
        #     x,
        #     shifted_action[:, idx],
        #     color=color,
        #     linestyle="--",
        # )

    if show_ylabel:
        ax.set_ylabel("Joint vel [rad/s]")
    if show_xlabel:
        ax.set_xlabel("Timestep at 56 Hz")
    if show_legend:
        ax.legend(loc="upper left")
    ax.set_title(f"{display_name} Leg Joint Velocities")# (Solid - Vel)")


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
    ax.set_ylabel("Velocity [m/s] or [rad/s]")
    ax.set_title("Measured vs Commanded Velocity")# (Solid - Base, Dashed - Commanded)")
    if show_legend:
        ax.legend(loc="upper left")

def plot_jointtorque_subplot(ax, x, joint_names, group_name, joint_torque, show_ylabel=True, show_xlabel=True, show_legend=True):
    """Plot joint torque groupwise on ax."""
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]
        display_name = GROUP_DISPLAY_NAMES[group_name]

        ax.plot(
            x,
            joint_torque[:, idx],
            color=color,
            linestyle="-",
            label=joint
        )

    if show_ylabel:
        ax.set_ylabel("Joint torque [Nm]")
    if show_xlabel:
        ax.set_xlabel("Timestep at 56 Hz")
    if show_legend:
        ax.legend(loc="upper left")
    ax.set_title(f"{display_name} Joint Torques")#, Dotted - Default)")

def plot_all_groups(data):
    #print(plt.rcParams) 
    plt.rcParams.update({ 
        #Figure 
        "figure.titlesize": 24, #Title over whole figure 
        "figure.dpi": 100, 
        "figure.constrained_layout.use": True, #Automatically adjust spacing between subplots to prevent overlap
        
        "figure.constrained_layout.h_pad": 0.1, #Height padding around figure
        "figure.constrained_layout.w_pad": 0.15, #Width padding around figure

        "figure.constrained_layout.hspace": 0.03, #Height between plots
        "figure.constrained_layout.wspace": 0.03, #Width between plots

        # #Font 
        "font.family": "sans-serif", 
        "font.size": 12, 

        # #Axes 
        "axes.titlesize": 16, #Title of subplot 
        "axes.titleweight": "normal", 
        "axes.labelsize": 14, #x- and ylabel 
        "axes.linewidth": 1, #Border around subplot 

        # #Tick labels 
        "xtick.labelsize": 10, 
        "ytick.labelsize": 10, 
        
        #Legend 
        "legend.fontsize": 10, #legend text size 
        "legend.frameon": False, #Box around legend
        "legend.fancybox": True, 

        # #Lines 
        "lines.linewidth": 1.5, #Thickness of plotted lines

        # #Grid 
        "axes.grid": False, 
        "grid.alpha": 0.3,
        "axes.spines.top": True, 
        "axes.spines.right": True,
    })

    """Create all group plots."""
    base_lin_vel = data["base_linear_velocity:"]
    base_ang_vel = data["base_angular_velocity:"]
    projected_gravity = data["projected_gravity:"]
    cmd_vel = data["commanded_vel:"]
    joint_pos = data["joint_positions:"]
    joint_vel = data["joint_velocity:"]
    last_action = data["last_action:"]
    shifted_action = data["shifted_action:"]

    joint_torques = data.get("joint_torques:")

    base_vel = np.concatenate((base_lin_vel[:, 0:2], base_ang_vel[:, 2:3]), axis=1)
    x = np.arange(joint_pos.shape[0])
    figures = []

    used_dof = joint_pos.shape[1]
    active_joint_groups = {
        name: joints 
        for name, joints in JOINT_GROUPS.items() 
        if all(JOINT_LABEL_TO_IDX[joint] < used_dof for joint in joints) 
    }


    # Plot joint pos
    for group_name, joint_names in active_joint_groups.items():
        fig, (jointpos_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2,1]}) 
        plot_jointpos_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action, show_ylabel=True, show_xlabel=False, show_legend=True)
        plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel, show_legend=True)
        figures.append((f"{group_name}_joint_pos", fig))

    # Plot joint vels
    # for group_name, joint_names in JOINT_GROUPS.items():
    #     fig, (jointvel_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3,1]}) 
    #     plot_jointvel_subplot(jointvel_ax, x, joint_names, group_name, joint_vel)
    #     plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel)
    #     figures.append((f"{group_name}_joint_vel", fig))
    
    # Plot joint torques with pose and cmd vels
    # if joint_torques is not None:
    #     for group_name, joint_names in active_joint_groups.items():
    #         fig, (jointpos_ax, base_vel_ax, joint_torques_ax) = plt.subplots(3, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [2,1,1]})
    #         plot_jointpos_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action)
    #         plot_basevel_subplot(base_vel_ax, x, base_vel, cmd_vel)
    #         plot_jointtorque_subplot(joint_torques_ax, x, joint_names, group_name, joint_torques)
    #         figures.append((f"{group_name}_joint_torque", fig))

    # Plot togheter
    fig, mosaic_ax = plt.subplot_mosaic([["FL", "FR"], ["HL", "HR"]], figsize=(12,8), sharex=True) 
    for group_name in ["FL", "FR", "HL",  "HR"]:
        plot_jointpos_subplot(mosaic_ax[group_name], x, JOINT_GROUPS[group_name], group_name, joint_pos, shifted_action, show_ylabel=False, show_xlabel=False, show_legend=(group_name=="FL"))
    #plot_basevel_subplot(mosaic_ax["VEL"], x, base_vel, cmd_vel)
    fig.supxlabel("Timestep at 56 Hz")
    fig.supylabel("Joint angle [rad]")
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
    filename = "/home/sundt/thesis/my_spot_thesis/graph_code/spot_joint_values.txt"
    save_path = "/home/sundt/thesis/my_spot_thesis/graph_code/plots_deployment/"
    save_ending = "_deployment_plot.pdf"
    
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