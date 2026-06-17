import ast
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
SAMPLE_RATE_HZ = 56
X_AXIS_LABEL = "Time [s]"

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
    "FL": "Front Left Leg",
    "FR": "Front Right Leg",
    "HL": "Hind Left Leg",
    "HR": "Hind Right Leg",
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
                print(f"Key: '{key}' not required or optional, skipping.")
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
def make_time_axis(sample_count, sample_rate_hz=SAMPLE_RATE_HZ):
    """Return elapsed time in seconds for evenly sampled log data."""
    return np.arange(sample_count) / sample_rate_hz


def plot_joint_group_subplot(ax, x, joint_names, group_name, joint_data, shifted_action=None, default_jointpos=False, plot_pos_bounds=False, show_ylabel=None, show_xlabel=None, show_legend=True, show_title=None):
    """Plot data for joint group on ax."""
    display_name = GROUP_DISPLAY_NAMES[group_name]
    for joint in joint_names:
        joint_type = joint.split("_")[1]
        idx = JOINT_LABEL_TO_IDX[joint]
        color = JOINT_COLOR_MAP[joint]
        ax.plot(
            x,
            joint_data[:, idx] + DEFAULT_JOINT_POSITION[idx] if default_jointpos else joint_data[:, idx],
            color=color,
            linestyle="-",
            label=joint_type
        )
        if shifted_action is not None:
            ax.plot(x,shifted_action[:, idx],color=color,linestyle="--",)
        if plot_pos_bounds:
            ax.hlines(y=DEFAULT_JOINT_POSITION[idx],xmin=0,xmax=max(x),color=color,linestyle=":",)
            ax.hlines(y=[UPPER_JOINT_BOUND[idx], LOWER_JOINT_BOUND[idx]],xmin=0,xmax=max(x),color=color,linestyle="-.",)

    if show_ylabel is not None:
        ax.set_ylabel(show_ylabel)
    if show_xlabel is not None:
        ax.set_xlabel(show_xlabel)
    if show_legend:
        ax.legend(loc="upper left")
    if show_title is not None:
        ax.set_title(f"{display_name} {show_title}")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))


def plot_basevel_subplot(ax, x, base_vel, cmd_vel, show_legend=True):
    """Plot vel comparison on ax."""
    for i, vel_name in enumerate(VEL_LABELS):
        vel_color = VEL_COLOR_MAP[vel_name] 

        ax.plot(
            x,
            base_vel[:, i],
            color=vel_color,
            linestyle="-",
            label=vel_name
        )

        ax.plot(
            x,
            cmd_vel[:, i],
            color=vel_color,
            linestyle="--",
        )

    ax.set_xlabel(X_AXIS_LABEL)
    ax.set_ylabel("Velocity [m/s] or [rad/s]")
    ax.set_title("Measured vs Commanded Velocity")# (Solid - Base, Dashed - Commanded)")
    if show_legend:
        ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))


def plot_joint_type_subplot(ax, x, joint_type, joint_data, shifted_action=None, default_jointpos=False, show_ylabel=True, show_xlabel=True, show_legend=True, show_title=None):
    """Plot all joints of a type togheter."""
    joint_names = [f"FL_{joint_type}", f"FR_{joint_type}", f"HL_{joint_type}", f"HR_{joint_type}"]
    joint_display_name = {"HX": "Hip X", "HY": "Hip Y", "KN": "Knee"}
    leg_color_map = {"FL": "tab:blue", "FR": "tab:orange", "HL": "tab:green", "HR": "tab:red"}
    for joint in joint_names:
        idx = JOINT_LABEL_TO_IDX[joint]
        leg_prefix = joint.split("_")[0]
        color = leg_color_map[leg_prefix]

        ax.plot(
            x,
            joint_data[:, idx] + DEFAULT_JOINT_POSITION[idx] if default_jointpos else joint_data[:, idx],
            color=color,
            linestyle="-",
            label=leg_prefix,
        )
        if shifted_action is not None:
            ax.plot(
                x,
                shifted_action[:, idx],
                color=color,
                linestyle="--",
            )

    if show_ylabel is not None:
        ax.set_ylabel(show_ylabel)
    if show_xlabel is not None:
        ax.set_xlabel(show_xlabel)
    if show_legend:
        ax.legend(loc="upper left")
    if show_title is not None:
        ax.set_title(f"{joint_display_name[joint_type]} {show_title}")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))

def plot_all_groups(data):
    """Create all group plots and extract the data."""
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
    #Extract x axis for plotting
    x = np.arange(joint_pos.shape[0]) / SAMPLE_RATE_HZ

    figures = []

    active_joint_groups = {
        name: joints 
        for name, joints in JOINT_GROUPS.items() 
        if all(JOINT_LABEL_TO_IDX[joint] < joint_pos.shape[1] for joint in joints) 
    }

    #Plot singular leg group
    if False:
        group_name = "FL"
        joint_names = active_joint_groups[group_name]
        fig, (jointpos_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2,1]}) 
        plot_joint_group_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action, default_jointpos=True, show_ylabel="Joint angle [rad]", show_xlabel=X_AXIS_LABEL, show_legend=True, show_title="Joint Positions")
        plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel, show_legend=True)
        figures.append((f"{group_name}_joint_pos", fig))

    # Plot joint pos for each leg group separately with base vel
    if True:
        for group_name, joint_names in active_joint_groups.items():
            fig, (jointpos_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2,1]}) 
            plot_joint_group_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action, plot_pos_bounds=False, default_jointpos=True, show_ylabel="Joint angle [rad]", show_legend=True, show_title="Joint Positions")
            plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel, show_legend=True)
            figures.append((f"{group_name}_joint_pos", fig))

    # Plot joint vels for each leg group separately with base vel
    if False:
        for group_name, joint_names in JOINT_GROUPS.items():
            fig, (jointvel_ax, basevel_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3,1]}) 
            plot_joint_group_subplot(jointvel_ax, x, joint_names, group_name, joint_vel, show_ylabel="Joint velocity [rad/s]", show_legend=True, show_title="Joint Velocities")
            plot_basevel_subplot(basevel_ax, x, base_vel, cmd_vel, show_legend=True)
            figures.append((f"{group_name}_joint_vel", fig))
    
    # Plot joint torques for each leg group separately with base vel
    if False:
        if joint_torques is not None:
            for group_name, joint_names in active_joint_groups.items():
                fig, (jointpos_ax, base_vel_ax, joint_torques_ax) = plt.subplots(3, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [2,1,1]})
                plot_joint_group_subplot(jointpos_ax, x, joint_names, group_name, joint_pos, shifted_action, default_jointpos=True, show_ylabel="Joint angle [rad]", show_legend=True, show_title="Joint Positions")
                plot_basevel_subplot(base_vel_ax, x, base_vel, cmd_vel)
                plot_joint_group_subplot(joint_torques_ax, x, joint_names, group_name, joint_torques, show_ylabel="Joint torque [Nm]", show_legend=True, show_title="Joint Torques")
                figures.append((f"{group_name}_joint_torque", fig))

    # Plot joint pos groups in a mosaic
    if False:
        fig, mosaic_ax = plt.subplot_mosaic([["FL", "FR"], ["HL", "HR"]], figsize=(12,8), sharex=True) 
        for group_name in ["FL", "FR", "HL", "HR"]:
            plot_joint_group_subplot(mosaic_ax[group_name], x, JOINT_GROUPS[group_name], group_name, joint_pos, shifted_action, default_jointpos=True, show_legend=(group_name=="FL"), show_title="Joint Positions")
        fig.supxlabel(X_AXIS_LABEL)
        fig.supylabel("Joint angle [rad]")
        #fig.suptitle(f"{GROUP_DISPLAY_NAMES[group_name]} Joint Positions")
        figures.append(("all_joint_pos", fig))
    
    #Plot joint type together for all legs
    if False:
        for joint_type in ["HX", "HY", "KN"]:
            fig, ax = plt.subplots(figsize=(12,8))
            plot_joint_type_subplot(ax, x, joint_type, joint_pos, shifted_action, default_jointpos=True, show_ylabel="Joint angle [rad]", show_xlabel=X_AXIS_LABEL, show_legend=True, show_title="Joint Positions")
            figures.append((f"{joint_type}_joint_pos", fig))

    #Return all created figures with their names for saving and showing
    return figures

def plot_multi_datasets(data_a, data_b):
    """Plot two datasets togheter for comparison."""
    figures = []

    # Align data 
    data_a, data_b = align_datasets(data_a, data_b)

    # Extract data
    joint_pos_a = data_a["joint_positions:"]
    shifted_action_a = data_a["shifted_action:"]

    joint_pos_b = data_b["joint_positions:"]
    shifted_action_b = data_b["shifted_action:"]

    x = np.arange(joint_pos_a.shape[0]) / SAMPLE_RATE_HZ

    figures = []

    active_joint_groups = {
        name: joints 
        for name, joints in JOINT_GROUPS.items() 
        if all(JOINT_LABEL_TO_IDX[joint] < joint_pos_a.shape[1] for joint in joints) 
    }

    # Plot all leg groups together for each dataset
    joint_a_color_map = {"HX": "tab:blue", "HY": "tab:red", "KN": "tab:green"}
    joint_b_color_map = {"HX": "tab:cyan", "HY": "tab:orange", "KN": "tab:olive"}
    for group_name, joint_names in active_joint_groups.items():
        if group_name == "Arm":
            continue
        fig, ax = plt.subplots(figsize=(12,8))
        display_name = GROUP_DISPLAY_NAMES[group_name]
        for joint in joint_names:
            joint_type = joint.split("_")[1]
            idx = JOINT_LABEL_TO_IDX[joint]
            color_a = joint_a_color_map[joint_type]
            color_b = joint_b_color_map[joint_type]
            #Plot dataset A
            ax.plot(
                x,
                joint_pos_a[:, idx] + DEFAULT_JOINT_POSITION[idx],
                color=color_a,
                linestyle="-",
                label=f"{joint_type} Deployment"
            )
            ax.plot(
                x,
                shifted_action_a[:, idx],
                color=color_a,
                linestyle="--",
                #label=f"{joint} Shifted Action A"
            )
            #Plot dataset B
            ax.plot(
                x, 
                joint_pos_b[:, idx] + DEFAULT_JOINT_POSITION[idx],
                color=color_b,
                linestyle="-",
                label=f"{joint_type} Simulation"
            )
            ax.plot(
                x,
                shifted_action_b[:, idx],
                color=color_b,
                linestyle="-.",
                #label=f"{joint} Shifted Action B"
            )
        ax.set_xlabel(X_AXIS_LABEL)
        ax.set_ylabel("Joint angle [rad]")
        ax.set_title(f"{display_name} Joint Positions Comparison")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.1f}"))
        
        figures.append((f"{group_name}_joint_pos_comparison", fig))
    return figures

def align_datasets(*datasets, key="joint_positions:"):
    """Trim multiple datasets to the same size based on the length of a specific key."""
    min_length = min(dataset[key].shape[0] for dataset in datasets)
    aligned_datasets = []
    for dataset in datasets:
        aligned_dataset = {k: v[:min_length] for k, v in dataset.items()}
        aligned_datasets.append(aligned_dataset)
    return aligned_datasets

def figure_saver(figures: list[tuple[str, plt.Figure]], save_path: str, save_dir: str, save_ending: str, dpi: int = 300):
    base_path = Path(save_path)
    path = base_path / save_dir
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory not found: {path.parent}")
    path.mkdir(exist_ok=True)
    
    for name, fig in figures:
        filename = f"{name}{save_ending}"
        fig.savefig(path / filename, dpi=dpi)#, bbox_inches="tight")
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
    #print(plt.rcParams) 
    plt.rcParams.update({ 
        #Figure 
        "figure.titlesize": 24, #Title over whole figure 
        "figure.dpi": 100, 
        "figure.constrained_layout.use": True, #Automatically adjust spacing between subplots to prevent overlap
        
        "figure.constrained_layout.h_pad": 0.1, #Height padding around figure
        "figure.constrained_layout.w_pad": 0.15, #Width padding around figure

        "figure.constrained_layout.hspace": 0.05, #Height between plots
        "figure.constrained_layout.wspace": 0.05, #Width between plots

        # #Font 
        "font.family": "sans-serif", 
        "font.size": 12, 

        # #Axes 
        "axes.titlesize": 24, #Title of subplot TEST WITH 20
        "axes.titleweight": "normal", 
        "axes.labelsize": 20, #x- and ylabel TEST WITH 16
        "axes.labelpad": 10, #Distance between axis and label
        "axes.linewidth": 1, #Border around subplot 


        # #Tick labels 
        "xtick.labelsize": 16, #TEST WITH 14
        "ytick.labelsize": 16, #TEST WITH 14
        
        #Legend 
        "legend.fontsize": 12, #legend text size 
        "legend.frameon": True, #Box around legend
        "legend.fancybox": True, 
        "legend.framealpha": 1, #Transparency of legend box
        "legend.facecolor": "white",

        # #Lines 
        "lines.linewidth": 1.5, #Thickness of plotted lines

        # #Grid 
        "axes.grid": False, 
        "grid.alpha": 0.3,
        "axes.spines.top": True, 
        "axes.spines.right": True,
    })

    filename_real = "/local/home/fredrik/thesis/my_spot_thesis/src/graph_code/spot_real_values.txt"
    filename_sim = "/local/home/fredrik/thesis/my_spot_thesis/src/graph_code/spot_sim_values.txt"
    save_path = "/local/home/fredrik/thesis/my_spot_thesis/data/plotting_images/"
    save_ending = "_deployment_plot.pdf"

    # data_sim = load_data(filename_sim)
    # validate_data(data_sim)
    # print_summary(data_sim)

    data_real = load_data(filename_real)
    #data1 = {key: arr[20*56:35*56] for key, arr in data.items()}
    #data2 = {key: arr[42*56:67*56] for key, arr in data.items()}
    #data = {key: np.concatenate((data1[key], data2[key]), axis=0) for key in data1 if key in data2}
    validate_data(data_real)
    print_summary(data_real)
    figures = plot_all_groups(data_real)
    #figures = plot_all_groups(data_sim)
    #figures = plot_multi_datasets(data_real, data_sim)

    flag = input("Do you want to save the plots? (y/n): ")
    if flag == "y":
        save_dir = input("Input the name of the directory inside plots_deployment to save the plots in: ").strip()
        figure_saver(figures, save_path, save_dir, save_ending)

    plt.show()

    #Extract single values for joint positions and shifted action at a specific index for debugging
    #jp = np.array(extract_single_count_for_key(data_real, key="joint_positions:", idx=3000))
    #sa = np.array(extract_single_count_for_key(data_real, key="shifted_action:", idx=3000))
    #print_jp_sa_values(data_real, 1000)


if __name__ == "__main__":
    main()
