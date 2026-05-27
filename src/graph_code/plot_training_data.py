# 1. We have a bunch of csv files
# 2. We want to plot all of them in different windows or one window with multiple smaller plots.
# 3. y axis - reward. x axis - time steps. but y is also different for all.

#Make a plot function, attributes can be file name/path, axis labels, title

# make window with number of thingys, call plot function for this area.
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
#import numpy as np


@dataclass
class TrainingData:
    name: str
    abr: str
    df: pd.DataFrame
    x_label: str
    y_label: str


def load_csv_files(files: list[str], titles: list[str], abrvs: list[str], x_cols: list[str], y_cols: list[str]) -> list[TrainingData]:
    data = []

    for file, title, abr, x_col, y_col in zip(files, titles, abrvs, x_cols, y_cols):
        df = pd.read_csv(file)
        td = TrainingData(name=title, abr=abr, df=df, x_label=x_col, y_label=y_col)
        data.append(td)

    return data


def add_rolling_smoothing(td: TrainingData, window: int = 50, column: str = "Value"):
    td.df["Rolling Smoothing"] = td.df[column].rolling(window=window, min_periods=1).mean()


def add_ema_smoothing(td: TrainingData, alpha: int = 0.1, column: str = "Value"):
    td.df["EMA Smoothing"] = td.df[column].ewm(alpha=alpha).mean()


def figure_saver(figures: list[tuple[str, plt.Figure]], save_path: str, save_ending: str, dpi: int = 300):
    path = Path(save_path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Parent directory not found: {path.parent}")
    
    path.mkdir(exist_ok=True)

    for abr, fig in figures:
        filename = f"{abr}{save_ending}"
        fig.savefig(path / filename, dpi=dpi, bbox_inches="tight")


def _labeling_helper(ax, y_label: str, x_label: str, title: str, legend_loc: str = "upper left"):
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend()

def _plot_helper(ax, df: pd.DataFrame, name: str | None = None,  x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None, raw_label: bool = True):
    if smoothed_data is not None and smoothed_data in df.columns:
        # Plot raw data at alpha
        ax.plot(df[x_col], df[y_col], alpha = 0.3, label=name + " - Raw" if raw_label is True else None)
        # Plot smoothed data
        ax.plot(df[x_col], df[smoothed_data], label=name + " - " + smoothed_data)

    else:
        # Only plot raw data at full alpha
        ax.plot(df[x_col], df[y_col], alpha = 1, label=name + " Data")

def plot_single_data(td: TrainingData, x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None):
    df = td.df
    fig, ax = plt.subplots(figsize=[12, 6])
    _plot_helper(ax, df, td.name, x_col, y_col, smoothed_data)
    _labeling_helper(ax, td.y_label, td.x_label, td.name)
    return fig, ax

# def plot_multi_data(td_list: list[TrainingData], x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None):
#     for td in td_list:
#         df = td.df
#         fig, ax = plt.subplots(figsize=[12, 6])
#         _plot_helper(ax, df, td.name, x_col, y_col, smoothed_data)
#         _labeling_helper(ax, td.y_label, td.x_label, td.name)

#     # Show all windows at once
#     plt.show()
#     return fig, ax

def plot_togheter_data(td_list: list[TrainingData], x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None):
    fig, ax = plt.subplots(figsize=[12,6])
    for td in td_list:
        df = td.df
        _plot_helper(ax, df, td.name, x_col, y_col, smoothed_data, raw_label=False)

    # Assume xlabel and ylabel is the same
    _labeling_helper(ax, td_list[0].y_label, td_list[0].x_label, "Terminations")
    # Show all windows at once
    return fig, ax

def main(): 
    #DIR
    #dir = "locomotion_policy_20K_2_phase_lleg_disabled/"
    dir = "Standing_policy/"
    #Saving
    save_path = "/local/home/fredrik/thesis/colcon_ws/src/my_spot_thesis/data/plotting_images/" + dir
    save_ending = "_training_plot.png"

    #Files
    file_path = "/local/home/fredrik/thesis/colcon_ws/src/my_spot_thesis/graph_code/data/training_logfiles/" + dir
    file_ending = ".csv"
    file_names = [
        "rewards/joint_pos_default_tracking",
        "rewards/mean_reward",
        "terminations/termination_body_count",
        "terminations/termination_non_foot_lleg_contact",
        "terminations/termination_timeout"
    ]
    titles = [
        "Joint Pos Default Tracking",
        "Total Mean Reward",
        "Body Contact",
        "Non Foot lleg Contact",
        "Timeout"
    ]
    abreviations = [
        "jpdt",
        "tmr",
        "bc",
        "nflc",
        "to"
    ]
    x_cols = ["Training Step", "Training Step", "Training Step", "Training Step", "Training Step"]
    y_cols = ["Reward Value", "Total Reward", "Terminations %", "Terminations %", "Terminations %"]
    # file_names = [
    #     "rewards/base_angular_velocity",
    #     "rewards/base_linear_velocity",
    #     "rewards/gait_reward",
    #     "stats/mean_reward",
    #     "stats/mean_episode_length",
    #     "stats/terrain_levels",
    #     "terminations/non_foot_lleg_contact_termination",
    #     "terminations/time_out_termination",
    #     "terminations/body_contact_termination"
    # ]
    # titles = [
    #     "Base Angular Velocity",
    #     "Base Linear Velocity",
    #     "Gait",
    #     "Total Mean Reward",
    #     "Mean Episode Length",
    #     "Terrain Level",
    #     "Non Foot lleg Contact",
    #     "Timeout",
    #     "Body Contact"
    # ]
    # abreviations = [
    #     "bav",
    #     "blv",
    #     "gait",
    #     "tmr",
    #     "mel",
    #     "ter_lvl",
    #     "nflc",
    #     "to",
    #     "bc"
    # ]
    # x_cols = ["Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step"]
    # y_cols = ["Reward Value", "Reward Value", "Reward Value", "Total Reward", "Episode Length", "Curriculum Level", "Termination %", "Termination %", "Termination %"]
    

    plot_togheter = [
        "Non Foot lleg Contact",
        "Timeout",
        "Body Contact"
    ]
    files = [f"{file_path}{f}{file_ending}" for f in file_names]
    #Minimal solution to check if the lists are the same length
    if not (len(file_names) == len(titles) == len(abreviations) == len(x_cols) == len(y_cols)):
        raise ValueError(f"All input lists must have the same length. \n file_names:{len(file_names)}, titles:{len(titles)}, x_cols:{len(x_cols)}, y_cols:{len(y_cols)}")

    training_data = load_csv_files(files, titles, abreviations, x_cols, y_cols)
    for data in training_data:
        add_rolling_smoothing(data)
        add_ema_smoothing(data)


    #Plotting and saving figs
    figures = []
    # Plot single data
    for data in training_data:
        if data.name not in plot_togheter:
            fig, _ = plot_single_data(td=data)
            figures.append((data.abr, fig))

    # Combined plot
    combined_data = [data for data in training_data if data.name in plot_togheter]
    fig, _ = plot_togheter_data(td_list=combined_data)#, smoothed_data="EMA Smoothing")
    figures.append(("terminations", fig))

    #Save everything
    flag = input("Do you want to save the figures? (y/n): ")
    if flag == "y":
        figure_saver(figures, save_path, save_ending)
    
    #Show everything
    plt.show()


if __name__ == "__main__":
    main()