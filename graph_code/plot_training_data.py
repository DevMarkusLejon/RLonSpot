# 1. We have a bunch of csv files
# 2. We want to plot all of them in different windows or one window with multiple smaller plots.
# 3. y axis - reward. x axis - time steps. but y is also different for all.

#Make a plot function, attributes can be file name/path, axis labels, title

# make window with number of thingys, call plot function for this area.
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np


@dataclass
class TrainingData:
    name: str
    df: pd.DataFrame
    x_label: str
    y_label: str


def load_csv_files(files: list[str], titles: list[str], x_cols: list[str], y_cols: list[str]) -> list[TrainingData]:
    data = []

    for file, title, x_col, y_col in zip(files, titles, x_cols, y_cols):
        df = pd.read_csv(file)
        td = TrainingData(name=title, df=df, x_label=x_col, y_label=y_col)
        data.append(td)

    return data


def add_rolling_smoothing(td: TrainingData, window: int = 50, column: str = "Value"):
    td.df["Rolling Smoothing"] = td.df[column].rolling(window=window, min_periods=1).mean()


def add_ema_smoothing(td: TrainingData, alpha: int = 0.1, column: str = "Value"):
    td.df["EMA Smoothing"] = td.df[column].ewm(alpha=alpha).mean()


def _figure_helper(name: str | None = None, figsize: [float, float] = [12, 8]):
    plt.figure(num=name, figsize=figsize)


def _labeling_helper(y_label: str, x_label: str, title: str, legend_loc: str = "upper left"):
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend(loc=legend_loc)

def _plot_helper(df: pd.DataFrame, name: str | None = None,  x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None, raw_label: bool = True):
    if smoothed_data is not None and smoothed_data in df.columns:
        # Plot raw data at alpha
        plt.plot(df[x_col], df[y_col], alpha = 0.3, label=name + " - Raw" if raw_label is True else None)
        # Plot smoothed data
        plt.plot(df[x_col], df[smoothed_data], label=name + " - " + smoothed_data)

    else:
        # Only plot raw data at full alpha
        plt.plot(df[x_col], df[y_col], alpha = 1, label=name + " Data")

def plot_single_data(td: TrainingData, x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None):
    df = td.df
    _figure_helper(name=td.name)
    _plot_helper(df, td.name, x_col, y_col, smoothed_data)
    _labeling_helper(td.y_label, td.x_label, td.name)
    plt.show()

def plot_multi_data(td_list: list[TrainingData], x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None):
    for td in td_list:
        df = td.df
        _figure_helper(name=td.name)
        _plot_helper(df, td.name, x_col, y_col, smoothed_data)
        _labeling_helper(td.y_label, td.x_label, td.name)

    # Show all windows at once
    plt.show()

def plot_togheter_data(td_list: list[TrainingData], x_col: str = "Step", y_col: str = "Value", smoothed_data: str | None = None):
    _figure_helper(name="Multiple Plots")
    for td in td_list:
        df = td.df
        _plot_helper(df, td.name, x_col, y_col, smoothed_data, raw_label=False)

    # Assume xlabel and ylabel is the same
    _labeling_helper(td_list[0].y_label, td_list[0].x_label, "Terminations")
    # Show all windows at once
    plt.show()

def main(): 
    #Saving
    save_path = "/home/sundt/thesis/colcon_ws/src/my_spot_thesis/spot_deploy_data/plots_training/twophasev2_20k"
    save_ending = "_plot_twophasev2.png"
    save_names = [
        "bav",
        "blv",
        "gait",
        "tmr",
        "mel",
        "ter_lvl",
        "all_terminations"
    ]
    #Files
    file_path = "/home/sundt/thesis/colcon_ws/src/my_spot_thesis/locomotion_policy_20K_2_phase_lleg_disabled/"
    file_ending = ".csv"
    file_names = [
        "rewards/base_angular_velocity",
        "rewards/base_linear_velocity",
        "rewards/gait_reward",
        "stats/mean_reward",
        "stats/mean_episode_length",
        "stats/terrain_levels",
        "terminations/non_foot_lleg_contact_termination",
        "terminations/time_out_termination",
        "terminations/body_contact_termination"
    ]
    titles = [
        "Base Angular Velocity",
        "Base Linear Velocity",
        "Gait",
        "Total Mean Reward",
        "Mean Episode Length",
        "Terrain Level",
        "Non Foot lleg Contact",
        "Time Out",
        "Body Contact"
    ]
    x_cols = ["Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step", "Training Step"]
    y_cols = ["Reward Value", "Reward Value", "Reward Value", "Total Reward", "Episode Length", "Curriculum Level", "Termination %", "Termination %", "Termination %"]
    files = [f"{file_path}{f}{file_ending}" for f in file_names]

    plot_togheter = [
        "Non Foot lleg Contact",
        "Time Out",
        "Body Contact"
    ]
    #Minimal solution to check if the lists are the same length
    if not (len(file_names) == len(titles) == len(x_cols) == len(y_cols)):
        raise ValueError(f"All input lists must have the same length. \n file_names:{len(file_names)}, titles:{len(titles)}, x_cols:{len(x_cols)}, y_cols:{len(y_cols)}")

    training_data = load_csv_files(files, titles, x_cols, y_cols)
    for data in training_data:
        add_rolling_smoothing(data)
        add_ema_smoothing(data)

    #plot_multi_data(td_list=[data for data in training_data if data.name not in plot_togheter], smoothed_data="EMA Smoothing")
    plot_togheter_data(td_list=[data for data in training_data if data.name in plot_togheter], smoothed_data="EMA Smoothing")
    # for data in trainig_data:
    #     plot_single_data(data)
    #     plot_single_data(td=data, smoothed_data="Rolling Smoothing")
    #     plot_single_data(td=data, smoothed_data="EMA Smoothing")

if __name__ == "__main__":
    main()