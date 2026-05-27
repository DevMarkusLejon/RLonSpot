import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({ 
    #Figure 
    "figure.titlesize": 24, #Title over whole figure 
    "figure.dpi": 100, 
    "figure.constrained_layout.use": True, #Automatically adjust spacing between subplots to prevent overlap
    
    "figure.constrained_layout.h_pad": 0.05, #Height padding around figure
    "figure.constrained_layout.w_pad": 0.15, #Width padding around figure

    "figure.constrained_layout.hspace": 0.05, #Height between plots
    "figure.constrained_layout.wspace": 0.05, #Width between plots

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

x = np.arange(500)

# Example signals
x_vel = [0]*100 + [1]*100 + [2]*100 + [3]*100 + [0]*100
y_vel = [0]*100 + [-1]*100 + [-2]*100 + [-1]*100 + [0]*100
yaw_vel = [0]*100 + [0.5]*100 + [1.0]*100 + [0.5]*100 + [0]*100

fig, axs = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)

for i, ax in enumerate(axs):
    ax.plot(x, x_vel, label='X', color='green')
    ax.plot(x, y_vel, label='Y', color='blue')
    ax.plot(x, yaw_vel, label='Yaw', color='red')

    ax.set_title(f'Plot {i+1}')

fig.supxlabel('Timestep at 56 Hz')
fig.supylabel('Velocity [m/s] or [rad/s]')
axs[0].legend(loc='upper left')

plt.show()