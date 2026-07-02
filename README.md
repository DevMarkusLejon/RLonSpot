# Master Thesis: Reinforcement Learning — From Simulation to Deployment on the Robot Platform Spot
This master thesis aims to explore the reinforcement learning capabilities of Boston Dynamics Spot attached with its arm payload. The thesis takes advantage of the Spot RL Researcher kit to deploy RL policies trained in Isaac Lab, which takes advantage of the simulation tool Isaac Sim.

This repository contains: code for deployment, code and data used in Isaac Lab, trained policies, logs, and code for plotting.

## Tested setup

The project was developed and tested across several separate systems. Other versions may work, but the setup below describes what was used during the thesis.

### Software used
- Ubuntu 22.04
- Python 3.10
- Boston Dynamics Spot SDK 
- Spot Joint-level API license
- Spot RL Researcher Kit deployment code
- Docker
- NVIDIA Container Toolkit
- Isaac Sim, run in a Docker container
- Isaac Lab, run in a Docker container

### Hardware used
- Boston Dynamics Spot
- Spot arm payload
- LTH's Bokertov server (External computing machine)
- Spot RL Researcher Kit
- NVIDIA Jetson Orin GTX

## How the systems were used

This repository combines files from several parts of the project, but not every part is run directly from this repository.

- Plotting and analysis scripts are run locally from this repository.
- Isaac Lab training and Isaac Sim simulation are run on an external computing machine inside Docker containers.
- Deployment code is run on a Jetson attached on Spot with the Spot RL Researcher Kit.
- Additional Spot SDK code is installed and run outside this repository.

For the requirements of each external system, refer to its official installation guide. The Python requirements for the plotting and analysis scripts in `src/graph_code/` are listed in `src/graph_code/requirements.txt`.

Install the local plotting dependencies with:

```bash
pip install -r src/graph_code/requirements.txt
```

## Installation and external resources

This repository does not contain the full installation instructions for the external tools. Follow the official documentation for each dependency:

### Boston Dynamics Spot SDK

- Spot SDK documentation: https://dev.bostondynamics.com/
- Python SDK quickstart: https://dev.bostondynamics.com/docs/python/quickstart.html
- Spot SDK GitHub repository: https://github.com/boston-dynamics/spot-sdk

The Spot SDK is required for communicating with the real robot.

### Spot RL Researcher Kit and example code

- Spot RL Researcher Kit: https://bostondynamics.com/reinforcement-learning-researcher-kit/
- Spot RL example repository: https://github.com/boston-dynamics/spot-rl-example

The deployment code in this repository is based on the Boston Dynamics Spot RL example repository. The RL Researcher Kit provides the joint-level API access and supporting tools needed for deploying reinforcement learning policies on Spot.

### Isaac Sim

- Isaac Sim documentation: https://docs.isaacsim.omniverse.nvidia.com/latest/
- Isaac Sim container installation: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
- Isaac Sim developer page: https://developer.nvidia.com/isaac-sim

Isaac Sim is the simulator used by Isaac Lab. In this project, Isaac Sim was run using Docker containers on a external computing machine.

### Isaac Lab

- Isaac Lab documentation: https://isaac-sim.github.io/IsaacLab/
- Isaac Lab Docker guide: https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html
- Isaac Lab GitHub repository: https://github.com/isaac-sim/IsaacLab

Isaac Lab is used for training and evaluating reinforcement learning policies in simulation for deployment on Spot. In this project, Isaac Lab was run using Docker containers on a external computing machine.


## Technical manual

Detailed setup instructions are provided in the technical manual located in `docs/`.

- [Technical manual](docs/technical_manual.pdf)

The manual covers the installation of Isaac Sim and Isaac Lab using Docker containers. It also describes how the project code and data were added to Isaac Lab.

The manual also includes installation notes for the Spot RL Researcher Kit and an overview of the changes made to the deployment code based on the Boston Dynamics Spot RL example repository.


## Repository structure

```text
.
├── data/
│   ├── deployment_logfiles/      # Logs recorded from real-world Spot deployment
│   ├── simulation_logfiles/      # Logs from simulation experiments
│   ├── training_logfiles/        # Training logs from Isaac Lab runs
│   └── plotting_images/          # Generated figures used for analysis
│
├── docs/                         # Notes, setup guides, and project documentation
├── media/
│   ├── images/                   # Images of Spot and experiments
│   └── videos/                   # Videos from experiments and demonstrations
│
├── src/
│   ├── deploy_code/              # Code used for deploying policies on Spot
│   └── graph_code/               # Scripts for plotting and analyzing logs
│
├── isaaclab_data/                # Isaac Lab environments, assets, and configuration
├── trained_policies/             # Exported policies and related training parameters
└── README.md
```
