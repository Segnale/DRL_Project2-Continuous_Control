# DRL_Project2-Continuous_Control

This repository contains the implementation of a PPO Algorithm to train an Agent handling a double joined arm. The simulation environment is provided in Unity [Unity ML-Agents.](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and consists of 20 agents.
This aim to solve the "reacher" problem proposed in the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program by keeping the extremity of the arm in contact with a moving target as much as possible.

---

## Installation

To run this code, you will need to download the prebuild Unity environment not provided in the repository. You need to select the environment for your OS:
* [x Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* [x Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [x Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [x Windows (64-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Place the file in the DRL_Project#2-Continuous_Control Folder and unzip.
The above links are for the case of 20 agents, which is the case solved in the present repository. An enviroment with one single arm can be downloaded from [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

Beside the Unity environment, Python 3.6 must be available with the Unity ML-Agents [(see this link)](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) installed, and few more packages (see `environment.env`).
 
## Environment 

The Agent need to move the two joints of the arm in a 3D domain. That means, 4 continuous actions, two for each joint:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The state space has 33 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.

The Reward function of the Agent can be resumed by:
- +0.039 when the arm touch the target in the simulation time step
- 0 in the case the arm do not touch the target.

The task is episodic, and it is considered solved if the agent can get an average score of +13 over 100 consecutive episodes.
 
## Instructions

Open `Navigation.ipynb` and run the code alongside with the provided instructions.
The notebook is devided in few steps:
1. Starting the Unity environment and setup the default brain to address one agent.
2. Analysis of the State and Action Spaces provided by the Unity environment.
3. Random Agent acting in the enviroment (example of the interaction between an agent and the environment).
4. Initial setup of the parameters of the PPO Algorithm to train the agent.
5. PPO agent training with score plotting.
6. An example of a trained agent for 20 arm reachers.

The file `agent_PPO.py` provides the implementation of the PPO algorithm.
The agent class is included in `dqn_agent.py` while the Deep Neural Network models used by the agent are in the file `model.py`.

More information about the resolution method used can be found in the [`Report.md`](https://github.com/Segnale/DRL_Project1-Navigation/blob/master/Report.md) of this repository.
