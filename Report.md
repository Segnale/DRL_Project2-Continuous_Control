# Deep Reinforcement Learning Project #2 - Report
Here below are reported the main characteristics of the Reinforcement Learning algorithm used to solve the DRLND <em>Reacher</em> environment.

## Learning Algorithm
The selected RL Algorithm was the Proximity Policy Optimization (PPO) and that mainly because it is pretty efficient and relatively easy to implement. Part of the algorithm implementation has been benchmarked from Olivier St-Amand one. You can find a link [here](https://github.com/ostamand/continuous-control).
The core of the algorithm is hold in the `agent_ppo.py` file. The `step` function includes:
* preparation of the trajectories by acting on the environment with the policy provided in the DNN.
* use the trajectory to calculate the Estimated Return using the General Advantage Estimation. Here the article with more details about the [GAE](https://arxiv.org/abs/1506.02438).
* use the updated policy to calculate the surrogate function and clip it to keep proximity.
* Calculate the loss for the actor (surrogate clipped function), the critic (using GEA). 
Actor and Critic Networks are optimized with the Adam optimizer at the end of each cycle.

More details of the PPO can be found [here](https://arxiv.org/abs/1707.06347).

## Model and Parameters
The models of the Critic and Actor are DNN of fully connected layers.
The Actor and Critic holds both two common hidden layers of 125 nodes each. 
The 20 agents are sharing the two networks. That improve training as they share the trajectories and experience accumulated.

Here the summary of parameters used:
  - Learning Rate: `LR` = 1e-4,
  - Number of Updates `epoch` = 10
  - Size of the Batches `batch_size` = 200: 10 for each arm,
  - Discount Rate `GAMMA` = 0.99,
 PPO and GAE:
  - Clipping factor `epsilon_clip` = 0.2 
  - GAE Discount Factor `gae_tau` = 0.95

## Results

The trend below shows the score avarege on 100 episodes achieved by the agent across the 800 episodes.
![Results](results/Training_201904201716.png)

The score of 30 is reached around episode 361 with a stable and alost linear improvement. After 400 episodes the results reach the maximum and constantly around 35 points. The result is limited by the duration of the episode and the exploration feature of the agent.
The performance of the agent after training overtakes the 36 points.


## ToDo list
Few ideas to work out with this project.

### Deeper Analysis of the Results 
The Learning trend shows a constant improvement of the agent score paused by few 'plateau'. Breaking the training process around those areas and observing the behavior of the agent would help understand what changes in the strategy used by the agent to improve the learning further.

### Generalize the Code
I would like to apply the agent to solve wider set of problems. Beside the actual implementation a modular and more generalized implementation of the PPO and GAE algorithm can be tested, and further used for other applications.

### Evolutionary Approach to select Parameters and Architecture
I am becoming a lazy man so; I would use an EA to find the best set of parameters and hyperparameters. The multiple agents environment provided may be leveraged to apply this strategy.
