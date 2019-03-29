from unityagents import UnityEnvironment
import numpy as np
import model
import my_methods
import torch
from agent_ppo import Agent
from unity_env import UnityEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# env = UnityEnvironment(file_name='Reachers_Windows_x86_64/Reacher.exe')

# brain_name = env.brain_names[0]
# brain = env.brains[brain_name]

# reset the environment
# env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
# num_agents = len(env_info.agents)
# print('Number of agents:', num_agents)

# size of each action
# action_size = brain.vector_action_space_size
# print('Size of each action:', action_size)

# examine the state space
# states = env_info.vector_observations
# state_size = states.shape[1]
# print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
# print('The state for the first agent looks like:', states[0])

env = UnityEnv(env_file='Reachers_Windows_x86_64/Reacher.exe', no_graphics=True)

discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 320
SGD_epoch = 4
seed = 0
# training loop max iterations
episode = 1500

print("\nRunning with: ", device, "\n")

# widget bar to display progress
#!pip install progressbar
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# keep track of progress
rewards = []

policy = model.Policy(env.state_size, env.action_size, seed).to(device)
agent = Agent(env, policy)
# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
# import torch.optim as optim
# optimizer = optim.Adam(policy.parameters(), lr=2e-4)

for e in range(episode):

    agent.step()
    

    if len(agent.episodes_reward) >= 100:
        r = agent.episodes_reward[:-101:-1]
        total_rewards = agent.episodes_reward[:-101:-1]
        rewards.append((agent.steps, min(r), max(r), np.mean(r), np.std(r)))

    # display some progress every 20 iterations
    if (e+1)%20 ==0 :
        print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))

    # update progress widget bar
    timer.update(e+1)

timer.finish()

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({'score': list(zip(*rewards))[3]})
# plot the score moving avarages to reduce the noise\n",
fig = plt.figure(figsize=[10,5])
ax = fig.add_subplot(111)
plt.title("Learning")
plt.plot(np.arange(len(rewards)), df)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
