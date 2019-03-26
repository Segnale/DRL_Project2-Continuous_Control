from unityagents import UnityEnvironment
import numpy as np
import model
import my_methods
import torch

env = UnityEnvironment(file_name='Reachers_Windows_x86_64/Reacher.exe')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
discount_rate = .99
epsilon = 0.1
beta = .01
tmax = 320
SGD_epoch = 4
seed = 0
# training loop max iterations
episode = 500

# widget bar to display progress
#!pip install progressbar
import progressbar as pb
widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# keep track of progress
mean_rewards = []

policy = model.Policy(state_size, action_size, seed).to(device)

# we use the adam optimizer with learning rate 2e-4
# optim.SGD is also possible
import torch.optim as optim

optimizer = optim.Adam(policy.parameters(), lr=1e-4)


for e in range(episode):

    # collect trajectories
    states, actions, rewards = \
        my_methods.collect_trajectories(env, policy, tmax=tmax)

    total_rewards = np.sum(rewards, axis=0)


    # gradient ascent step
    for _ in range(SGD_epoch):

        L = -my_methods.clipped_surrogate(policy, actions, states, rewards,
                                          epsilon=epsilon, beta=beta)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        del L

    # the clipping parameter reduces as time goes on
    epsilon*=.999

    # the regulation term also reduces
    # this reduces exploration in later runs
    beta*=.995

    # get the average reward of the parallel environments
    mean_rewards.append(np.mean(total_rewards))

    # display some progress every 20 iterations
    if (e+1)%20 ==0 :
        print("Episode: {0:d}, score: {1:f}".format(e+1,np.mean(total_rewards)))

    # update progress widget bar
    timer.update(e+1)

timer.finish()
