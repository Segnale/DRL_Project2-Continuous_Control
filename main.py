from unityagents import UnityEnvironment
import numpy as np

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """Policy Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.out = nn.Softmax()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.out(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_trajectories(env, policy, tmax=200, nrand=5):

        #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    #prob_list=[]
    action_list=[]


    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of parallel instances
    num_agents=len(env_info.agents)

    # start all parallel agents
    #env.step([1]*num_agents)

    # perform nrand random steps
    for _ in range(nrand):
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to the environment

        # @@@@ fr1, re1, _, _ = envs.step(np.random.randn(num_agents, action_size))
        # @@@@ fr2, re2, _, _ = envs.step([0]*n)

    for t in range(tmax):

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        states = env_info.vector_observations               # get next state (for each agent)
        rewards = env_info.rewards                          # get reward (for each agent)
        actions = policy(states)                            # Take actions from the policy
        #@@@@probs = policy(states).squeeze().cpu().detach().numpy()

        #@@@@action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        #@@@@probs = np.where(action==RIGHT, probs, 1.0-probs)


        # advance the game (0=no action)
        # we take one action and skip game forward
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        rewards = env_info.rewards                          # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished

        #@@@@fr1, re1, is_done, _ = envs.step(action)
        #@@@@fr2, re2, is_done, _ = envs.step([0]*n)
        #@@@@reward = re1 + re2

        # store the result
        state_list.append(states)
        reward_list.append(rewards)
        #@@@@prob_list.append(probs)
        action_list.append(actions)

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if np.any(dones):                                  # exit loop if episode finished
            break


    # return pi_theta, states, actions, rewards, probability
    return state_list, action_list, reward_list


# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.float, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = policy(states)

    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))


    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

policy = Policy(state_size, action_size, seed).to(device)

for e in range(episode):

    # collect trajectories
    states, actions, rewards = \
        collect_trajectories(env, policy, tmax=tmax)

    total_rewards = np.sum(rewards, axis=0)


    # gradient ascent step
    for _ in range(SGD_epoch):

        L = -clipped_surrogate(policy, old_probs, states, actions, rewards,
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
        print(total_rewards)

    # update progress widget bar
    timer.update(e+1)

timer.finish()
