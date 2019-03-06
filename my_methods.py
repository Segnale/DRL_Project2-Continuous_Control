import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_trajectories(env, policy, tmax=200, nrand=5):

        #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    #prob_list=[]
    action_list=[]


    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size

    # number of parallel instances
    num_agents=len(env_info.agents)


    # start all parallel agents
    #env.step([1]*num_agents)

    # perform nrand random steps
    for _ in range(nrand):
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to the environment

    for t in range(tmax):

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        states = env_info.vector_observations               # get next state (for each agent)
        rewards = env_info.rewards                          # get reward (for each agent)

        #states = torch.from_numpy(states)
        actions = policy(torch.tensor(states, dtype=torch.float, device=device)).squeeze().cpu().detach().numpy()#.detach().numpy()                            # Take actions from the policy
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
        state_list.append(torch.from_numpy(states).float().to(device))#state_list.append(states)
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
def clipped_surrogate(policy, actions, states, rewards,
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
    #actions = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)
    states = torch.stack(states)
    # convert states to policy (or probability)
    new_probs = policy(states.view(np.prod(states.shape[:2]),*states.shape[2:])).view(320,20,4)

    # ratio for clipping
    ratio = new_probs/actions

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    rewards = rewards.unsqueeze(2).repeat(1,1,4)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(actions+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-actions+1.e-10))


    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)
