from collections import deque
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_trajectories(env_info, policy, tmax=200, nrand=5):

    # number of parallel instances
    n=len(env_info.agents)

    #initialize returning lists and start the game!
    state_list=[]
    reward_list=[]
    prob_list=[]
    action_list=[]


    # start all parallel agents
    #envs.step([1]*n)

    # perform nrand random steps
    brain = env.brains[brain_name]
    for _ in range(nrand):
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment

        fr1, re1, _, _ = envs.step(np.random.randn(num_agents, action_size))
        fr2, re2, _, _ = envs.step([0]*n)

    for t in range(tmax):

        # prepare the input


        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        states = env_info.vector_observations               # get next state (for each agent)

        probs = policy(states).squeeze().cpu().detach().numpy()


        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action==RIGHT, probs, 1.0-probs)


        # advance the game (0=no action)
        # we take one action and skip game forward
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment


        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0]*n)

        reward = re1 + re2

        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, \
        action_list, reward_list




def dqn(n_episodes, max_t, eps_start, eps_end, eps_decay, env, agent, brain_name):

    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    previous_scores = 10.0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        #state = env.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            action = action.astype(np.int32)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished


            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)> previous_scores:
            previous_scores = np.mean(scores_window)
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            best_scores = previous_scores
        if i_episode==n_episodes:
            print('\nAgent learning for {:d} episodes!\tBest Average Score: {:.2f}'.format(i_episode, best_scores))
            break
    return scores
