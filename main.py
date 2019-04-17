from unityagents import UnityEnvironment
import numpy as np


env = UnityEnvironment(file_name='Reachers_Windows_x86_64/Reacher.exe')

# get the default brain
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
state = env_info.vector_observations
state_size = state.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(state.shape[0], state_size))
print('The state for the first agent looks like:', state[0])

import torch
from agent_ppo import Agent
from model import Policy

# Agent Parameters and more
episodes = 500
gamma = 0.99
timesteps = 100
ratio_clip = 0.2
batch_size = int(10*20)
epochs = 10
gradient_clip = 10.0
lrate = 1e-4
log_each = 10
beta = 0.01
gae_tau = 0.95
decay_steps = None
solved = 30.0
out_file = 'saved/ppo.ckpt'

#Load

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('Running on:', device)

policy = Policy(state_size, action_size).to(device)

agent = Agent(
    state_size,
    action_size,
    num_agents,
    policy,
    device,
    nsteps = timesteps,
    gamma = gamma,
    epochs = epochs,
    nbatchs = batch_size,
    ratio_clip = ratio_clip,
    lrate = lrate,
    gradient_clip = gradient_clip,
    beta = beta,
    gae_tau = gae_tau
)

from utilities import Plotting
import pdb
import progressbar as pb
from collections import namedtuple, deque

rewards = []
last_saved = 0
scores = deque(maxlen=100)
PScores = []
mean = 0

 # Environment initialization and initial state
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations

# Score Trend Initialization
plot = Plotting(
    title ='Learning Process',
    y_label = 'Score',
    x_label = 'Episode #',
    x_range = 250,
)
plot.show()

# Progress Bar to monitor the training
widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA(), ' Score:']
timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

for episode in range(episodes):
    
    score = np.zeros(num_agents)
    trajectory_raw = []
    
    # Trajectory collection
    for _ in range(timesteps):
        
        action, log_p, value = agent.act(state)
        action = np.clip(action, -1, 1) 
        
        # Environment response
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations         # get next state (for each agent)
        reward = np.array(env_info.rewards)               # get reward (for each agent)
        done = np.array(env_info.local_done)              # see if episode finished
        
        score += reward
        
        # check if some episodes are done
        for i, d in enumerate(done):
            if d:
                scores.append(score[i])                 # collect agent score
                score[i] = 0                            # reset agent score
        
        trajectory_raw.append((agent.tensor_from_np(state), action, reward, log_p, value, 1-done))
        state = next_state
        
    agent.step(state, trajectory_raw)
    
    mean = np.sum(scores)/100
    timer.update(episode+1)
    PScores.append(mean)
    plot.Update(list(range(episode+1)),PScores)
        
    if episode >= 100:
        if (episode+1)%50 == 0 :
            print("Iteration: {0:d}, score: {1:f}".format(episode+1,np.mean(scores)))

    if out_file and mean >= solved and mean > last_saved:
        last_saved = mean
        agent.save(out_file)
        print("saved")

timer.finish()

# Save Training Trend
#  Plotting the entire set of training episodes
end_plot = Plotting(
    title ='Learning Process',
    y_label = 'Score',
    x_label = 'Episode #',
    x_values = list(range(episode+1)),
    y_values = PScores
)
# Create the results directory if missed
dirn = os.path.dirname('results/')
if not os.path.exists(dirn):
    os.mkdir(dirn)

# Save a picture of the Trend
currentDT = datetime.datetime.now()
end_plot.save('results/Training_'+ currentDT.strftime("%Y%m%d%H%M")+'.png')
