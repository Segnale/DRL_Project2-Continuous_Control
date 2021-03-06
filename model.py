import torch
import torch.nn as nn 
import torch.nn.functional as F

class Policy(nn.Module):

    def __init__(self, state_size, action_dim, core):
        super(Policy, self).__init__()
        self.state_size = state_size
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_size, core[0])
        self.fc2 = nn.Linear(core[0], core[1])

        self.fc_actor = nn.Linear(core[-1], self.action_dim)
        self.fc_critic = nn.Linear(core[-1], 1)

        self.std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x, action=None):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Actor
        mean = torch.tanh(self.fc_actor(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic
        # State value V(s)
        v = self.fc_critic(x)

        return action, log_prob, dist.entropy(), v

