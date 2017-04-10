import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Linear(1, 2)

        self.sm = Variable(torch.Tensor([0])) # short memory
        self.lm = Variable(torch.Tensor([0])) # long memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        self.sm = 0.9*self.sm + 0.1*x
        signal = torch.abs(self.sm - self.lm)
        action_scores = self.fc(signal)
        """
        x = F.relu(self.fc1(signal))
        action_scores = self.fc2(signal)
        """
        #self.lm = 0.9*self.lm + 0.1*self.sm
        self.lm = 0.99*self.lm + 0.01*x
        return F.softmax(action_scores)


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(Variable(state))
    action = probs.multinomial()
    return action.data


def rand_learn():
    action = np.random.choice(model.saved_actions)
    reward = np.random.choice(model.rewards)
    action.reinforce(reward)
    optimizer.zero_grad()
    autograd.backward([action], [None])
    optimizer.step()


def learn():
    action = model.saved_actions[-1]
    reward = model.rewards[-1]
    action.reinforce(reward)
    optimizer.zero_grad()
    autograd.backward([action], [None])
    optimizer.step()


for i_episode in count(1):
    state = env.reset()
    for t in range(10000):
        action = select_action(state)
        state, reward = env.step(action.data[0,0])

        if reward>0 or np.random.rand()>0.9:
            model.rewards.append(reward)
            model.saved_actions.append(action)

        if action==0:
            rand_learn()
        else:
            learn()
