from Tkinter import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

master = Tk()
goal = 0
var_goal = StringVar()

def force_reinforce(var, reward):
    if var.creator.reward is torch.autograd.stochastic_function._NOT_PROVIDED:
        var.creator.reward = reward
    else:
        var.creator.reward += reward

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(2, 2)

        self.sm = Variable(torch.Tensor([0,0])).unsqueeze(0) # short memory
        self.lm = Variable(torch.Tensor([0,0])).unsqueeze(0) # long memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        self.sm = 0.5*self.sm + 0.5*x
        signal = torch.abs(self.sm - self.lm)
        print(signal)

        action_scores = self.fc1(signal)

        self.lm = 0.99*self.lm + 0.01*x

        return F.softmax(action_scores)

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

def select_action(state):
    probs = model(Variable(state))
    print(probs)
    action = probs.multinomial()
    return action


def rand_learn():
    indice = np.random.choice(range(len(model.rewards)))
    action = model.saved_actions[indice]
    reward = model.rewards[indice]
    force_reinforce(action,reward)
    optimizer.zero_grad()
    autograd.backward([action], [None], retain_variables=True)
    optimizer.step()


def learn():
    action = model.saved_actions[-1]
    reward = model.rewards[-1]
    force_reinforce(action,reward)
    optimizer.zero_grad()
    autograd.backward([action], [None], retain_variables=True)
    optimizer.step()

def update(signal):
    signal = torch.Tensor([signal,signal]).float().unsqueeze(0)
    action = select_action(signal)
    print(action.data[0,0])

    reward = 0
    if action.data[0,0]==1 and goal==1:
            reward = 1
    if action.data[0,0]==1 and goal==0:
            reward = -1

    if np.abs(reward)>0 or np.random.rand()>0.9 or len(model.saved_actions)<10:
        model.rewards.append(reward)
        model.saved_actions.append(action)

    if action.data[0,0]==0:
        rand_learn()
    else:
        learn()

def set_goal(new_goal):
    global goal
    goal = new_goal
    print("goal = "+str(goal))
    var_goal.set('goal = '+str(goal))



Button(master, text='S1', height = 10, width = 30, command=lambda:update(0)).grid(row=0, column=0, sticky=W, pady=4)
Button(master, text='S2', height = 10, width = 30, command=lambda:update(1)).grid(row=0, column=1, sticky=W, pady=4)

Button(master, text='goal 0', height = 10, width = 30, command=lambda:set_goal(0)).grid(row=1, column=0, sticky=W, pady=4)
Button(master, text='goal 1', height = 10, width = 30, command=lambda:set_goal(1)).grid(row=1, column=1, sticky=W, pady=4)

Label(master, height = 10, textvariable = var_goal).grid(row=2, sticky=EW, pady=4)

mainloop( )
