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

GAMMA = 0.9

last_state = Variable(torch.Tensor([0,0])).unsqueeze(0)
last_action = 0
last_reward = 0

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.lstm = nn.LSTMCell(2, 2)
        self.fc = nn.Linear(2, 2)
        #self.softmax = nn.LogSoftmax()

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.hiddens = []
        self.cells = []

    def forward(self, input, hidden):
        hx,cx = self.lstm(input,hidden)
        output = self.fc(hx)
        #output = self.softmax(output)
        return output, hx, cx

    def initHidden(self):
        self.cell_state = Variable(torch.zeros(1,2))
        self.hidden_state = Variable(torch.zeros(1,2))

model = Policy()
model.initHidden()
last_hidden = model.hidden_state
last_cell = model.cell_state
optimizer = optim.Adam(model.parameters(), lr=0.01)

def select_action(state):
    output, model.hidden_state, model.cell_state = model(state, [model.hidden_state, model.cell_state])
    print('val '+str(output.data))
    probs = F.softmax(output)
    print('probs '+str(probs.data))
    action = probs.multinomial()
    return action.data[0,0]

def learn(indice):
    state = model.states[indice]
    next_state = model.next_states[indice].detach()
    action = model.actions[indice]
    reward = model.rewards[indice]
    hidden = model.hiddens[indice]
    cell = model.hiddens[indice]

    output, next_hidden, next_cell = model(state, [hidden, cell])
    value = output[0,action]
    output,_,_ = model(next_state, [next_hidden.detach(), next_hidden.detach()])
    #'''
    next_action_probs = F.softmax(output)
    next_action = next_action_probs.multinomial().data[0,0]
    next_value = output[0,next_action]
    '''
    next_value = output.max(1)[0]
    #'''
    expected = GAMMA*next_value + reward
    td_loss = F.smooth_l1_loss(value, expected)

    optimizer.zero_grad()
    td_loss.backward(retain_variables=True)
    optimizer.step()

def update(signal):
    global last_action
    global last_state
    global last_reward
    global last_hidden
    global last_cell

    state = Variable(torch.Tensor([signal,0]).float()).unsqueeze(0)

    if np.abs(last_reward)>0 or np.random.rand()>0.9 or len(model.states)<10:
        model.states.append(last_state)
        model.next_states.append(state)
        model.rewards.append(last_reward)
        model.actions.append(last_action)
        model.hiddens.append(last_hidden)
        model.cells.append(last_cell)

    last_hidden = model.hidden_state
    last_cell = model.cell_state
    action = select_action(state)
    print(action)

    reward = 0
    if action==1 and goal==1:
            reward = 1
    if action==1 and goal==0:
            reward = -1

    if action==0:
        learn(np.random.choice(len(model.states)))
    else:
        learn(-1)

    last_action = action
    last_state = state
    last_reward = reward

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
