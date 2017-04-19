#!/usr/bin/env python
#coding: utf-8

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Policy(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Policy, self).__init__()

        self.input_size = input_size
        self.nb_action = nb_action

        self.lstm = nn.LSTMCell(self.input_size, self.input_size)
        self.fc = nn.Linear(self.input_size, self.nb_action)
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
        self.cell_state = Variable(torch.zeros(1,self.input_size))
        self.hidden_state = Variable(torch.zeros(1,self.input_size))


class Drqn():
    def __init__(self, input_size=6, nb_action=2, gamma=0.9):
        self.gamma = gamma

        self.model = Policy(input_size, nb_action)
        self.model.initHidden()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        self.last_hidden = self.model.hidden_state
        self.last_cell = self.model.cell_state
        self.last_state = Variable(torch.Tensor(input_size)).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        output, self.model.hidden_state, self.model.cell_state = self.model(state, [self.model.hidden_state, self.model.cell_state])
        probs = F.softmax(output)
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, indice):
        state = self.model.states[indice]
        next_state = self.model.next_states[indice].detach()
        action = self.model.actions[indice]
        reward = self.model.rewards[indice]
        hidden = self.model.hiddens[indice]
        cell = self.model.cells[indice]

        output, next_hidden, next_cell = self.model(state, [hidden, cell])
        value = output[0,action]
        output,_,_ = self.model(next_state, [next_hidden.detach(), next_hidden.detach()])
        #'''
        next_action_probs = F.softmax(output)
        next_action = next_action_probs.multinomial().data[0,0]
        next_value = output[0,next_action]
        '''
        next_value = output.max(1)[0]
        #'''
        expected = self.gamma*next_value + reward
        td_loss = F.smooth_l1_loss(value, expected)

        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = Variable(torch.Tensor(new_signal).float()).unsqueeze(0)

        if np.abs(self.last_reward)>0 or np.random.rand()>0.9 or len(self.model.states)<10:
            self.model.states.append(self.last_state)
            self.model.next_states.append(new_state)
            self.model.rewards.append(self.last_reward)
            self.model.actions.append(self.last_action)
            self.model.hiddens.append(self.last_hidden)
            self.model.cells.append(self.last_cell)

        self.last_hidden = self.model.hidden_state
        self.last_cell = self.model.cell_state
        action = self.select_action(new_state)

        if action==0:
            self.learn(np.random.choice(len(self.model.states)))
        else:
            self.learn(-1)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        return action
