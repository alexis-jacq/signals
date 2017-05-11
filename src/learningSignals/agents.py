#!/usr/bin/env python
#coding: utf-8

import numpy as np
import random

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
    def __init__(self, input_size=6, nb_action=2, gamma=0.1):
        self.gamma = gamma

        self.model = Policy(input_size, nb_action)
        self.model.initHidden()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)

        self.last_hidden = self.model.hidden_state
        self.last_cell = self.model.cell_state
        self.last_state = Variable(torch.Tensor(input_size)).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        output, self.model.hidden_state, self.model.cell_state = self.model(state, [self.model.hidden_state, self.model.cell_state])
        probs = F.softmax(output)
        self.scores = probs
        action = probs.multinomial()
        return action.data[0,0]

    def learn(self, states,next_states,rewards,actions,hiddens,cells):

        output, next_hiddens, next_cells = self.model(states, [hiddens, cells])
        values = output.gather(1, actions.unsqueeze(1)).squeeze(1)
        output,_,_ = self.model(next_states.detach(), [next_hiddens.detach(), next_hiddens.detach()])
        next_values = output.max(1)[0].squeeze(1) * self.gamma
        expected = next_values + rewards
        td_loss = F.smooth_l1_loss(values, expected)

        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = Variable(torch.Tensor(new_signal).float()).unsqueeze(0)

        if self.last_action!=0 or len(self.model.states)<1:#np.abs(self.last_reward)>0 or np.random.rand()>0.9 or len(self.model.states)<100:
            self.model.states.append(self.last_state)
            self.model.next_states.append(new_state)
            self.model.rewards.append(self.last_reward)
            self.model.actions.append(self.last_action)
            self.model.hiddens.append(self.last_hidden)
            self.model.cells.append(self.last_cell)
            if len(self.model.states)>1000:
                del self.model.states[0]
                del self.model.next_states[0]
                del self.model.rewards[0]
                del self.model.actions[0]
                del self.model.hiddens[0]
                del self.model.cells[0]

        self.last_hidden = self.model.hidden_state
        self.last_cell = self.model.cell_state
        action = self.select_action(new_state)

        memory  = zip(self.model.states,self.model.next_states,self.model.rewards,self.model.actions,self.model.hiddens,self.model.cells)
        batch_size = 100
        if len(memory)<batch_size:
            sample = random.sample(memory,len(memory))
        else:
            sample = random.sample(memory,batch_size)
        mem_len = min(len(memory),batch_size)
        batch_states = reduce(lambda x,y: torch.cat([x,y],0),[sample[j][0] for j in range(mem_len)])
        batch_next_states = reduce(lambda x,y: torch.cat([x,y],0),[sample[j][1] for j in range(mem_len)])
        batch_rewards = Variable(torch.Tensor([sample[j][2] for j in range(mem_len)]))
        batch_actions = Variable(torch.LongTensor([sample[j][3] for j in range(mem_len)]))
        batch_hiddens = reduce(lambda x,y: torch.cat([x,y],0),[sample[j][4] for j in range(mem_len)])
        batch_cells = reduce(lambda x,y: torch.cat([x,y],0),[sample[j][5] for j in range(mem_len)])

        self.learn(batch_states,batch_next_states,batch_rewards,batch_actions,batch_hiddens, batch_cells)

        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        return action
