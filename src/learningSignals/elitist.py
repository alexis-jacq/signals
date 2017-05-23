#!/usr/bin/env python
#coding: utf-8

from __future__ import print_function
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(size, 20)
        self.fc2 = nn.Linear(20, size)
    def forward(self, input):
        x = F.relu(self.fc1(input))
        output = self.fc2(x)
        return output

class Discriminator(nn.Module):
    def __init__(self, size, nb_moods):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(size, 20)
        self.fc2 = nn.Linear(20, nb_moods)
    def forward(self, input):
        x = F.relu(self.fc1(input))
        output = F.sigmoid(self.fc2(x))
        return output

class ReplayMemory(object):
    """ Facilitates memory replay. """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    def push(self, event):
        self.memory.append(event)
        if len(self.memory)>self.capacity:
            del self.memory[0]
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)

class ReplayAmorces(object):
    """ Facilitates memory replay. """
    def __init__(self, capacity,len_amorce):
        self.capacity = capacity
        self.amorces = [torch.Tensor(len_amorce)*0]
    def push(self, new_amorce):
        self.amorces.append(new_amorce)
        if len(self.amorces)>self.capacity:
            del self.amorces[0]
    def sample(self):
        return np.random.choice(self.amorces)

class Elitist():
    def __init__(window_size = 200, nb_moods = 3, batch_size = 32, amorce_size=50):
        self.nb_moods = nb_moods
        self.window_size = window_size
        self.batch_size = batch_size
        self.amorce_size = amorce_size
        self.discriminator = Discriminator(window_size,nb_moods).double()
        self.generators = [Generator(window_size).double() for _ in range(nb_moods)]
        self.memories = [ReplayMemory(100000) for _ in range(nb_moods)]
        self.amorces = [ReplayAmorces(1000, amorce_size) for _ in range(nb_moods)]
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        self.optimizersG = [optim.Adam(G.parameters(),lr = 0.001) for G in self.generators]
        self.optimizerD = optim.Adam(self.discriminator.parameters(),lr = 0.001)
        self.labels = []
        for i in range(nb_moods):
            label_i = torch.zeros(1,nb_moods)
            label_i[0,i] = 1
            self.labels.append(label_i)

    def learn(self, interval, mood):
        interval = torch.Tensor(interval[:2*self.window_size])
        input = Variable(interval[:window_size], requires_grad=True).unsqueeze(0)
        target = Variable(interval[window_size:], requires_grad=False).unsqueeze(0)
        # train discriminator on the last experience:
        out = D(subinput)
        nout = D(subtarget)
        loss = criterionD(out, Variable(self.labels[mood], requires_grad=False).double())
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerD.step()
        # if signal detected --> remember it
        other_moods = set(range(nb_moods))-{mood}
        other_losses = [criterionD(out, Variable(self.labels[other_mood], requires_grad=False).double()) \
                        for other_mood in other_moods]
        if loss.data[0] < min([other_loss.data[0] for other_loss in other_losses]):
            self.memories[mood].push((input,target))
        else:
            nloss = criterion(nout, Variable(self.labels[mood], requires_grad=False).double())
            other_nlosses = [criterionD(nout, Variable(self.labels[other_mood], requires_grad=False).double()) \
                            for other_mood in other_moods]
            if nloss.data[0] < min([other_nloss.data[0] for other_nloss in other_nlosses]):
                self.amorces[mood].push(target.data[0,0:self.amorce_size])
        # train generator of this mood on memorized experiences
        if len(self.memories[mood].memory)>self.batch_size:
            self.optimizersG[mood].zero_grad()
            batchinput, batchtarget = self.memories[mood].sample(self.batch_size)
            loss = criterionG(self.generators[mood](batchinput),batchtarget)
            loss.backward(retain_variables=True)
            self.optimizersG[mood].step()

    def generate(self, mood, length=1000, repeat=False, delay=200):
        signal = torch.DoubleTensor(1,length)*0
        signal[self.window_size-self.amorce_size:self.window_size] = self.amorces[mood].sample()
        input = Variable(signal[:self.window_size]).unsqueeze(0)
        output = self.generators[mood](input).squeeze(0).data[:,:]
        signal[self.window_size:2*self.window_size] = output
        if repeat:
            pass #TODO
        return signal
