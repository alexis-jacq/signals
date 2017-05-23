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
        self.amorces = [0.01*torch.rand(len_amorce).double()]
        self.init = True
    def push(self, new_amorce):
        self.amorces.append(new_amorce)
        if len(self.amorces)>self.capacity:
            del self.amorces[0]
        if self.init:
            del self.amorces[0]
            self.init = False
    def sample(self):
        return np.random.choice(self.amorces)

class Elitist():
    def __init__(self, window_size=200, nb_moods=3, batch_size=32, amorce_size=100):
        self.nb_moods = nb_moods
        self.window_size = window_size
        self.batch_size = batch_size
        self.amorce_size = amorce_size
        self.discriminator = Discriminator(window_size,nb_moods).double()
        self.generators = [Generator(window_size).double() for _ in range(nb_moods)]
        self.memories = [ReplayMemory(1000) for _ in range(nb_moods)]
        self.amorces = [ReplayAmorces(100, amorce_size) for _ in range(nb_moods)]
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
        interval = torch.DoubleTensor(interval)
        #TODO: handle multi-dimension signal
        input = Variable(interval[0,:self.window_size], requires_grad=True).unsqueeze(0)
        target = Variable(interval[0,self.window_size:2*self.window_size], requires_grad=False).unsqueeze(0)
        # train discriminator on the last experience:
        self.optimizerD.zero_grad()
        out = self.discriminator(input)
        nout = self.discriminator(target)
        loss = self.criterionD(out, Variable(self.labels[mood], requires_grad=False).double())
        loss.backward()
        self.optimizerD.step()
        # if signal detected --> remember it
        other_moods = set(range(self.nb_moods))-{mood}
        other_losses = [self.criterionD(out, Variable(self.labels[other_mood], requires_grad=False).double()) \
                        for other_mood in other_moods]
        if loss.data[0] < min([other_loss.data[0] for other_loss in other_losses]):
            self.memories[mood].push((input,target))

        out_test = self.discriminator(self.generators[mood](input))
        loss_test = self.criterionD(out_test, Variable(self.labels[mood], requires_grad=False).double())
        other_test_losses = [self.criterionD(out_test, Variable(self.labels[other_mood], requires_grad=False).double()) \
                        for other_mood in other_moods]
        if loss_test.data[0] < min([other_loss_test.data[0] for other_loss_test in other_test_losses]):
            self.amorces[mood].push(input.data[0,-self.amorce_size:])
            
        # train generators on memorized experiences
        for any_mood in range(self.nb_moods):
            if len(self.memories[any_mood].memory)>self.batch_size:
                self.optimizersG[any_mood].zero_grad()
                batchinput, batchtarget = self.memories[any_mood].sample(self.batch_size)
                loss = self.criterionG(self.generators[any_mood](batchinput),batchtarget)
                loss.backward(retain_variables=True)
                self.optimizersG[any_mood].step()
                # also train discriminator:
                self.optimizerD.zero_grad()
                out = self.discriminator(batchinput)
                loss = self.criterionD(out, Variable(self.labels[any_mood], requires_grad=False).double().expand_as(out))
                loss.backward()
                self.optimizerD.step()

    def generate(self, mood, length=400, repeat=False, delay=200):
        signal = 0.01*torch.rand(1,length).double()
        signal[0,self.window_size-self.amorce_size:self.window_size] = self.amorces[mood].sample()
        input = Variable(signal[0,:self.window_size]).unsqueeze(0)
        output = self.generators[mood](input)
        output2 = self.generators[mood](output)
        signal[0,self.window_size:2*self.window_size] = output.data[:,:].squeeze(0)
        signal[0,2*self.window_size:3*self.window_size] = output2.data[:,:].squeeze(0)
        if repeat:
            pass #TODO
        return signal.numpy()
