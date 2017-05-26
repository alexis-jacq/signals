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
        #self.cv1 = nn.Conv1d(1, 1, 5, 1, 2, 1) # local corrector
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x
        #output = self.cv1(x.unsqueeze(1))
        #return output.squeeze(1)
'''
class Discriminator(nn.Module):
    def __init__(self, size, nb_moods):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(size, 20)
        self.fc2 = nn.Linear(20, nb_moods)
    def forward(self, input):
        x = F.relu(self.fc1(input))
        output = F.sigmoid(self.fc2(x))
        return output
'''
class Discriminator(nn.Module):
    def __init__(self, size, nb_moods, nb_channels):
        super(Discriminator, self).__init__()
        self.h_size = 2#int(np.floor((size-1)/4)+1)
        self.cv = nn.Conv1d(nb_channels, 10, 49, (size-1), 24, 1)
        self.fc = nn.Linear(10*self.h_size, nb_moods)
    def forward(self, input):
        #x = F.relu(self.cv(input.unsqueeze(1)))
        x = F.relu(self.cv(input))
        x = x.view(-1, 10*self.h_size)
        output = F.sigmoid(self.fc(x))
        return output
#'''
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
    def __init__(self, window_size=200, nb_moods=3, nb_channels=3, batch_size=32, amorce_size=100):
        self.nb_moods = nb_moods
        self.nb_channels = nb_channels
        self.window_size = window_size
        self.batch_size = batch_size
        self.amorce_size = amorce_size
        self.discriminator = Discriminator(window_size,nb_moods,nb_channels).double()
        self.generators = [[Generator(window_size).double() for _ in range(nb_moods)] for _ in range(nb_channels)]
        self.memories = [ReplayMemory(10000) for _ in range(nb_moods)]
        self.amorces = [ReplayAmorces(1000, amorce_size) for _ in range(nb_moods)]
        self.criterionG = nn.MSELoss()
        self.criterionD = nn.BCELoss()
        self.optimizersG = [[optim.Adam(G.parameters(),lr = 0.001) for G in self.generators[channel]] for channel in range(nb_channels)]
        self.optimizerD = optim.Adam(self.discriminator.parameters(),lr = 0.001)
        self.labels = []
        for i in range(nb_moods):
            label_i = torch.zeros(1,nb_moods)
            label_i[0,i] = 1
            self.labels.append(label_i)

    def learn(self, interval, mood):
        interval = torch.DoubleTensor(interval)
        input = Variable(interval[:,:self.window_size], requires_grad=True).unsqueeze(0)
        target = Variable(interval[:,self.window_size:2*self.window_size], requires_grad=True).unsqueeze(0)
        # train discriminator on the last experience:
        self.optimizerD.zero_grad()
        out1 = self.discriminator(input)
        out2 = self.discriminator(target)
        loss1 = self.criterionD(out1, Variable(self.labels[mood], requires_grad=False).double())
        loss2 = self.criterionD(out2, Variable(self.labels[mood], requires_grad=False).double())
        loss = loss1+loss2
        loss.backward()
        self.optimizerD.step()
        # if signal detected --> remember it
        other_moods = set(range(self.nb_moods))-{mood}
        other_losses = [self.criterionD(out2, Variable(self.labels[other_mood], requires_grad=False).double()) \
                        for other_mood in other_moods]
        if loss2.data[0]*2 < min([other_loss.data[0] for other_loss in other_losses]): # if target is recognizable
            self.memories[mood].push((input,target.detach()))

        #TODO: do this for all activated moods (+ moods activations)
        generateds = map(lambda (x,c):x.forward(input[:,c,:]), [(self.generators[c][mood],c) for c in range(self.nb_channels)])
        generated = reduce(lambda x,y:torch.cat((x,y),0), generateds).unsqueeze(0)
        out_test = self.discriminator(torch.cat((input[:,:,-25:],generated[:,:,:25]),2))
        loss_test = self.criterionD(out_test, Variable(self.labels[mood], requires_grad=False).double())
        other_test_losses = [self.criterionD(out_test, Variable(self.labels[other_mood], requires_grad=False).double()) \
                        for other_mood in other_moods]
        if loss_test.data[0]*2 < min([other_loss_test.data[0] for other_loss_test in other_test_losses]):
            self.amorces[mood].push(input.data[:,:,-self.amorce_size:])

        # train generators on memorized experiences
        for any_mood in range(self.nb_moods):
            if len(self.memories[any_mood].memory)>self.batch_size:
                batchinput, batchtarget = self.memories[any_mood].sample(self.batch_size)
                for channel in range(self.nb_channels):
                    self.optimizersG[channel][any_mood].zero_grad()
                    loss = self.criterionG(self.generators[channel][any_mood](batchinput[channel,:]),batchtarget[channel,:])
                    loss.backward(retain_variables=True)
                    self.optimizersG[channel][any_mood].step()
                # also train discriminator:
                self.optimizerD.zero_grad()
                out = self.discriminator(batchinput)
                loss = self.criterionD(out, Variable(self.labels[any_mood], requires_grad=False).double().expand_as(out))
                loss.backward()
                self.optimizerD.step()

    def generate(self, mood, length=400, repeat=False, delay=200):
        signal = 0.01*torch.rand(self.nb_channels,length).double()
        signal[:,self.window_size-self.amorce_size:self.window_size] = self.amorces[mood].sample()
        input = Variable(signal[:,:self.window_size]).unsqueeze(0)

        generateds = map(lambda (x,c):x.forward(input[:,c,:]), [(self.generators[c][mood],c) for c in range(self.nb_channels)])
        generated = reduce(lambda x,y:torch.cat((x,y),0), generateds).unsqueeze(0)

        generateds2 = map(lambda (x,c):x.forward(generated[:,c,:]), [(self.generators[c][mood],c) for c in range(self.nb_channels)])
        generated2 = reduce(lambda x,y:torch.cat((x,y),0), generateds2).unsqueeze(0)

        signal[:,self.window_size:2*self.window_size] = generated.data[:,:].squeeze(0)
        signal[:,2*self.window_size:3*self.window_size] = generated2.data[:,:].squeeze(0)
        if repeat:
            pass #TODO generate following signal. if discriminator not ok, first do amorce then add following signal
        return signal.numpy()
