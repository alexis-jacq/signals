from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        self.amorces = [torch.Tensor(len_amorce)]

    def push(self, new_amorce):
        self.amorces.append(new_amorce)
        if len(self.amorces)>self.capacity:
            del self.amorces[0]

    def sample(self):
        return np.random.choice(self.amorces)

if __name__ == '__main__':
    # set ramdom seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    data = torch.load('traindata3.pt')
    input = torch.from_numpy(data[:, :-1])
    N,L = input.size()

    # build the model
    window_size = 200
    G1 = Generator(window_size).double()
    G2 = Generator(window_size).double()
    G3 = Generator(window_size).double()
    D = Discriminator(window_size,3).double()

    memory1 = ReplayMemory(100000)
    memory2 = ReplayMemory(100000)
    memory3 = ReplayMemory(100000)
    batch_size = 32

    criterionG = nn.MSELoss()
    criterion = nn.BCELoss()

    optimizerG1 = optim.Adam(G1.parameters(),lr = 0.001)
    optimizerG2 = optim.Adam(G2.parameters(),lr = 0.001)
    optimizerG3 = optim.Adam(G3.parameters(),lr = 0.001)
    optimizerD = optim.Adam(D.parameters(),lr = 0.001)

    pred = torch.DoubleTensor(N/3,L)
    pred[:,:] = input[1*N/3:2*N/3,:] # mood 2

    y = pred.numpy()
    # draw the result
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(window_size), yi[:window_size], color, linewidth = 2.0)
        plt.plot(np.arange(window_size, L), yi[window_size:], color + ':', linewidth = 2.0)
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    i = -1
    plt.savefig('predict%d.pdf'%i)
    plt.close()

    len_amorce = 50
    amorces1 = ReplayAmorces(1000, len_amorce)
    amorces2 = ReplayAmorces(1000, len_amorce)
    amorces23= ReplayAmorces(1000, len_amorce)

    label1 = torch.zeros(1,3)
    label1[:,0] = 1
    label2 = torch.zeros(1,3)
    label2[:,1] = 1
    label3 = torch.zeros(1,3)
    label3[:,2] = 1

    #begin to train
    for i in range(15):
        print('STEP: ', i)
        for w in range(0,L-2*window_size):
            subinput1 = Variable(input[i,w:w+window_size], requires_grad=True).unsqueeze(0)
            subtarget1 = Variable(input[i,w+window_size:w+2*window_size], requires_grad=False).unsqueeze(0)
            subinput2 = Variable(input[N/3+i,w:w+window_size], requires_grad=True).unsqueeze(0)
            subtarget2 = Variable(input[N/3+i,w+window_size:w+2*window_size], requires_grad=False).unsqueeze(0)
            subinput3 = Variable(input[2*N/3+i,w:w+window_size], requires_grad=True).unsqueeze(0)
            subtarget3 = Variable(input[2*N/3+i,w+window_size:w+2*window_size], requires_grad=False).unsqueeze(0)

            # train discriminator with real:
            out1 = D(subinput1)
            out2 = D(subinput2)
            out3 = D(subinput3)
            nout1 = D(subtarget1)
            nout2 = D(subtarget2)
            nout3 = D(subtarget3)
            loss1 = criterion(out1, Variable(label1, requires_grad=False).double())
            loss2 = criterion(out2, Variable(label2, requires_grad=False).double())
            loss3 = criterion(out3, Variable(label3, requires_grad=False).double())
            loss_real = loss1 + loss2 + loss3
            optimizerD.zero_grad()
            loss_real.backward()
            optimizerD.step()

            loss12 = criterion(out1, Variable(label2, requires_grad=False).double())
            loss13 = criterion(out1, Variable(label3, requires_grad=False).double())
            if loss1.data[0] < min(loss12.data[0],loss13.data[0]):
                memory1.push((subinput1,subtarget1))
            else:
                nloss1 = criterion(nout1, Variable(label1, requires_grad=False).double())
                nloss12 = criterion(nout1, Variable(label2, requires_grad=False).double())
                nloss13 = criterion(nout1, Variable(label3, requires_grad=False).double())
                if nloss1.data[0] < min(nloss12.data[0],nloss13.data[0]):
                    amorces2.push(subtarget1.data[0,0:len_amorce])
            if len(memory1.memory)>batch_size:
                optimizerG1.zero_grad()
                batchinput1, batchtarget1 = memory1.sample(batch_size)
                loss1 = criterionG(G1(batchinput1),batchtarget1)
                loss1.backward(retain_variables=True)
                optimizerG1.step()

            loss21 = criterion(out2, Variable(label1, requires_grad=False).double())
            loss23 = criterion(out2, Variable(label3, requires_grad=False).double())
            if loss2.data[0] < min(loss21.data[0],loss23.data[0]):
                memory2.push((subinput2,subtarget2))
            else:
                nloss2 = criterion(nout2, Variable(label2, requires_grad=False).double())
                nloss21 = criterion(nout2, Variable(label1, requires_grad=False).double())
                nloss23 = criterion(nout2, Variable(label3, requires_grad=False).double())
                if nloss2.data[0] < min(nloss21.data[0],nloss23.data[0]):
                    amorces2.push(subtarget2.data[0,0:len_amorce])
            if len(memory2.memory)>batch_size:
                optimizerG2.zero_grad()
                batchinput2, batchtarget2 = memory2.sample(batch_size)
                loss2 = criterionG(G2(batchinput2),batchtarget2)
                loss2.backward(retain_variables=True)
                optimizerG2.step()

            loss31 = criterion(out3, Variable(label1, requires_grad=False).double())
            loss32 = criterion(out3, Variable(label2, requires_grad=False).double())
            if loss3.data[0] < min(loss31.data[0],loss32.data[0]):
                memory3.push((subinput3,subtarget3))
            else:
                nloss3 = criterion(nout3, Variable(label3, requires_grad=False).double())
                nloss31 = criterion(nout3, Variable(label1, requires_grad=False).double())
                nloss32 = criterion(nout3, Variable(label2, requires_grad=False).double())
                if nloss3.data[0] < min(nloss31.data[0],nloss32.data[0]):
                    amorces3.push(subtarget3.data[0,0:len_amorce])
            if len(memory3.memory)>batch_size:
                optimizerG3.zero_grad()
                batchinput3, batchtarget3 = memory3.sample(batch_size)
                loss3 = criterionG(G3(batchinput3),batchtarget3)
                loss3.backward(retain_variables=True)
                optimizerG3.step()



        # begin to predict
        pred = torch.DoubleTensor(N/3,L)*0
        pred[:,:window_size] = input[1*N/3:2*N/3,:window_size]
        for w in range(0,L-2*window_size-9,window_size+0*np.random.randint(window_size/2,window_size)):

            test = Variable(pred[:,w:w+window_size], requires_grad=False)
            output = G2(test)
            pred[:,w+window_size:w+2*window_size] = output.data[:,:]
            pred[0,w+2*window_size-len_amorce:w+2*window_size] = amorces2.sample()
            pred[1,w+2*window_size-len_amorce:w+2*window_size] = amorces2.sample()
            pred[2,w+2*window_size-len_amorce:w+2*window_size] = amorces2.sample()

        y = pred.numpy()
        # draw the result
        plt.figure(figsize=(30,10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        def draw(yi, color):
            plt.plot(np.arange(window_size), yi[:window_size], color, linewidth = 2.0)
            plt.plot(np.arange(window_size, L), yi[window_size:], color + ':', linewidth = 2.0)
        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()
