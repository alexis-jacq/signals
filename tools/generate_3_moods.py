import math
import numpy as np
import torch
T = 20
L = 1000
N = 100
np.random.seed(2)
x = np.empty((3*N, L), 'int64')
z = np.ones((N, L), 'int64')
x[:,:] = np.array(range(L))+ np.random.randint(-4*T, 4*T, 3*N).reshape(3*N, 1)


# mood 1
for k in range(N):
    r = np.array(range(L)) + np.random.randint(-4*T, 4*T)
    z[k,:] = [(i/70)%2 for i in r]
x[:N,:] = z*x[:N,:]

# mood 2
z = z*0
for k in range(N):
    l = np.random.randint(100, 300)
    r = np.random.randint(100, L-l)
    z[k,r:r+100] = 1
x[N:2*N,:] = z*x[N:2*N,:]


# mood 3
z = z*0
x[2*N:3*N,:] = z*x[2*N:3*N,:]

data = (np.sin(x / 1.0 / T)+ (1-2*np.random.rand(3*N,L))/100.).astype('float64')
torch.save(data, open('traindata3.pt', 'wb'))
