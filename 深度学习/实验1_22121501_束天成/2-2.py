#!/usr/bin/env python
# coding: utf-8

# In[40]:


import torch 
from IPython import display 
from matplotlib import pyplot as plt 
import torch.nn as nn
import numpy as np 
import random

lr = 0.03 
import torch.utils.data as Data 
batch_size = 10
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    )

class LinearNet(nn.Module): 
    def __init__(self, n_feature): 
        super(LinearNet, self).__init__() 
        self.linear = nn.Linear(n_feature, 1)
    def forward(self, x): 
        y = self.linear(x) 
        return y 
net =LinearNet(num_inputs)

net = nn.Sequential() 
net.add_module('linear', nn.Linear(num_inputs, 1)) 
net.add_module


from torch.nn import init 
init.normal_(net[0].weight, mean=0, std=0.01) 
init.constant_(net[0].bias, val=0) #也可以直接修改bias的data：net[0].bias.data.fill_(0)
loss = nn.MSELoss()
import torch.optim as optim 
optimizer = optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(1, num_epochs + 1): 
    for X, y in data_iter: 
        output = net(X) 
        l = loss(output, y.view(-1, 1)) 
        optimizer.zero_grad() 
        l.backward() 
        optimizer.step() 
    print('epoch %d, loss: %f' % (epoch, l.item()))


# In[ ]:




