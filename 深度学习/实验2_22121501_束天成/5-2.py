#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
import numpy as np
import torch.nn as nn
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms  as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
#导入进行多回归任务所需要的包
mnist_train = MNIST(
    root='./datasets/MNIST',
    train = True,
    download =True,
    transform=transforms.ToTensor())
#导入训练集所需要的测试数据
mnist_test = MNIST(
    root='./datasets/MNIST',
    train = False,
    download =True,
    transform=transforms.ToTensor())
#导入测试集所需要的测试数据
batch_size = 32
# 把训练数据放入 DataLoader 
train_iter = DataLoader(
    dataset = mnist_train,
    batch_size = batch_size,
    shuffle = True,
)
# 把测试数据放入 DataLoader
test_iter = DataLoader(
    dataset = mnist_test,
    batch_size = batch_size,
    shuffle = True,
)


class MM1(nn.Module):    
    def __init__(self,num_inputs, num_outputs, num_hiddens1, num_hiddens2,drop_prob1,drop_prob2):
        super(MM1,self).__init__()
        self.linear1 = nn.Linear(num_inputs,num_hiddens1)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(drop_prob1)
        self.linear2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.drop2 = nn.Dropout(drop_prob2)
        self.linear3 = nn.Linear(num_hiddens2,num_outputs)
        self.flatten  = nn.Flatten()
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.linear3(x)
        y = self.relu(x)
        return y
    
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        ls, count = 0, 0
        for X,y in train_iter:
            l=loss(net(X),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            ls += l.item()
            count += y.shape[0]
        train_ls.append(ls)
        ls, count = 0, 0
        for X,y in test_iter:
            l=loss(net(X),y)
            ls += l.item()
            count += y.shape[0]
        test_ls.append(ls)
        print('epoch: %d, train loss: %f, test loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

num_inputs,num_hiddens1,num_hiddens2,num_outputs =784, 256,256,10
num_epochs=20
lr = 0.001
drop_probs = np.arange(0,0.5,0.1)
Train_ls, Test_ls = [], []
for drop_prob in drop_probs:
    net = MM1(num_inputs, num_outputs, num_hiddens1, num_hiddens2, drop_prob,drop_prob)
    for param in net.parameters():
        nn.init.normal_(param,mean=0, std= 0.01)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr)
    train_ls, test_ls = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters,lr,optimizer)
    Train_ls.append(train_ls)
    Test_ls.append(test_ls)
    
x = np.linspace(0,len(train_ls),len(train_ls))
plt.figure(figsize=(10,8))
for i in range(0,len(drop_probs)):
    plt.plot(x,Train_ls[i],label= 'drop_prob=%.1f'%(drop_probs[i]),linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.title('train loss with dropout')
plt.show()

input = torch.randn(2, 5, 5)
m = nn.Sequential(
nn.Flatten()
)
output = m(input)
output.size()


# In[ ]:




