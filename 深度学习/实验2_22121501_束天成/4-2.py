#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
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
#读取数据
batch_size =64
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

#训练次数和学习率，迭代次数等相关内容
num_epochs ,lr = 50, 0.01
class MM1(nn.Module):
    def __init__(self,num_inputs=784, num_outputs=10, num_hiddens=100):
        super(MM1,self).__init__()
        self.linear1 = nn.Linear(num_inputs,num_hiddens)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens,num_outputs)    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.relu(x)
        return y
#定义单层神经网络

def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        ls, count = 0, 0
        for X,y in train_iter:
            X = X.reshape(-1,num_inputs)
            l=loss(net(X),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            ls += l.item()
            count += y.shape[0]
        train_ls.append(ls)
        ls, count = 0, 0
        for X,y in test_iter:
            X = X.reshape(-1,num_inputs)
            l=loss(net(X),y)
            ls += l.item()
            count += y.shape[0]
            test_ls.append(ls)
        print('epoch: %d, train loss: %f, test loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

hiddens = [50,100,150]

#定义输入层神经元个数和输出层神经元个数
num_inputs, num_outputs = 784, 10

#定义损失函数
loss = nn.CrossEntropyLoss()
Train_loss, Test_loss = [], []
for cur_hiddens in hiddens:
    net = MM1(num_inputs, num_outputs, cur_hiddens)
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001)
    for param in net.parameters():
        nn.init.normal_(param,mean=0, std= 0.01)
    train_ls, test_ls = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters,lr,optimizer)
    Train_loss.append(train_ls)
    Test_loss.append(test_ls)
    
x = np.linspace(0,len(train_ls),len(train_ls))

plt.figure(figsize=(10,8))
for i in range(0,len(hiddens)):
    plt.plot(x,Train_loss[i],label= f'Neuronss:{ hiddens[i]}',linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
plt.legend()
plt.title('Train loss vs different hiddens')
plt.show()

# In[ ]:




