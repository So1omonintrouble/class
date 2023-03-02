#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
next(iter(train_iter))[0].shape
next(iter(test_iter))[0].shape

#训练次数和学习率
num_epochs ,lr = 50, 0.01
num_inputs, num_outputs = 28*28, 10

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
class MM2(nn.Module):    
    def __init__(self,num_inputs=784, num_outputs=10, num_hiddens1=100, num_hiddens2=100):
        super(MM2,self).__init__()
        self.linear1 = nn.Linear(num_inputs,num_hiddens1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2,num_outputs)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        y = self.relu(x)
        return y
    
class MM3(nn.Module):
    def __init__(self,num_inputs=784, num_outputs=10, num_hiddens1=100, num_hiddens2=100, num_hiddens3=100):
        super(MM3,self).__init__()
        self.linear1 = nn.Linear(num_inputs,num_hiddens1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2,num_hiddens3)
        self.linear4 = nn.Linear(num_hiddens3,num_outputs)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        y = self.relu(x)
        return y
    
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        ls, count = 0, 0
        for X,y in train_iter:
            X = X.reshape(-1,num_inputs)  #[32, 28, 28]  ->  [32, 784]  
            l=loss(net(X),y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            ls += l.item()*y.shape[0]
        train_ls.append(ls)
        ls, count = 0, 0
        for X,y in test_iter:
            X = X.reshape(-1,num_inputs)
            l=loss(net(X),y)
            ls += l.item()*y.shape[0]
        test_ls.append(ls)
        print('epoch: %d, train loss: %f, test loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

total_net = [MM1,MM2,MM3]
Train_loss, Test_loss = [], []
#定义损失函数
loss = nn.CrossEntropyLoss()
for cur_net in total_net:
    net = cur_net()
    for param in net.parameters():
        nn.init.normal_(param,mean=0, std= 0.01)
    optimizer = torch.optim.SGD(net.parameters(),lr = 0.001)
    
    train_ls, test_ls = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters,lr,optimizer)
    Train_loss.append(train_ls)
    Test_loss.append(test_ls)
    
x = np.linspace(0,len(train_ls),len(train_ls))
plt.figure(figsize=(10,8))
for i in range(0,3):
    plt.plot(x,Train_loss[i],label= f'with {i+1} hiddens layers:',linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
plt.legend()
plt.title('train loss')
plt.show()


# In[ ]:




