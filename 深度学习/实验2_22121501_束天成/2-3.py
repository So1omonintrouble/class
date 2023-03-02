#!/usr/bin/env python
# coding: utf-8

# In[4]:

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
#定义输入层，输出层等相关内容
num_input,num_hidden1,num_hidden2,num_output = 28*28,512,256,10

class MM(nn.Module):#定义模型训练函数
    def __init__(self,num_input,num_hidden1,num_hidden2,num_output):
        super(MM,self).__init__()
        self.linear1 = nn.Linear(num_input,num_hidden1)
        self.linear2 = nn.Linear(num_hidden1,num_hidden2)
        self.linear3 = nn.Linear(num_hidden2,num_output)#定义几个神经网络层的关系
    def forward(self,input):
        input = input.view(-1,784)
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)#定义传递关系
        return out
    
net = MM(num_input,num_hidden1,num_hidden2,num_output)
for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.001)
    
def train(net,train_iter,test_iter,loss,num_epochs):
    train_ls,test_ls,train_acc,test_acc = [],[],[],[]
    for epoch in range(num_epochs):
        train_ls_sum,train_acc_sum,n = 0,0,0
        for x,y in train_iter:#定义训练函数模型
            y_pred = net(x)
            l = loss(y_pred,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls_sum +=l.item()
            train_acc_sum += (y_pred.argmax(dim = 1)==y).sum().item()
            n += y_pred.shape[0]
        train_ls.append(train_ls_sum)
        train_acc.append(train_acc_sum/n)
        
        test_ls_sum,test_acc_sum,n = 0,0,0#定义测试函数模型
        for x,y in test_iter:
            y_pred = net(x)
            l = loss(y_pred,y)
            test_ls_sum +=l.item()
            test_acc_sum += (y_pred.argmax(dim = 1)==y).sum().item()
            n += y_pred.shape[0]
        test_ls.append(test_ls_sum)
        test_acc.append(test_acc_sum/n)
        print('epoch %d, train_loss %.6f,test_loss %f, train_acc %.6f,test_acc %f'
              %(epoch+1, train_ls[epoch],test_ls[epoch], train_acc[epoch],test_acc[epoch]))
    return train_ls,test_ls,train_acc,test_acc

#训练次数和学习率
num_epochs = 20
lr = 0.01
loss  = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr=lr)

train_loss,test_loss,train_acc,test_acc = train(net,train_iter,test_iter,loss,num_epochs)

x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()


# In[ ]:




