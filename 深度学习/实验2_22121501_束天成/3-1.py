#!/usr/bin/env python
# coding: utf-8

# In[6]:

import torch
import numpy as np
import random
from IPython import display
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST
from torch import nn
import torch.utils.data as Data
import torch.optim as optim
from torch.nn import init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
#导入相关的数据包等相关材料
#下载MNIST手写数据集
mnist_train = MNIST(
    root='./datasets/MNIST',
    train = True,
    download =True,
    transform=transforms.ToTensor())
#导入训练集所需要的训练数据
mnist_test = MNIST(
    root='./datasets/MNIST',
    train = False,
    download =True,
    transform=transforms.ToTensor())
#导入测试集所需要的测试数据
#读取数据
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

class FlattenLayer(torch.nn.Module):
    def __init__(self):
         super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0],-1)
#定义模型的前向传播过程
#模型定义和参数初始化
num_inputs,num_hiddens,num_outputs = 784,256,10

def use_ReLU():
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs,num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens,num_outputs)
        )
    return net
#实现ReLU激活函数
def use_ELU():
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs,num_hiddens),
        nn.ELU(),
        nn.Linear(num_hiddens,num_outputs)
        )
    return net
#实现ELU激活函数
def use_Sigmoid():
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs,num_hiddens),
        nn.Sigmoid(),
        nn.Linear(num_hiddens,num_outputs)
        )
    return net
#实现Sigmoid激活函数
def init_params(net):
    for params in net.parameters():
        init.normal_(params,mean=0,std=0.01)
    return  torch.optim.SGD(net.parameters(),lr)
#对训练参数进行初始化
#训练次数和学习率
num_epochs = 20
lr = 0.01
#定义交叉熵损失函数
loss = torch.nn.CrossEntropyLoss()

def evaluate_testset(data_iter,net):
    acc_sum,loss_sum,n = 0.0,0.0,0
    for X,y in data_iter:
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
        l = loss_fn(y_hat,y) # l是有关小批量X和y的损失
        loss_sum += l.item()*y.shape[0]
        n+=y.shape[0]
    return acc_sum/n,loss_sum/n
#定义损失函数等相关内容
#定义模型训练函数
def train(model,train_loader,test_loader,loss_fn,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    train_ls = []
    test_ls = []
    for epoch in range(num_epochs): # 训练模型一共需要num_epochs个迭代周期
        train_loss_sum, train_acc_num,total_examples = 0.0,0.0,0
        for x, y in train_loader: # x和y分别是小批量样本的特征和标签
            y_pred = model(x)
            loss = loss_fn(y_pred, y)  #计算损失
            optimizer.zero_grad() # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step() #梯度更新
            total_examples += y.shape[0]
            train_loss_sum += loss.item()
            train_acc_num += (y_pred.argmax(dim=1)==y).sum().item()
        train_ls.append(train_loss_sum)
        test_acc,test_loss = evaluate_testset(test_loader,model)
        test_ls.append(test_loss)
        print('epoch %d, train_loss %.6f,test_loss %f,train_acc %.6f,test_acc %.6f'%(epoch+1, train_ls[epoch],test_ls[epoch],train_acc_num/total_examples,test_acc))
    return  train_ls,test_ls

def show_plots(mytrain_loss,mytest_loss):
    x = np.linspace(0,len(mytrain_loss),len(mytest_loss))
    plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
    plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
     
net = use_ReLU()
optimizer = init_params(net)
train_loss,test_loss = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters,lr,optimizer)
show_plots(train_loss,test_loss )

net = use_ELU()
optimizer = init_params(net)
train_loss,test_loss = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters,lr,optimizer)
show_plots(train_loss,test_loss )

net = use_Sigmoid()
optimizer = init_params(net)
train_loss,test_loss = train(net,train_iter,test_iter,loss,num_epochs,batch_size,net.parameters,lr,optimizer)
show_plots(train_loss,test_loss )


# In[ ]:





# In[ ]:




