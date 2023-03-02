#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch  
import numpy as np  
import torch.nn as nn
import random  
from IPython import display  
from matplotlib import pyplot as plt  
import torchvision  
import torchvision.transforms as transforms
#导入相关的数据包等
#下载MNIST手写数据集
mnist_train = torchvision.datasets.MNIST(root='./Datasets/MNIST', train=True,
download=True, transform=transforms.ToTensor())#下载相关内容
mnist_test = torchvision.datasets.MNIST(root='./Datasets/MNIST', train=False,
download=True, transform=transforms.ToTensor())
#读取数据
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
num_workers=0)
#把相应的参数放到指定位置
#初始化参数
num_inputs,num_hiddens,num_outputs = 784,256,10
#设置输入、隐藏层和输出层
def Datanum(): 
    A1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens,num_inputs)), dtype=torch.float32)
    B1 = torch.zeros(1, dtype=torch.float32)
    A2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs,num_hiddens)), dtype=torch.float32)
    B2 = torch.zeros(1, dtype=torch.float32)
    params =[A1,B1,A2,B2]
    for param in params:
        param.requires_grad_(requires_grad=True)
    return A1,B1,A2,B2
num_epochs=30
lr = 0.001
#定义手动实现的规则
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob   
#定义drop损失大小
def net(X, is_training=True):
    X = X.view(-1, num_inputs)
    H1 = (torch.matmul(X, A1.t()) + B1).relu()
    if is_training:
        H1 = dropout(H1, drop_prob1)
    return (torch.matmul(H1,A2.t()) + B2).relu()    

#定义相关模型

#定义模型训练函数
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,lr=None,optimizer=None):
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
            l=loss(net(X,is_training=True),y)
            ls += l.item()
            count += y.shape[0]
        test_ls.append(ls)
        print('epoch: %d, train loss: %f, test loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

drop_probs = np.arange(0,0.5,0.1)
Train_ls, Test_ls = [], []
for drop_prob in drop_probs:
    drop_prob1 = drop_prob
    A1,B1,A2,B2 = Datanum()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([A1,B1,A2,B2],lr = 0.001)
    train_ls, test_ls =  train(net,train_iter,test_iter,loss,num_epochs,batch_size,lr,optimizer)   
    Train_ls.append(train_ls)
    Test_ls.append(test_ls)

x = np.linspace(0,len(train_ls),len(train_ls))
plt.figure(figsize=(10,8))
for i in range(0,len(drop_probs)):
    plt.plot(x,Train_ls[i],label= 'drop_prob=%.1f'%(drop_probs[i]),linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
# plt.legend()
plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
plt.title('train loss with dropout')
plt.show()


# In[ ]:





# In[ ]:




