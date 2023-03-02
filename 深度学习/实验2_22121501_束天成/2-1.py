#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch 
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from torch.nn import init
#导入必要的数据包等其他相关内容
num_input ,num_example = 500,10000
true_w = torch.ones(1,num_input)*0.0056
true_b = 0.028
a = torch.tensor(np.random.normal(0,0.001,size  = (num_example,num_input)),dtype = torch.float32)
b = torch.mm(x_data,true_w.t()) +true_b
b += torch.normal(0,0.001,b.shape)
#定义总的数据集
a1_train = a[:7000]
a1_test  = a[7000:]
b1_train = b[:7000]
b1_test  = b[7000:]
batch_size = 50
#定义相关训练集和其他内容
train_dataset = TensorDataset(a1_train,b1_train)
train_iter = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,## mini batch size
    shuffle = True,#打乱相关数据
    num_workers = 0,# 多线程来读数据,在Windows下需要设置为0
)
#定义相关测试集和其他内容
test_dataset = TensorDataset(a1_test,b1_test)
test_iter = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0,
)

model= nn.Sequential(OrderedDict([
    ('linear1',nn.Linear(num_input,256)),
    ('linear2',nn.Linear(256,128)),
    ('linear3',nn.Linear(128,1)),
])
)

for param in model.parameters():
    init.normal_(param,mean = 0 ,std = 0.001)
#定义部分参数的初始值   
lr = 0.001
loss = nn.MSELoss()
#定义损失函数
optimizer = torch.optim.SGD(model.parameters(),lr)
#定义优化算法
def train(model,train_iter,test_iter,loss,num_epochs,batch_size,lr):
    train_ls,test_ls = [],[]
    for epoch in range(num_epochs):
        train_ls_sum ,test_ls_sum = 0,0
        for x,y in train_iter:
            y_pred = model(x)
            l = loss(y_pred,y)
            optimizer.zero_grad()#梯度先清零，和手动不一样
            l.backward()
            optimizer.step()
            train_ls_sum += l.item()
        for x ,y in test_iter:#训练集函数
            y_pred = model(x)
            l = loss(y_pred,y)
            test_ls_sum +=l.item()
        train_ls.append(train_ls_sum)
        test_ls.append(test_ls_sum)
        print('epoch %d,train_loss %.6f,test_loss %f'%(epoch+1, train_ls[epoch],test_ls[epoch]))
    return train_ls,test_ls
#定义测试函数
num_epochs = 200
train_loss ,test_loss = train(model,train_iter,test_iter,loss,num_epochs,batch_size,lr)
#设置迭代次数等相关参数
x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()

# In[ ]:




