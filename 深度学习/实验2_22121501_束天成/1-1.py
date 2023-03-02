#!/usr/bin/env python
# coding: utf-8

# In[47]:

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision as transforms
import torch.utils.data as Data
import random  
from IPython import display  
from matplotlib import pyplot as plt  
#导入相关的数据包等
#自定义数据包
num_inputs = 500
num_examples = 10000
true_w = torch.ones(500,1)*0.0056
true_b = 0.028
a_features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
b_labels = torch.mm(a_features,true_w) + true_b
b_labels += torch.tensor(np.random.normal(0, 0.01, size=b_labels.size()), dtype=torch.float)
#手动生成相关的训练集，数据集
#服从特定要求数据集
trainfeatures =a_features[:7000]
trainlabels = b_labels[:7000]
print(trainfeatures.shape)
#定义训练集
testfeatures =a_features[7000:]
testlabels = b_labels[7000:]
print(testfeatures.shape)
#定义测试集
#读取数据
batch_size = 50
dataset = Data.TensorDataset(trainfeatures, trainlabels)
# 对训练的数据集进行整合
# 把 dataset 放入 DataLoader
train_iter = Data.DataLoader(
    dataset=dataset, # torch TensorDataset format
    batch_size=batch_size, # mini batch size
    shuffle=True, # 打乱数据 ，训练过程一般需要打乱
    num_workers=0, # 多线程来读数据， 注意在Windows下需要设置为0
)
dataset = Data.TensorDataset(testfeatures, testlabels)
# 对测试的数据集进行整合
# 把 dataset 放入 DataLoader
test_iter = Data.DataLoader(
    dataset=dataset, # torch TensorDataset format
    batch_size=batch_size, # mini batch size
    shuffle=True, # 打乱数据 ，训练过程一般需要打乱
    num_workers=0, # 多线程来读数据， 注意在Windows下需要设置为0
)
num_hiddens,num_outputs = 256,1#设置隐藏层，输出层
W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens,num_inputs)), dtype=torch.float32)
b1 = torch.zeros(1, dtype=torch.float32)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs,num_hiddens)), dtype=torch.float32)
b2 = torch.zeros(1, dtype=torch.float32)
params =[W1,b1,W2,b2]
for param in params:
    param.requires_grad_(requires_grad=True)
#对权重和偏移大小进行整合，设置梯度值   


def relu(x):
    x = torch.max(input=x,other=torch.tensor(0.0))
    return x

def net(X):
    X = X.view((-1,num_inputs))
    H = relu(torch.matmul(X,W1.t())+b1)
    return torch.matmul(H,W2.t())+b2

def squared_loss(y_hat,y):
     return (y_hat-y.view(y_hat.size()))**2/2
#定义最小化均方误差

def SGD(paras,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size
#定义随机梯度下降法
loss = torch.nn.MSELoss()
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        result=model.forward(X)
        acc_sum+=(result.argmax(dim=1)==y).float().sum().item()
        test_1_sum+=loss_func(result,y).item()
        n+=y.shape[0]
        c+=1  
        acc_sum += (net(X)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n,test_1_sum/c
#定义训练精度           
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    train_ls = []  
    test_ls = [] 
    for epoch in range(num_epochs):
        train_l_sum, train_acc_num,n = 0.0,0.0,0
        for X, y in train_iter: # x和y分别是小批量样本的特征和标签
            y_hat = net(X)
            l = loss(y_hat, y.view(-1,1)) # l是有关小批量X和y的损失
            #梯度清零  
            if optimizer is not None:  
                optimizer.zero_grad()  
            elif params is not None and params[0].grad is not None:  
                 for param in params:  
                    param.grad.data.zero_()  
            l.backward() # 小批量的损失对模型参数求梯度  
            if optimizer is None:  
                SGD(params,lr,batch_size)  
            else:  
                optimizer.step() 
        train_labels = trainlabels.view(-1,1)  
        test_labels = testlabels.view(-1,1)  
        train_ls.append(loss(net(trainfeatures),train_labels).item())  
        test_ls.append(loss(net(testfeatures),test_labels).item()) 
        print('epoch %d, train_loss %.6f,test_loss %f'%(epoch+1, train_ls[epoch], test_ls[epoch])) 
    return train_ls,test_ls
lr = 0.05  
num_epochs = 200    
train_loss,test_loss = train(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
#开始训练
x = np.linspace(0,len(train_loss),len(train_loss))  
plt.plot(x,train_loss,label="train_loss",linewidth=1.5)  
plt.plot(x,test_loss,label="test_loss",linewidth=1.5)  
plt.xlabel("epoch")  
plt.ylabel("loss")  
plt.legend()  
plt.show()

# In[ ]:




