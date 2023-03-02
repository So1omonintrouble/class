#!/usr/bin/env python
# coding: utf-8

# In[21]:

import torch
import matplotlib.pyplot as plt
import numpy as np 
import random#导入需要的模板
num_inputs = 2#定义输入个数
n_data = torch.ones(20, num_inputs)
x1 = torch.normal(2 * n_data, 1)
y1 = torch.ones(20)
x2 = torch.normal(-2 * n_data, 1)
y2 = torch.zeros(20)
aa = torch.cat((x1, x2), 0).type(torch.FloatTensor)#人工构造数据集
bb = torch.cat((y1, y2), 0).type(torch.FloatTensor)#人工构造数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)#对数据采取随机读取的方法
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)#将w进行初始化
b = torch.zeros(1, dtype=torch.float32)#将b进行初始化
w.requires_grad_(requires_grad=True) 
b.requires_grad_(requires_grad=True)
def logits(X, w, b):  
    y = torch.mm(X, w) + b  
    return  1/(1+torch.pow(np.e,-y))
def logits_loss(y_hat, y): #手动实现二元交叉熵损失函数
    y = y.view(y_hat.size())  
    return  -y.mul(torch.log(y_hat))-(1-y).mul(torch.log(1-y_hat)) 
def sgd(params, lr, batch_size):  
    for param in params:  
        param.data -= lr * param.grad / batch_size
def evaluate_accuracy():  
    acc_sum,n = 0.0,0  
    for X,y in data_iter(batch_size, aa, bb):  
        y_hat = net(X, w, b)  
        y_hat = torch.squeeze(torch.where(y_hat>0.5,torch.tensor(1.0),torch.tensor(0.0)))  
        acc_sum += (y_hat==y).float().sum().item()  
        n+=y.shape[0]  
    return acc_sum/n

lr = 0.0005
num_epochs = 10  
net = logits  
loss = logits_loss  
batch_size = 5  
epochlist = np.arange(1,num_epochs+1)  
losslist = []  
for epoch in range(num_epochs): # 训练模型的迭代周期  
    train_l_num, train_acc_num,n = 0.0,0.0,0    
    for X, y in data_iter(batch_size, aa, bb): # x和y分别是小批量样本的特征和标签  
        y_hat = net(X, w, b)  
        l = loss(y_hat, y).sum()   
        l.backward() # 小批量的损失对模型参数求梯度  
        sgd([w, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数  
        w.grad.data.zero_() # 梯度清零  
        b.grad.data.zero_() # 梯度清零  
        #计算每个epoch的loss  
        train_l_num += l.item()  
        #计算训练样本的准确率  
        y_hat = torch.squeeze(torch.where(y_hat>0.5,torch.tensor(1.0),torch.tensor(0.0)))  
        train_acc_num += (y_hat==y).sum().item()  
        #每一个epoch的所有样本数  
        n+= y.shape[0]  
    test_acc = evaluate_accuracy()  
    print('epoch %d, loss %.4f,train_acc %f,test_acc %f'%(epoch+1,train_l_num/n, train_acc_num/n, test_acc))  
print(aa)
print(bb)
print( '\n', w) 
print( '\n', b)


# In[ ]:




