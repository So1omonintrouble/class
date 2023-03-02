#!/usr/bin/env python
# coding: utf-8

# In[43]:


import torch  
import numpy as np  
import random  
from IPython import display  
from matplotlib import pyplot as plt  
import torchvision  
import torchvision.transforms as transforms 
mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=True,
download=True, transform=transforms.ToTensor()) 
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=False,  
download=True, transform=transforms.ToTensor())
batch_size = 10  
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,  
num_workers=0)  
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,  
num_workers=0)  
print(test_iter)  
num_inputs = 784  
num_outputs = 10  
  
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)  
b = torch.zeros(num_outputs,dtype = torch.float)  
 
W.requires_grad_(requires_grad = True)  
b.requires_grad_(requires_grad = True) 
def softmax(X):  
    X_exp = X.exp()  
    partition = X_exp.sum(dim = 1, keepdim=True)  
    return X_exp / partition  
def net(X):  
   #torch.mm  矩阵相乘  view（）改变矩阵维度为1行 num_input列  
    f_x = torch.mm(X.view((-1,num_inputs)),W) + b  
    return softmax(f_x)  
def cross_entropy(y_hat, y):  
    return -torch.log(y_hat.gather(1, y.view(-1,1)))  
def sgd(params, lr, batch_size):  
    for param in params:  
         param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data
def evaluate_accuracy(data_iter,net):  
    acc_sum,n = 0.0,0  
    for X,y in data_iter:  
        #print(len(X)) 小批量数据集 每个X中有 256个图像  
        #print((net(X).argmax(dim=1)==y).float().sum().item())  
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()  
        n+=y.shape[0]  
    return acc_sum/n  
def train(net, train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer = None):  
    for epoch in range(num_epochs):  
       #模型训练次数 num_epochs次  
        train_l_num, train_acc_num,n = 0.0,0.0,0  
        for X,y in train_iter:  
           #X 为小批量256个图像 1*28*28 y为标签    
           # 计算X softmax下的值   与损失函数值  
            y_hat = net(X)   
            l = loss(y_hat,y).sum()  
            l.backward()  
            sgd([W, b], lr, batch_size) # 使用小批量随机梯度下降迭代模型参数  
            W.grad.data.zero_() # 梯度清零  
            b.grad.data.zero_()  
            #计算每个epoch的loss  
            train_l_num += l.item()  
            #计算训练样本的准确率  
            train_acc_num += (y_hat.argmax(dim=1)==y).sum().item()  
        #每一个epoch的所有样本数  
            n+= y.shape[0]  
        #计算测试样本的准确率  
            test_acc = evaluate_accuracy(test_iter,net)  
            print('epoch %d, loss %.4f,train_acc %.3f,test_acc %.3f'%(epoch+1,train_l_num/n, train_acc_num/n, test_acc))  
num_epochs ,lr = 30,0.1  
train(net, train_iter, test_iter, cross_entropy, num_epochs,batch_size, [W, b], lr) 


# In[ ]:




