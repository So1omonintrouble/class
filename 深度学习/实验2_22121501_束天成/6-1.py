#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
batch_size =128
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

#初始化参数
num_inputs,num_hiddens,num_outputs = 784,256,10
def init_param():
    A1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens,num_inputs)), dtype=torch.float32)
    B1 = torch.zeros(1, dtype=torch.float32)
    A2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs,num_hiddens)), dtype=torch.float32)
    B2 = torch.zeros(1, dtype=torch.float32)
    params =[A1,B1,A2,B2]
    for param in params:
        param.requires_grad_(requires_grad=True)
    return A1,B1,A2,B2

def relu(x):
    x = torch.max(input=x,other=torch.tensor(0.0))
    return x

#定义模型
def net(X):
    X = X.view((-1,num_inputs))
    H = relu(torch.matmul(X,A1.t())+B1)
    return torch.matmul(H,A2.t())+B2

#定义交叉熵损失函数
loss = torch.nn.CrossEntropyLoss()

#定义随机梯度下降法
def SGD(paras,lr):
    for param in params:
        param.data -= lr * param.grad
        
#测试集loss
def evaluate_loss(data_iter,net):
    acc_sum,loss_sum,n = 0.0,0.0,0
    for X,y in data_iter:
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
        l = loss(y_hat,y) # l是有关小批量X和y的损失
        loss_sum += l.sum().item()*y.shape[0]
        n+=y.shape[0]
    return acc_sum/n,loss_sum/n

def l2_penalty(w):
    return (w**2).sum()/2
#定义L2范数惩罚项
#定义模型训练函数
def train(net,train_iter,test_iter,loss,num_epochs,batch_size,lr=None,optimizer=None,sxlambda=0):  
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        ls, count = 0, 0
        for X,y in train_iter :
            X = X.reshape(-1,num_inputs)
            l=loss(net(X),y)+ sxlambda*l2_penalty(A1) + sxlambda*l2_penalty(A2)#训练集中加入惩罚项
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            ls += l.item()
            count += y.shape[0]
        train_ls.append(ls)
        ls, count = 0, 0
        for X,y in test_iter:
            X = X.reshape(-1,num_inputs)
            l=loss(net(X),y) +  sxlambda*l2_penalty(A1) + sxlambda*l2_penalty(A2)#测试集中加入惩罚项
            ls += l.item()
            count += y.shape[0]
        test_ls.append(ls)
        print('epoch: %d, train loss: %f, test loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

lr = 0.01
num_epochs = 50
Lamda = [0,0.1,0.2,0.3]#定义所有的惩罚因子数目=
Train_ls, Test_ls = [], []
for lamda in Lamda:
    print("current lambda is %f"%lamda)
    A1,B1,A2,B2 = init_param()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([A1,B1,A2,B2],lr = 0.001)
    train_ls, test_ls = train(net,train_iter,test_iter,loss,num_epochs,batch_size,lr,optimizer,lamda)   
    Train_ls.append(train_ls)#训练集所有的惩罚因子
    Test_ls.append(test_ls)#测试集所有的惩罚因子

x = np.linspace(0,len(Train_ls[1]),len(Train_ls[1]))
plt.figure(figsize=(10,8))
for i in range(0,len(Lamda)):
    plt.plot(x,Train_ls[i],label= f'L2_Regularization:{Lamda [i]}',linewidth=1.5)
    plt.xlabel('different epoch')
    plt.ylabel('loss')
plt.legend(loc=2, bbox_to_anchor=(1.1,1.0),borderaxespad = 0.)
plt.title('train loss')
plt.show()


# In[ ]:




