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

num_input,num_hidden ,num_output = 784,256,10

def init_w_b():
    A1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    B1 = torch.zeros(num_hiddens, dtype = torch.float)
    A2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    B2 = torch.zeros(num_outputs,dtype=torch.float)

    params = [A1,B1,A2,B2]
    for param in params:
        param.requires_grad_(requires_grad=True)
    return A1,B1,A2,B2#定义惩罚权重因子

class MM1(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens):
        super(MM1,self).__init__()
        self.linear1 = nn.Linear(num_inputs,num_hiddens)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_hiddens,num_outputs)
        self.flatten  = nn.Flatten()
    
    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.relu(x)
        return y#定义神经网络层数等相关内容
    
def train_torch(lamda):#定义训练模型
    num_epochs = 50
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        ls, count = 0, 0
        for X,y in train_iter:
            l=loss(net(X),y)
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
            ls += l.item()
            count += y.shape[0]
        train_ls.append(ls)
        ls, count = 0, 0
        for X,y in test_iter:
            l=loss(net(X),y)
            ls += l.item()
            count += y.shape[0]
        test_ls.append(ls)
        
        print('epoch: %d, train loss: %f, test loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

Lamda = [0,1,2,3,4]
torch_Train_ls, torch_Test_ls = [], []
for lamda in Lamda:
    A1,B1,A2,B2 = init_w_b()
    loss = nn.CrossEntropyLoss()
    net = MM1(num_inputs, num_outputs, num_hiddens)
    optimizer_w = torch.optim.SGD([A1,A2],lr = 0.001,weight_decay=lamda)
    optimizer_b = torch.optim.SGD([B1,B2],lr = 0.001)
    train_ls, test_ls = train_torch(lamda)
    torch_Train_ls.append(train_ls)
    torch_Test_ls.append(test_ls)
    
x = np.linspace(0,len(torch_Train_ls[1]),len(torch_Train_ls[1]))
plt.figure(figsize=(10,8))
for i in range(0,len(Lamda)):
    plt.plot(x,torch_Train_ls[i],label= f'L2_Regularization:{Lamda [i]}',linewidth=1.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
plt.legend(loc=2, bbox_to_anchor=(1.1,1.0),borderaxespad = 0.)
plt.title('train loss')
plt.show()


# In[ ]:




