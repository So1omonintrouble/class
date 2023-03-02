#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch  
import numpy as np  
import random  
from torch import nn  
from IPython import display  
from matplotlib import pyplot as plt  
import torchvision  
import torchvision.transforms as transforms  
from torch.nn import init  

mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=True,  
download=True, transform=transforms.ToTensor())  
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=False,  
download=True, transform=transforms.ToTensor())  
#读取数据  
batch_size = 256  
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,  
num_workers=0)  
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,  
num_workers=0)  
print(test_iter)  

#定义输入和输出  
num_inputs = 784  
num_outputs = 10  
#定义网络模型  
class LinearNet(nn.Module):  
    def __init__(self, num_inputs, num_outputs):  
        super(LinearNet, self).__init__()  
        self.linear = nn.Linear(num_inputs, num_outputs)  
    def forward(self, x): # x shape: (batch, 1, 28, 28)  
        y = self.linear(x.view(x.shape[0], -1))  
        return y        
net = LinearNet(num_inputs, num_outputs) 
# 初始化参数w和b  
init.normal_(net.linear.weight, mean=0, std=0.01)  
init.constant_(net.linear.bias, val=0)  
#nn模块实现交叉熵损失函数--包含了softmax函数  
cross_entropy = nn.CrossEntropyLoss()  
#nn模块实现交叉熵损失函数--包含了softmax函数  
cross_entropy = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)  
#测试集准确率  
def evaluate_accuracy(data_iter,net):  
    acc_sum,n = 0.0,0  
    for X,y in data_iter:  
        #print(len(X)) 小批量数据集 每个X中有 256个图像  
        #print((net(X).argmax(dim=1)==y).float().sum().item())  
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()  
        n+=y.shape[0]  
    return acc_sum/n 
def train(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None, lr=None, optimizer=None):  
    for epoch in range(num_epochs):  
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0  
        for X, y in train_iter:  
            y_hat = net(X)  
            l = loss(y_hat, y).sum()  
            optimizer.zero_grad() # 梯度清零  
            l.backward() # 计算梯度  
            optimizer.step()  # 随机梯度下降算法, 更新参数  
            train_l_sum += l.item()  
            #训练集准确率  
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()  
            n += y.shape[0]  
        test_acc = evaluate_accuracy(test_iter, net)  
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))  
num_epochs = 30  
train(net, train_iter, test_iter, cross_entropy, num_epochs,batch_size,None,None, optimizer)  



# In[ ]:




