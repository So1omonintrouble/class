#!/usr/bin/env python
# coding: utf-8

# In[38]:


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
num_inputs = 200
num_examples = 10000
a1 = torch.normal(0.1,1,(10000, num_inputs))  
b1 = torch.ones(10000,1)    
a1_train = a1[:7000]  
a1_test = a1[7000:] #定义数据集A及其对应的标签 
a2 = torch.normal(-0.1,1,(10000, num_inputs))  
b2 = torch.zeros(10000,1) 
a2_train = a2[:7000]  
a2_test = a2[7000:] #定义数据集B及其对应的标签
trainfeatures = torch.cat((a1_train,a2_train), 0).type(torch.FloatTensor)#合并训练特征集A 
trainlabels = torch.cat((b1[:7000], b2[:7000]), 0).type(torch.FloatTensor)#合并训练特征集A 对应的标签
testfeatures = torch.cat((a1_test,a2_test), 0).type(torch.FloatTensor)  #合并训练特征集B
testlabels = torch.cat((b1[7000:], b2[7000:]), 0).type(torch.FloatTensor)#合并训练特征集B 对应的标签

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

num_hiddens,num_outputs = 256,1#设置隐藏层大小，输出层大小
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

loss = torch.nn.MSELoss()

def SGD(paras,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad/batch_size
#定义随机梯度下降法

def evaluate_accuracy(data_iter,net):  
    acc_sum,test_1_sum,n,c = 0.0,0.0,0,0  
    for X,y in data_iter: 
        acc_sum += (net(X).argmax(dim=1)==y).float().sum().item()  
        n+=y.shape[0]  
    return acc_sum/n
print(trainfeatures)  
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
            train_l_sum += l.item()*y.shape[0]               
            n+= y.shape[0]
        train_acc_num += (y_hat.argmax(dim=1)==y).sum().item()
        test_acc = evaluate_accuracy(test_iter,net)
        test_labels = testlabels.view(-1,1)  
        train_ls.append(train_l_sum/n)  
        test_ls.append(loss(net(testfeatures),test_labels).item())       
        print('epoch %d, train_loss %.6f,test_loss %.6f,train_acc %.3f'%(epoch+1, train_ls[epoch],test_ls[epoch],train_acc_num/n))  
    return train_ls,test_ls
lr = 0.01  
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




