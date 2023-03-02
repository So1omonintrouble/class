#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.nn import init
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#导入需要的数据包
num_inputs,num_example = 200,10000
a1 = torch.normal(2,1,(num_example,num_inputs))
b1 = torch.ones((num_example,1))
a2 = torch.normal(-2,1,(num_example,num_inputs))
b2 = torch.zeros((num_example,1))
a = torch.cat((a1,a2),dim=0)
b = torch.cat((b1,b2),dim = 0)
#合并相应的数据集及其标签
a1_train = a[:7000]
a1_test  = a[7000:]
b1_train = b[:7000]
b1_test  = b[7000:]
#分别对应训练集和测试集
#定义总的数据集
batch_size = 256
train_dataset = TensorDataset(a1_train,b1_train)
train_iter = DataLoader(
    dataset = train_dataset,## mini batch size
    shuffle = True,#打乱相关数据
    num_workers = 0,# 多线程来读数据,在Windows下需要设置为0
    batch_size = batch_size
)
test_dataset = TensorDataset(a1_test,b1_test)
test_iter = DataLoader(
    dataset = test_dataset,## mini batch size
    shuffle = True,#打乱相关数据
    num_workers = 0,# 多线程来读数据,在Windows下需要设置为0
    batch_size = batch_size
)

num_input,num_hidden,num_output = 200,256,1#定义模型参数和模型结构等相关内容
class net(nn.Module):
    def __init__(self,num_input,num_hidden,num_output):
        super(net,self).__init__()#定义模型的输入和输出内容等
        self.linear1 = nn.Linear(num_input,num_hidden,bias =False)
        self.linear2 = nn.Linear(num_hidden,num_output,bias=False)
    def forward(self,input):#定义模型的前向传播过程
        out = self.linear1(input)
        out = self.linear2(out)
        return out
    
model = net(num_input,num_hidden,num_output)
print(model)

for param in model.parameters():
    init.normal_(param,mean=0,std=0.001)#模型参数的内容进行初始化
    
lr = 0.001
loss = nn.BCEWithLogitsLoss()#定义损失函数
optimizer = optim.SGD(model.parameters(),lr)#定义优化函数
def train(net,train_iter,test_iter,loss,num_epochs,batch_size):
    train_ls,test_ls,train_acc,test_acc = [],[],[],[]
    for epoch in range(num_epochs):
        train_ls_sum,train_acc_sum,n = 0,0,0
        for x,y in train_iter:#训练集的训练函数
            y_pred = model(x)
            l = loss(y_pred,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls_sum +=l.item()
            train_acc_sum += (((y_pred>0.5)==y)+0.0).sum().item()
            n += y_pred.shape[0]
        train_ls.append(train_ls_sum)
        train_acc.append(train_acc_sum/n)
        
        test_ls_sum,test_acc_sum,n = 0,0,0
        for x,y in test_iter:#测试集的训练函数
            y_pred = model(x)
            l = loss(y_pred,y)
            test_ls_sum +=l.item()
            test_acc_sum += (((y_pred>0.5)==y)+0.0).sum().item()
            n += y_pred.shape[0]
        test_ls.append(test_ls_sum)
        test_acc.append(test_acc_sum/n)
        print('epoch %d, train_loss %.6f,test_loss %f, train_acc %.6f,test_acc %f'
              %(epoch+1, train_ls[epoch],test_ls[epoch], train_acc[epoch],test_acc[epoch]))
    return train_ls,test_ls,train_acc,test_acc
#训练次数和学习率
num_epochs = 200
train_loss,test_loss,train_acc,test_acc = train(model,train_iter,test_iter,loss,num_epochs,batch_size)

x = np.linspace(0,len(train_loss),len(train_loss))
plt.plot(x,train_loss,label="train_loss",linewidth=1.5)
plt.plot(x,test_loss,label="test_loss",linewidth=1.5)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()


# In[ ]:




