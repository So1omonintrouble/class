#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch  
import numpy as np  
import torch.nn as nn
import random  
from IPython import display  
from matplotlib import pyplot as plt  
import torchvision  
import torchvision.transforms as transforms
from pandas import *
get_ipython().run_line_magic('matplotlib', 'inline')
#导入必要的数据包等相关内容


num_inputs = 200
num_examples = 10000
a1 = torch.normal(0.1,1,(10000, num_inputs))  
b1 = torch.ones(10000,1)    

a2 = torch.normal(-0.1,1,(10000, num_inputs))  
b2 = torch.zeros(10000,1) 
class_2_features = torch.cat((a1,a2),dim=0)
class_2_labels = torch.cat((b1,b2))
index = [i for i in range(len(class_2_labels))]
np.random.shuffle(index)
X_train = class_2_features[index,:]
y_train = class_2_labels[index]




#定义数据的读取
def get_data_iter(X_train, y_train, X_valid, y_valid,batch_size):
    train_dataset = torch.utils.data.TensorDataset(X_train,y_train)
    test_dataset = torch.utils.data.TensorDataset(X_valid,y_valid)
    train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_iter, test_iter

class MM(nn.Module):#定义模型训练函数
     def __init__(self,n_feature):
        super(MM,self).__init__()
        self.linear1 = nn.Linear(n_feature,100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100,1)
        self.Sigmoid = nn.Sigmoid()
    
     def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.Sigmoid(x)
        return y
    
#模型训练
#模型训练
def train(train_iter,test_iter,if_reshape,num_epochs,num_inputs,net,loss):
    optimizer = torch.optim.SGD(net.parameters(),lr=0.001)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        ls, count = 0, 0
        if if_reshape ==False:
            for X,y in train_iter:
                l=loss(net(X),y.view(-1,1))
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                ls += l.item()
                count += y.shape[0]
            train_ls.append(ls/count)
            ls, count = 0, 0
            for X,y in test_iter:
                l=loss(net(X),y.view(-1,1))
                ls += l.item()
                count += y.shape[0]
        else:
            for X,y in train_iter:
                X = X.reshape(-1,num_inputs)
                l=loss(net(X),y).sum()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                ls += l.item()
                count += y.shape[0]
            train_ls.append(ls/count)
            ls, count = 0, 0
            for X,y in test_iter:
                X = X.reshape(-1,num_inputs)
                l=loss(net(X),y).sum()
                ls += l.item()
                count += y.shape[0]
        test_ls.append(ls/count)
        print('epoch: %d, train loss: %f, valid loss: %f'%(epoch+1,train_ls[-1],test_ls[-1]))
    return train_ls,test_ls

#获取k折交叉验证某一折的训练集和验证集
def get_kfold_data(k, i, X, y):
    fold_size = X.shape[0]//k
    val_start = i * fold_size
    if i  != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end],y[val_start:val_end]
        X_train = torch.cat((X[0:val_start],X[val_end:]),dim=0)
        y_train = torch.cat((y[0:val_start],y[val_end:]),dim=0)
    else:
        X_valid,y_valid = X[val_start:], y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
    
    return X_train, y_train, X_valid, y_valid
#循环K次，取平均值，内容来源于PPT
def k_fold(k, X_train, y_train,if_reshape,num_epochs,num_inputs,net,loss):
    my_k_train_ls, my_k_valid_ls = [], []
    train_loss_sum, valid_loss_sum = 0, 0
    for i in range(k):
        print('第', i+1, '折验证结果')
        X_train, y_train, X_valid, y_valid = get_kfold_data(k, i, X_train, y_train)
        train_iter, valid_iter = get_data_iter(X_train, y_train, X_valid, y_valid,batch_size=100)
        train_loss, val_loss = train(train_iter,valid_iter,if_reshape,num_epochs,num_inputs,net,loss)
        
        my_k_train_ls.append(train_loss)
        my_k_valid_ls.append(val_loss)
        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
    
    print("最终平均k折交叉验证结果")
    
    print(f'average train loss: {train_loss_sum/k}')
    print(f'average valid loss: {valid_loss_sum/k}')
    
    return my_k_train_ls, my_k_valid_ls

k=10
num_epochs= 20
num_inputs = 200
net = MM(num_inputs)
loss= nn.MSELoss()
if_reshape=False
net = MM(200)

k_train_ls, k_valid_ls = k_fold(k,X_train, y_train,if_reshape,num_epochs,num_inputs ,net,loss)

# 绘图
train_loss, valid_loss = [], []
for i in range(len(k_train_ls)):
    train_loss.append(k_train_ls[i][-1])
    valid_loss.append(k_valid_ls[i][-1])
    
x = np.linspace(0,len(k_train_ls),len(k_train_ls))
plt.plot(x,train_loss,'o-',label='train_loss',linewidth=1.5)
plt.plot(x,valid_loss,'o-',label='valid_loss',linewidth=1.5)
plt.xlabel('K value')
plt.ylabel('loss')
plt.legend()
plt.show()

# 绘制表格
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
randn = np.random.randn
idx = []
#10折交叉验证，20轮
for i in range(1,21):
    idx.append(f'epoch {i}')
#10折交叉验证，20轮
data_train, data_valid = np.zeros((10,20)),np.zeros((10,20))
for i in range(10):
    for j in range(20):
        data_train[i,j], data_valid[i,j] = k_train_ls[i][j], k_valid_ls[i][j] 
       
df = DataFrame(data_train.T, index=idx, columns=['第1折', '第2折', '第3折', '第4折', '第5折','第6折', '第7折', '第8折', '第9折', '第10折'])
                                                
vals = np.around(df.values,7)
fig = plt.figure(figsize=(8,3))
ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])
the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns,
                    colWidths = [0.1]*vals.shape[1], loc='center',cellLoc='center')
the_table.set_fontsize(20)

the_table.scale(2.5,2.58)


# In[ ]:





# In[ ]:




